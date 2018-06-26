# Standard imports
import copy
import warnings
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import clone as sklearn_clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from collections import Counter

def handle_warnings(*args):
    pass

class CoEnsembling(BaseEstimator, ClassifierMixin):
    """ Co-training wrapper for sklearn classifiers.

    Parameters
    ----------
    base_classifiers : array-like of ClassifierMixin, shape = [n_classifiers]
        List of base classifier instances to use for prediction

    max_iter : int, (default=100)
        Maximum number of iterations to perform the co-training loop. If set to 0, the co-training
        loop will never run and the output will be the same as the base classifiers output

    Attributes
    ----------
    classifiers_ : list of ClassifierMixin
        Classifier instances

    weights : 
        DONT USE

    max_iter : int, (default=100)
        Maximum number of iterations to perform the self-training loop. If set to 0, the
        self-training loop will never run and the output will be the same as the base classifier
        output

    confidence_threshold : float, (default=0.9)
        Specifies the probability at which the wrapper will consider a prediction to be certain
        enough to be added to the training set in the next iteration

    boostrapping : boolean, (default=False)
        Whether to use bootstrapping at the beginning of the co-ensembling algorithm. When enabled,
        samples will be bootstrapped (i.e., n data points will be sampled with replacement) for each
        classifier independently before the co-ensembling algorithm starts.

    max_bootstrapped_samples_per_iter : int, (default=-1)
        If at least 0, this number is used as the maximum number of samples to bootstrap in each
        iteration. For example, if max_bootstrapped_samples_per_iter=10, the 10 samples with the
        highest probability (from `predict_proba`, and at least with at least probability
        `prob_threshold`) will be added to the list of labeled samples

    track_X : array-like, shape = [n_samples, n_features]
        Features matrix to use for tracking classifier performance over time. If left empty, performance
        will not be tracked

    track_y : array-like, shape = [n_samples]
        Labels of samples to use for tracking classifier performance over time. If left empty, performance
        will not be tracked

    track_callback : callable, (default=print)
        Callback function for when the prediction quality is tracked after each iteration
    """

    def __init__(self, base_classifiers, weights = None, max_iter = 100, max_pseudolabeled_samples_per_iter = 10, confidence_threshold = 0.7, bootstrapping = False, track_X = None, track_y = None, track_callback = print):
        self.base_classifiers = list(base_classifiers)
        self.weights = weights
        self.max_iter = max_iter
        self.max_pseudolabeled_samples_per_iter = max_pseudolabeled_samples_per_iter
        self.confidence_threshold = confidence_threshold
        self.bootstrapping = bootstrapping
        self.track_X = track_X
        self.track_y = track_y
        self.track_callback = track_callback

    def fit(self, X, y):
        """ All the input data is provided in matrix X (labeled and unlabeled) and corresponding
        label matrix y with a dedicated marker value for unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this
        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels

        Returns
        -------
        self : returns an instance of self.
        """

        # Check input
        X, y = check_X_y(X, y, accept_sparse = True, force_all_finite = False)
        check_classification_targets(y)

        # Store number of classes
        self.num_classes_ = len(np.unique(y[y != -1]))

        # Setup performance tracking
        self.performance_ = []

        # Split samples into labeled and unlabeled groups
        mask_labeled = (y != -1)
        mask_unlabeled = (y == -1)
        indices_labeled = np.where(mask_labeled)[0]
        indices_unlabeled = np.where(mask_unlabeled)[0]

        # Keep track of the labeled ans pseudo-labeled samples and the unlabeled samples that can be
        # pseudo-labeled for each classifier
        if self.bootstrapping:
            classifiers_indices_labeled = [resample(indices_labeled) for clf in self.base_classifiers]
        else:
            classifiers_indices_labeled = [indices_labeled for clf in self.base_classifiers]

        classifiers_indices_labeled_base = [x.copy() for x in classifiers_indices_labeled]
        classifiers_y = [y.copy() for clf in self.base_classifiers]
        classifiers_indices_tolabel = [indices_unlabeled for clf in self.base_classifiers]
        classifiers_unlabeled_tolabel = [list(range(len(indices_unlabeled))) for clf in self.base_classifiers]

        for i in range(self.max_iter):
            # Copy classifier instances from base classifier instances
            self.classifiers_ = [copy.deepcopy(base_classifier) for base_classifier in self.base_classifiers]

            # Train all classifiers
            i = 0
            for classifier in self.classifiers_:
                indices_use = classifiers_indices_labeled[i]
                X_use = X[indices_use, :]
                y_use = classifiers_y[i][indices_use]

                if len(y_use[y_use == -1]) > 0:
                    raise ValueError('There are unlabeled data points in the constructed training set. This is not a user error.')

                with warnings.catch_warnings():
                    warnings.showwarning = handle_warnings
                    classifier.fit(X_use, y_use)

                i = i + 1

            # Track classifier performance if it's enabled
            self._maybe_track_performance()

            # Make predictions for all unlabeled data for all classifiers
            classifiers_y_pred = [classifier.predict(X[mask_unlabeled, :]) for classifier in self.classifiers_]

            # Transform classifier predictions for unlabeled data into matrix
            classifiers_y_pred = np.array(classifiers_y_pred).T

            # Pseudo-labeling
            for clfi in range(len(self.classifiers_)):
                # Copy full predictions array
                Hi_y_pred = classifiers_y_pred.copy()

                # Remove current classifier's predictions
                Hi_y_pred = np.delete(Hi_y_pred, clfi, axis = 1)

                # Consider only examples that are still to be labeled for this classifier
                Hi_y_pred = Hi_y_pred[classifiers_unlabeled_tolabel[clfi], :]

                # Find the most-occurring prediction and the corresponding count for each sample
                values_counts  = [np.unique(row, return_counts = True) for row in Hi_y_pred]
                #[print(x) for x in values_counts]
                max_ind = [np.argmax(row[1]) for row in values_counts]
                max_values = [values_counts[i][0][max_ind[i]] for i in range(len(values_counts))]
                max_counts = [values_counts[i][1][max_ind[i]] for i in range(len(values_counts))]
                confidence = np.array(max_counts) / float(Hi_y_pred.shape[1])
                #print(len(confidence[confidence < 0.5]))
                # Todo: set confidence of samples that are not still to be labeled to -1

                # Pseudo-label the samples with the highest confidence
                max_pseudolabeled_samples_per_iter = min(self.max_pseudolabeled_samples_per_iter, len(confidence))

                use_pseudolabel_indices = np.argpartition(
                    confidence,
                    -max_pseudolabeled_samples_per_iter
                )[-max_pseudolabeled_samples_per_iter:]

                use_pseudolabel_indices = [i for i in use_pseudolabel_indices if confidence[i] >= self.confidence_threshold]

                use_pseudolabel_original_indices = indices_unlabeled[use_pseudolabel_indices]

                classifiers_indices_labeled[clfi] = list(classifiers_indices_labeled_base[clfi]) + list(use_pseudolabel_original_indices)
                classifiers_y[clfi][use_pseudolabel_original_indices] = [max_values[i] for i in use_pseudolabel_indices]

        return self

    def predict(self, X):
        """ Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """

        if self.weights is None:
            weights = [1.] * len(self.classifiers_)
        else:
            weights = self.weights


        classes = {}
        class_names = []

        def get_class_index(class_name):
            if class_name not in classes:
                classes[class_name] = len(classes)
                class_names.append(class_name)

            return classes[class_name]

        num_samples = X.shape[0]

        # Make predictions for all data for all classifiers
        predictions_matrix = np.zeros((X.shape[0], self.num_classes_))

        i = 0
        for classifier in self.classifiers_:
            # Make predictions for classifier
            y_pred = classifier.predict(X)

            # Get class indices (for use in numpy) corresponding to predictions
            y_pred_indices = [get_class_index(pred) for pred in classifier.predict(X)]

            # Update predictions matrix such that, in each row, the column corresponding to this
            # classifier's prediction is increased by the classifier's weight
            predictions_matrix[range(num_samples), y_pred_indices] = predictions_matrix[range(num_samples), y_pred_indices] + weights[i]

            i = i + 1

        # Convert predicted class indices to class names again
        pred = [class_names[pred_index] for pred_index in np.argmax(predictions_matrix, axis = 1)]

        return pred

    def get_tracked_performance(self):
        """ Retrieve the performance tracked during the training phase

        Returns
        ----------
        tracked_performance : list of float
            Performance tracked in training iterations
        """

        return self.performance_

    def _maybe_track_performance(self):
        """ If performance tracking is enabled, evaluate the algorithm on the tracking set and
        record the performance
        """

        if self.track_X is not None and self.track_y is not None:
            pred_y = self.predict(self.track_X)
            score = accuracy_score(pred_y, self.track_y)
            self.track_callback(str(1 - score))
            self.performance_.append(score)

    def get_tracked_performance(self):
        """ Retrieve the performance tracked during the training phase

        Returns
        ----------
        tracked_performance : list of float
            Performance tracked in training iterations
        """

        return self.performance_
