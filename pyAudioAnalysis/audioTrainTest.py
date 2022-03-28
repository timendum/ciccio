import pickle as cPickle

import numpy as np
import sklearn.decomposition
import sklearn.ensemble
import sklearn.metrics
import sklearn.svm
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from pyAudioAnalysis import MidTermFeatures as aF

shortTermWindow = 0.050
shortTermStep = 0.050


def classifier_wrapper(classifier, classifier_type, test_sample):
    """
    This function is used as a wrapper to pattern classification.
    ARGUMENTS:
        - classifier:        a classifier object of type sklearn.svm.SVC or
                             kNN (defined in this library) or sklearn.ensemble.
                             RandomForestClassifier or sklearn.ensemble.
                             GradientBoostingClassifier  or
                             sklearn.ensemble.ExtraTreesClassifier
        - classifier_type:   "svm" or "knn" or "randomforests" or
                             "gradientboosting" or "extratrees"
        - test_sample:        a feature vector (np array)
    RETURNS:
        - R:            class ID
        - P:            probability estimate
    """
    class_id = -1
    probability = -1
    if classifier_type == "svm":
        class_id = classifier.predict(test_sample.reshape(1, -1))[0]
        probability = classifier.predict_proba(test_sample.reshape(1, -1))[0]
    else:
        raise ValueError("Unsupported classifier_type: " + classifier_type)
    return class_id, probability


def train_svm(features, labels, c_param, kernel="linear"):
    """
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the sklearn functionality
              for SVM training
              See function trainSVM_feature() to use a wrapper on both the
              feature extraction and the SVM training
              (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a feature matrix [n_samples x numOfDimensions]
        - labels:           a label matrix: [n_samples x 1]
        - n_estimators:     number of trees in the forest
        - c_param:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value.
        For a different kernel, other types of parameters should be provided.
    """
    svm = sklearn.svm.SVC(C=c_param, kernel=kernel, probability=True, gamma="auto")
    svm.fit(features, labels)
    return svm


def extract_features_and_train(
    paths,
    mid_window,
    mid_step,
    short_window,
    short_step,
    classifier_type,
    model_name,
    train_percentage=0.90,
    dict_of_ids=None,
):
    """
    This function is used as a wrapper to segment-based audio feature extraction
    and classifier training.
    ARGUMENTS:
        paths:                      list of paths of directories. Each directory
                                    contains a signle audio class whose samples
                                    are stored in seperate WAV files.
        mid_window, mid_step:       mid-term window length and step
        short_window, short_step:   short-term window and step
        classifier_type:            "svm" or "knn" or "randomforest" or
                                    "gradientboosting" or "extratrees"
        model_name:                 name of the model to be saved
        dict_of_ids:                a dictionary which has as keys the full path of audio files and as values the respective group ids
    RETURNS:
        None. Resulting classifier along with the respective model
        parameters are saved on files.
    """

    # STEP A: Feature Extraction:
    features, class_names, file_names = aF.multiple_directory_feature_extraction(
        paths, mid_window, mid_step, short_window, short_step
    )
    file_names = [item for sublist in file_names for item in sublist]
    if dict_of_ids:
        list_of_ids = [dict_of_ids[file] for file in file_names]
    else:
        list_of_ids = None
    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return

    # n_feats = features[0].shape[1]
    # feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    for i, feat in enumerate(features):
        if len(feat) == 0:
            print("trainSVM_feature ERROR: " + paths[i] + " folder is empty or non-existing!")
            return

    # STEP B: classifier Evaluation and Parameter Selection:
    if classifier_type == "svm":
        classifier_par = np.array([0.001, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0])
    else:
        raise ValueError("Unsupported classifier_type: " + classifier_type)

    # get optimal classifier parameter:
    temp_features = []
    for feat in features:
        temp = []
        for i in range(feat.shape[0]):
            temp_fv = feat[i, :]
            if (not np.isnan(temp_fv).any()) and (not np.isinf(temp_fv).any()):
                temp.append(temp_fv.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        temp_features.append(np.array(temp))
    features = temp_features

    best_param = evaluate_classifier(
        features,
        class_names,
        classifier_type,
        classifier_par,
        1,
        list_of_ids,
        n_exp=-1,
        train_percentage=train_percentage,
    )

    print("Selected params: {0:.5f}".format(best_param))

    # STEP C: Train and Save the classifier to file
    # Get featues in the X, y format:
    features, labels = features_to_matrix(features)

    # Use mean/std standard feature scaling:
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    mean = scaler.mean_.tolist()
    std = scaler.scale_.tolist()

    # Then train the final classifier
    if classifier_type == "svm":
        classifier = train_svm(features, labels, best_param)
    else:
        raise ValueError("Unsupported classifier_type: " + classifier_type)

    # And save the model to a file, along with
    # - the scaling -mean/std- vectors)
    # - the feature extraction parameters

    if classifier_type == "svm":
        with open(model_name, "wb") as fid:
            cPickle.dump(classifier, fid)
        save_path = model_name + "_MEANS"
        save_parameters(
            save_path,
            mean,
            std,
            class_names,
            mid_window,
            mid_step,
            short_window,
            short_step,
        )
    else:
        raise ValueError("Unsupported classifier_type: " + classifier_type)


def save_parameters(path, *parameters) -> None:
    with open(path, "wb") as file_handle:
        for param in parameters:
            cPickle.dump(param, file_handle, protocol=cPickle.HIGHEST_PROTOCOL)


def load_model(model_name):
    """
    This function loads an SVM model either for classification or training.
    ARGMUMENTS:
        - SVMmodel_name:     the path of the model to be loaded
    """
    with open(model_name + "_MEANS", "rb") as fo:
        mean = cPickle.load(fo)
        std = cPickle.load(fo)
        classNames = cPickle.load(fo)
        mid_window = cPickle.load(fo)
        mid_step = cPickle.load(fo)
        short_window = cPickle.load(fo)
        short_step = cPickle.load(fo)

    mean = np.array(mean)
    std = np.array(std)

    with open(model_name, "rb") as fid:
        svm_model = cPickle.load(fid)

    return (
        svm_model,
        mean,
        std,
        classNames,
        mid_window,
        mid_step,
        short_window,
        short_step,
    )


def group_split(X, y, train_indeces, test_indeces, split_id):
    """
    This function splits the data in train and test set according to train/test indeces based on LeaveOneGroupOut
    ARGUMENTS:
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        train_indeces: The training set indices
        test_indeces: The testing set indices
        split_id: the split number
    RETURNS:
         List containing train-test split of inputs.

    """
    train_index = train_indeces[split_id]
    test_index = test_indeces[split_id]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def evaluate_classifier(
    features,
    class_names,
    classifier_name,
    params,
    parameter_mode,
    list_of_ids=None,
    n_exp=-1,
    train_percentage=0.90,
):
    """
    ARGUMENTS:
        features:     a list ([numOfClasses x 1]) whose elements containt
                      np matrices of features. Each matrix features[i] of
                      class i is [n_samples x numOfDimensions]
        class_names:    list of class names (strings)
        classifier_name: svm or knn or randomforest
        params:        list of classifier parameters (for parameter
                       tuning during cross-validation)
        parameter_mode:    0: choose parameters that lead to maximum overall
                             classification ACCURACY
                          1: choose parameters that lead to maximum overall
                          f1 MEASURE
        n_exp:        number of cross-validation experiments
                      (use -1 for auto calculation based on the num of samples)
        train_percentage: percentage of training (vs validation) data
                          default 0.90

    RETURNS:
         bestParam:    the value of the input parameter that optimizes the
         selected performance measure
    """

    # transcode list of feature matrices to X, y (sklearn)
    X, y = features_to_matrix(features)

    # features_norm = features;
    n_classes = len(features)
    ac_all = []
    f1_all = []
    f1_std_all = []
    pre_class_all = []
    rec_classes_all = []
    f1_classes_all = []
    cms_all = []

    # dynamically compute total number of samples:
    # (so that if number of samples is >10K only one train-val repetition
    # is performed)
    n_samples_total = X.shape[0]

    if n_exp == -1:
        n_exp = int(50000 / n_samples_total) + 1

    if list_of_ids:
        train_indeces, test_indeces = [], []
        gss = GroupShuffleSplit(n_splits=n_exp, train_size=0.8)
        for train_index, test_index in gss.split(X, y, list_of_ids):
            train_indeces.append(train_index)
            test_indeces.append(test_index)

    for C in params:
        # for each param value
        cm = np.zeros((n_classes, n_classes))
        f1_per_exp = []
        # y_pred_all = []
        # y_test_all = []
        for e in range(n_exp):
            y_pred = []
            # for each cross-validation iteration:
            print(
                "Param = {0:.5f} - classifier Evaluation "
                "Experiment {1:d} of {2:d}".format(C, e + 1, n_exp)
            )
            # split features:

            if list_of_ids:
                X_train, X_test, y_train, y_test = group_split(X, y, train_indeces, test_indeces, e)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1 - train_percentage
                )

            # mean/std scale the features:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)

            # train multi-class svms:
            if classifier_name == "svm":
                classifier = train_svm(X_train, y_train, C)
            else:
                raise ValueError("Unsupported classifier_type: " + classifier_name)

            # get predictions and compute current comfusion matrix
            cmt = np.zeros((n_classes, n_classes))
            X_test = scaler.transform(X_test)
            for i_test_sample in range(X_test.shape[0]):
                y_pred.append(
                    classifier_wrapper(classifier, classifier_name, X_test[i_test_sample, :])[0]
                )
            # current confusion matrices and F1:
            cmt = sklearn.metrics.confusion_matrix(y_test, y_pred)
            f1t = sklearn.metrics.f1_score(y_test, y_pred, average="macro")
            # aggregated predicted and ground truth labels
            # (used for the validation of final F1)
            # y_pred_all += y_pred
            # y_test_all += y_test.tolist()

            f1_per_exp.append(f1t)
            if cmt.size != cm.size:
                all_classes = set(y)
                split_classes = set(y_test.tolist() + y_pred)
                missing_classes = all_classes.difference(split_classes)
                missing_classes = list(missing_classes)
                missing_classes = [int(x) for x in missing_classes]
                for mm in missing_classes:
                    cmt = np.insert(cmt, mm, 0, axis=0)
                for mm in missing_classes:
                    cmt = np.insert(cmt, mm, 0, axis=1)
            cm = cm + cmt
        cm = cm + 0.0000000010

        rec = np.array([cm[ci, ci] / np.sum(cm[ci, :]) for ci in range(cm.shape[0])])
        pre = np.array([cm[ci, ci] / np.sum(cm[:, ci]) for ci in range(cm.shape[0])])

        pre_class_all.append(pre)
        rec_classes_all.append(rec)

        f1 = 2 * rec * pre / (rec + pre)

        # this is just for debugging (it should be equal to f1)
        # f1_b = sklearn.metrics.f1_score(y_test_all, y_pred_all, average="macro")
        # Note: np.mean(f1_per_exp) will not be exacty equal to the
        # overall f1 (i.e. f1 and f1_b because these are calculated on a
        # per-sample basis)
        f1_std = np.std(f1_per_exp)
        # print(np.mean(f1), f1_b, f1_std)

        f1_classes_all.append(f1)
        ac_all.append(np.sum(np.diagonal(cm)) / np.sum(cm))

        cms_all.append(cm)
        f1_all.append(np.mean(f1))
        f1_std_all.append(f1_std)

    print("\t\t", end="")
    for i, c in enumerate(class_names):
        if i == len(class_names) - 1:
            print("{0:s}\t\t".format(c), end="")
        else:
            print("{0:s}\t\t\t".format(c), end="")
    print("OVERALL")
    print("\tC", end="")
    for c in class_names:
        print("\tPRE\tREC\tf1", end="")
    print("\t{0:s}\t{1:s}".format("ACC", "f1"))
    best_ac_ind = np.argmax(ac_all)
    best_f1_ind = np.argmax(f1_all)
    for i in range(len(pre_class_all)):
        print("\t{0:.3f}".format(params[i]), end="")
        for c in range(len(pre_class_all[i])):
            print(
                "\t{0:.1f}\t{1:.1f}\t{2:.1f}".format(
                    100.0 * pre_class_all[i][c],
                    100.0 * rec_classes_all[i][c],
                    100.0 * f1_classes_all[i][c],
                ),
                end="",
            )
        print("\t{0:.1f}\t{1:.1f}".format(100.0 * ac_all[i], 100.0 * f1_all[i]), end="")
        if i == best_f1_ind:
            print("\t best f1", end="")
        if i == best_ac_ind:
            print("\t best Acc", end="")
        print("")

    if parameter_mode == 0:
        # keep parameters that maximize overall classification accuracy:
        print("Confusion Matrix:")
        print_confusion_matrix(cms_all[best_ac_ind], class_names)
        return params[best_ac_ind]
    elif parameter_mode == 1:
        # keep parameters that maximize overall f1 measure:
        print("Confusion Matrix:")
        print_confusion_matrix(cms_all[best_f1_ind], class_names)
        print(f"Best macro f1 {100 * f1_all[best_f1_ind]:.1f}")
        print(f"Best macro f1 std {100 * f1_std_all[best_f1_ind]:.1f}")
        return params[best_f1_ind]


def print_confusion_matrix(cm, class_names):
    """
    This function prints a confusion matrix for a particular classification task.
    ARGUMENTS:
        cm:            a 2-D np array of the confusion matrix
                       (cm[i,j] is the number of times a sample from class i
                       was classified in class j)
        class_names:    a list that contains the names of the classes
    """

    if cm.shape[0] != len(class_names):
        print("printConfusionMatrix: Wrong argument sizes\n")
        return

    for c in class_names:
        if len(c) > 4:
            c = c[0:3]
        print("\t{0:s}".format(c), end="")
    print("")

    for i, c in enumerate(class_names):
        if len(c) > 4:
            c = c[0:3]
        print("{0:s}".format(c), end="")
        for j in range(len(class_names)):
            print("\t{0:.2f}".format(100.0 * cm[i][j] / np.sum(cm)), end="")
        print("")


def features_to_matrix(features):
    """
    features_to_matrix(features)

    This function takes a list of feature matrices as argument and returns
    a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - feature_matrix:    a concatenated matrix of features
        - labels:            a vector of class indices
    """

    labels = np.array([])
    feature_matrix = np.array([])
    for i, f in enumerate(features):
        if i == 0:
            feature_matrix = f
            labels = i * np.ones((len(f), 1))
        else:
            feature_matrix = np.vstack((feature_matrix, f))
            labels = np.append(labels, i * np.ones((len(f), 1)))
    return feature_matrix, labels
