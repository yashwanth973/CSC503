#Import all neeeded resources
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#Determining data for cross-validation
O = np.array([f for f, m in training_data])
y = np.array([m for f, m in training_data])
classes = np.unique(y)
N_FOLDS = 5

# Define cross-validation function
def cross_validate(K, folds, report=False):
    model = KNeighborsClassifier(n_neighbors=K, metric="jaccard", algorithm="brute") #brute here tell the model to calculate the distant directly
    accuracies, precisions, recalls, f_measures = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        model.fit(O[train_idx], y[train_idx]) # train the model based on the unfolded set
        y_pred = model.predict(O[test_idx]) # test the model based on the folded set
        y_test = y[test_idx] # retrieve real result

        fold_p, fold_r, fold_f = [], [], []
        for c in classes:
          tp = np.sum((y_test == c) & (y_pred == c))
          fp = np.sum((y_test != c) & (y_pred == c))
          fn = np.sum((y_test == c) & (y_pred != c))

          precision = tp / (tp + fp) if tp + fp else 0.0
          recall = tp / (tp + fn) if tp + fn else 0.0
          f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0

          fold_p.append(precision)
          fold_r.append(recall)
          fold_f.append(f1)
        
        precisions.append(np.mean(fold_p))
        recalls.append(np.mean(fold_r))
        f_measures.append(np.mean(fold_f))
        accuracies.append(np.mean(y_pred == y_test))
    return accuracies, precisions, recalls, f_measures
