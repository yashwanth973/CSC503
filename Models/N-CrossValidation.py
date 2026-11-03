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

#Getting the best K 
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2026)
folds = list(skf.split(O, y))     # same splits reused for every k

candidate_ks = [1, 3, 5, 7, 11, 15, 21, 31, 41, 51, 100] # testing among these K values 
results = {}

for K in candidate_ks:
    acc, prec, rec, f = cross_validate(K, folds)
    results[K] = {"acc": np.mean(acc), 
                  "std": np.std(acc),
                  "f": np.mean(f),
                  "prec": np.mean(prec),
                  "rec": np.mean(rec),
                  "f": np.mean(f)}

print(f"{'k':>4} {'Acc':>7} {'Std':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 44)
for K, s in results.items():
    print(f"{K:>4} {s['acc']:>7.3f} {s['std']:>7.3f} "
          f"{s['prec']:>7.3f} {s['rec']:>7.3f} {s['f']:>7.3f}")

best_k = max(results, key=lambda k: results[k]["f"])   # or "acc"
print("Best k:", best_k)

#show the actual output with respect to that best K value
cross_validate(best_k, folds,report = True);]
