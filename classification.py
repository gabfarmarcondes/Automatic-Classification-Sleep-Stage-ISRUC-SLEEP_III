"""
Classification for ISRUC Sleep Staging

Loads features (isruc_features.npz), runs 5 classifiers with subject-wise
10-fold cross-validation (GroupKFold), and reports accuracy, macro-F1 and
confusion matrices.

Why GroupKFold: with 10 subjects, GroupKFold(10) puts one whole subject per
fold. No epoch from a test subject ever appears in training -> no leakage.
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


# Load
data = np.load("isruc_features.npz", allow_pickle=True)
X, y, groups = data["features"], data["y"], data["groups"]
feature_names = data["feature_names"]

print(f"Features: {X.shape}, Labels: {y.shape}, Subjects: {len(np.unique(groups))}")


# Classifiers (the 5 the assignment lists)
def make_classifiers():
    """Fresh classifier instances (rebuilt each fold to avoid state leakage)."""
    return {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32),
                             max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


# Cross-Validation (subject-wise, 10 folds)
gkf = GroupKFold(n_splits=10)

# Store out-of-fold predictions per classifier (for global metrics + confusion)
results = {name: {"y_true": [], "y_pred": []} for name in make_classifiers()}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # IMPORTANT: fit the scaler ONLY on training data, then apply to test.
    # Fitting on the whole set would leak test info into training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    test_subject = np.unique(groups[test_idx])
    print(f"Fold {fold:2d} | test subject(s): {test_subject} | "
          f"train={len(train_idx)}, test={len(test_idx)}")

    for name, clf in make_classifiers().items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name]["y_true"].extend(y_test)
        results[name]["y_pred"].extend(preds)


# Report
print("\n" + "=" * 55)
print(f"{'Classifier':<16}{'Accuracy':>12}{'Macro-F1':>12}")
print("=" * 55)

for name in results:
    yt = np.array(results[name]["y_true"])
    yp = np.array(results[name]["y_pred"])
    acc = accuracy_score(yt, yp) * 100
    f1 = f1_score(yt, yp, average="macro") * 100
    print(f"{name:<16}{acc:>11.1f}%{f1:>11.1f}%")

print("=" * 55)

# Confusion matrix for each classifier (percent per true class)
for name in results:
    yt = np.array(results[name]["y_true"])
    yp = np.array(results[name]["y_pred"])
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3, 4])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100  # row-normalized %

    print(f"\nConfusion Matrix — {name} (% per true row)")
    print("        " + "".join(f"{s:>7}" for s in STAGE_NAMES))
    for i, row in enumerate(cm_pct):
        print(f"{STAGE_NAMES[i]:>6}  " + "".join(f"{v:>6.1f}%" for v in row))