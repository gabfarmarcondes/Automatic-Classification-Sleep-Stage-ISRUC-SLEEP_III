"""
Static Classifier Combination for ISRUC Sleep Staging

Builds an ensemble of 20+ classifiers by combining:
    5 feature subsets  x  4 base classifiers  = 20 members
Each member is (feature subset + classifier). Their predictions are
combined by MAJORITY VOTING. We then check if the ensemble beats the
best single classifier (SVM, 74.5% from the previous step).

Still uses subject-wise 10-fold (GroupKFold) — same honest protocol.
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# Load
data = np.load("isruc_features.npz", allow_pickle=True)
X, y, groups = data["features"], data["y"], data["groups"]
feature_names = list(data["feature_names"])
print(f"Features: {X.shape}, Subjects: {len(np.unique(groups))}")
print(f"Available features: {feature_names}\n")


# 5 FEATURE SUBSETS (by index into the 15 features)
# Indices:
# 0 mean, 1 std, 2 var, 3 skew, 4 kurtosis,
# 5 hjorth_activity, 6 hjorth_mobility, 7 hjorth_complexity,
# 8 zero_crossing_rate,
# 9 delta, 10 theta, 11 alpha, 12 sigma, 13 beta, 14 sef95
FEATURE_SUBSETS = {
    "all":        list(range(15)),                 # everything
    "freq_only":  [9, 10, 11, 12, 13, 14],         # only frequency bands + SEF95
    "time_only":  [0, 1, 2, 3, 4, 5, 6, 7, 8],     # only time-domain
    "hjorth+freq":[5, 6, 7, 9, 10, 11, 12, 13],    # Hjorth + bands
    "compact":    [9, 10, 11, 12, 13, 5, 1],       # a small discriminative set
}


# 4 Base Classifiers (fresh instances each call)

def make_base_classifiers():
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "tree": DecisionTreeClassifier(random_state=42),
        "svm": SVC(kernel="rbf", C=1.0, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    }


# Total members = 5 subsets x 4 classifiers = 20
N_MEMBERS = len(FEATURE_SUBSETS) * len(make_base_classifiers())
print(f"Ensemble size: {N_MEMBERS} members "
      f"({len(FEATURE_SUBSETS)} feature subsets x "
      f"{len(make_base_classifiers())} classifiers)\n")


# Cross-Validation with majority voting
gkf = GroupKFold(n_splits=10)
ens_true, ens_pred = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
    y_train, y_test = y[train_idx], y[test_idx]

    # Collect each member's prediction on the test set
    # member_preds: shape (n_members, n_test_samples)
    member_preds = []

    for subset_name, cols in FEATURE_SUBSETS.items():
        # Slice the chosen feature columns
        X_train = X[train_idx][:, cols]
        X_test = X[test_idx][:, cols]

        # Scale (fit on train only — no leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for clf_name, clf in make_base_classifiers().items():
            clf.fit(X_train, y_train)
            member_preds.append(clf.predict(X_test))

    member_preds = np.array(member_preds)  # (n_members, n_test)

    # MAJORITY VOTE per test sample (column)
    voted = []
    for col in range(member_preds.shape[1]):
        votes = member_preds[:, col]
        winner = Counter(votes).most_common(1)[0][0]  # most frequent label
        voted.append(winner)

    ens_true.extend(y_test)
    ens_pred.extend(voted)
    print(f"Fold {fold:2d} done (test subject {np.unique(groups[test_idx])})")


# Report
ens_true = np.array(ens_true)
ens_pred = np.array(ens_pred)

acc = accuracy_score(ens_true, ens_pred) * 100
f1 = f1_score(ens_true, ens_pred, average="macro") * 100

print("\n" + "=" * 50)
print("ENSEMBLE (majority voting, 20 members)")
print("=" * 50)
print(f"Accuracy: {acc:.1f}%")
print(f"Macro-F1: {f1:.1f}%")
print(f"\nCompare to best single (SVM): 74.5% acc / 71.8% F1")

cm = confusion_matrix(ens_true, ens_pred, labels=[0, 1, 2, 3, 4])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
print("\nConfusion Matrix — Ensemble (% per true row)")
print("        " + "".join(f"{s:>7}" for s in STAGE_NAMES))
for i, row in enumerate(cm_pct):
    print(f"{STAGE_NAMES[i]:>6}  " + "".join(f"{v:>6.1f}%" for v in row))