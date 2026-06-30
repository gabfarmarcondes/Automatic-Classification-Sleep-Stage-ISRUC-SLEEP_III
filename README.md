# Sleep Stage Classification — ISRUC-Sleep-III

Machine learning pipeline for automatic sleep stage classification from single-channel EEG signals (C3-A2), using the public ISRUC-Sleep-III dataset with 10 subjects.

Classified stages follow the AASM standard: **Wake**, **N1**, **N2**, **N3**, and **REM**.

---

## Project Structure

```
.
├── epoch_data.py        # Step 1: segment the raw signal into epochs
├── extract_features.py  # Step 2: extract features per epoch
├── classification.py    # Step 3: evaluate 5 individual classifiers
├── combination.py       # Step 4: ensemble by majority voting
└── ISRUC-Sleep-III/     # Raw dataset (not versioned)
```

---

## File Descriptions

### `epoch_data.py`
Reads the `.rec` files (EDF format) and annotation `.txt` files for each of the 10 subjects. Extracts the C3-A2 EEG channel, segments the signal into 30-second epochs (AASM standard), and remaps labels from `{0, 1, 2, 3, 5}` to `{0, 1, 2, 3, 4}` (REM: 5 → 4). Saves all concatenated data to `isruc_data.npz`.

**Output:** `isruc_data.npz` — arrays `X` (epochs × samples), `y` (labels), and `groups` (subject id per epoch).

---

### `extract_features.py`
Loads `isruc_data.npz` and transforms each raw epoch (6000 samples at 200 Hz) into a **15-feature** vector:

| Domain | Features |
|---|---|
| Time | mean, std, variance, skewness, kurtosis, Hjorth (activity, mobility, complexity), zero-crossing rate |
| Frequency | relative band power for delta, theta, alpha, sigma, beta; SEF95 |

**Output:** `isruc_features.npz` — arrays `features` (epochs × 15), `y`, `groups`, and `feature_names`.

---

### `classification.py`
Loads `isruc_features.npz` and evaluates **5 classifiers** with subject-wise cross-validation (`GroupKFold`, 10 folds — one subject per fold). The scaler is fit only on training data within each fold to prevent data leakage.

Classifiers: **k-NN**, **Decision Tree**, **SVM (RBF)**, **MLP**, and **Random Forest**.

Reports accuracy, macro-F1, and row-normalized confusion matrices.

---

### `combination.py`
Builds an ensemble of **20 members** (5 feature subsets × 4 base classifiers) and combines their predictions by **majority voting**. Uses the same 10-fold `GroupKFold` protocol.

Feature subsets: `all`, `freq_only`, `time_only`, `hjorth+freq`, and `compact`.  
Base classifiers: k-NN, Decision Tree, SVM, and Random Forest.

The result is compared against the best individual classifier from the previous step.

---

## Requirements

```bash
pip install numpy scipy scikit-learn mne
```

---

## Dataset

Download **ISRUC-Sleep-III** and extract it to the project root, keeping the structure below:

```
ISRUC-Sleep-III/
├── 1/
│   ├── 1.rec
│   └── 1_1.txt
├── 2/
│   ├── 2.rec
│   └── 2_1.txt
...
└── 10/
    ├── 10.rec
    └── 10_1.txt
```

---

## Execution Order

Run the scripts **in the order below**. Each step generates the file consumed by the next.

```bash
# Step 1 — segment the raw signal into 30s epochs
python epoch_data.py
# Generates: isruc_data.npz

# Step 2 — extract features from each epoch
python extract_features.py
# Generates: isruc_features.npz

# Step 3 — evaluate 5 classifiers individually
python classification.py

# Step 4 — evaluate the majority-voting ensemble (20 members)
python combination.py
```

---

## Results

### Accuracy and Macro-F1 — Individual Classifiers

Evaluated with subject-wise cross-validation (GroupKFold, 10 folds).

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| SVM (RBF) | 74.5%| 71.8%|
| Random Forest | 72.9%| 70.%|
| MLP | 71.7%| 70.2%|
| k-NN | 71.7%| 70.2%|
| Decision Tree | 62.3%| 59.8%|

---

### Confusion Matrix — SVM

Values as percentage per row (row = true class).

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **Wake** | **86.2%**| 8.5%| 3.0%| 0.9%| 1.3%|
| **N1** | 14.8%| **44.0%**| 22.9%| 0.2%| 18.0%|
| **N2** | 2.4%| 5.2%| **78.4%**| 8.8%| 5.2%|
| **N3** | 0.7%| 0.0%| 18.1%| **80.2%**| 1.0%|
| **REM** | 2.5%| 16.6%| 10.4%| 0.8%| **69.8%**|

---

### SVM vs. Ensemble (Majority Voting)

The ensemble combines 20 members (5 feature subsets × 4 base classifiers).

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **Wake** | **86.5%**| 7.8%| 3.4%| 0.9%| 1.4%|
| **N1** | 16.5%| **42.9%**| 22.8%| 0.6%| 17.2%|
| **N2** | 2.9%| 5.2%| **73.4%**| 12.8%| 5.6%|
| **N3** | 0.6%| 0.0%| 21.3%| **77.3%**| 0.7%|
| **REM** | 2.7%| 21.2%| 14.3%| 1.5%| **60.2%**|

---

## Discussion and Analysis

### Strengths
The Wake, N2 and N3 stages were classifieds with a high success rate for all models. This result is consistent with the sleep physiology and with the features extracted: the N3 is dominated by large amplitude delta waves; N2 presents sleep spindles in the sigma band, a specific spectral marker; and wakefulness has rapid activity (beta) and greater signatures that the features can represent well.

Among the classifiers, the SVM with RBF kernel had the best performance, which is consistent with its effectiveness in moderate-dimensional problems and with well-designed features.

### Weaknesses
Stage N1 was the most difficult to classify, with a sucess rate of around 44%. This has a physiological explanation: N1 is a transition stage between wakefulness and sleep, short and with mixed-features EEG, which shares properties with both Wake as with N2 and REM. As it does not have unique and stable spectral marker, it is confused with neighboring stages. A well-known problem in the literature, which it even affects agreement between human experts.

The observed confusions are physiologically coherent: N1 is confused with Wake and REM (all with low amplitude and mixed frequency EEG), and N3 is confused with N2 (neighboring stages in the sleep depth continuum, both with slow activity). The imbalance of classes, with N2 dominant and N1/REM in the minority, also contributes to the difficulty in rare classes.