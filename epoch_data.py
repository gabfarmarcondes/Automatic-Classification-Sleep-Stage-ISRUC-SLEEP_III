"""
Epochin Pipeline

Reads all 10 subjects, extracts the C3-A2 EEG channel, segments the signal into 30 second epochs,
pairs each epoch with its sleep-stage label, and saves everything into a single .npz file for fast reuse

Output: isruc_data.npz containing
    X -> signal matrix, shape (n_epochs_total, samples_per_epoch)
    Y -> label vector, shape (n_epochs_total)
    groups -> subject id per epoch, shape (n_epochs_total)
"""

import os
import shutil
import tempfile
import numpy as np
import mne

# Configuration
DATA_ROOT = "./ISRUC-Sleep-III" # folder containing subfolders with the subjects
SUBJECTS= list(range(1,11)) # subjects 1 to 10
EEG_CHANNEL = "C3-A2" # standard single channel for sleep staging
EPOCH_SEC = 30 # epoch length in secods (AASM standard)
SPECIALIST = 1 # which expert's annotation to use (1 or 2)

# Label remap: ISRUC uses {0,1,2,3,5}. We compress to contiguos {0,1,2,3,4}
# 0 = wake, 1 = N1, 2 = N2, 3 = N3, 5 = REM -> REM becomes 4
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}

OUTPUT_FILE = "isruc_data.npz"

# Helper: read a .rec file (which is EDF under the hood)
def read_rec_as_edf(rec_path):
    # MNE refuses the .rec extension, so we copy to a temporary .edf first
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp_edf = tmp.name
    shutil.copy(rec_path, tmp_edf)
    raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose=False)
    os.remove(tmp_edf)
    return raw

# Helper: process a single subject
def process_subject(subject_id):
    # Return (X_subject, y_subject) for one subject, or (None, None) on mismatch:
    folder = os.path.join(DATA_ROOT, str(subject_id))
    rec_path = os.path.join(folder, f"{subject_id}.rec")
    txt_path = os.path.join(folder, f"{subject_id}_{SPECIALIST}.txt")

    # 1 - read the signal and pick the EEG channel
    raw = read_rec_as_edf(rec_path)
    sfreq = int(raw.info["sfreq"]) # 200Hz for ISRUC
    samples_per_epoch = sfreq * EPOCH_SEC # 200 * 30 = 6000 samples

    # pick_channels keeps only the channel we want and get_data returns shape (1, n_samples)
    raw.pick_channels([EEG_CHANNEL])
    signal = raw.get_data()[0] # 1D array of thw whole night

    # 2 - read the stage labels
    with open(txt_path, "r") as fh:
        raw_labels = [int(line.strip()) for line in fh if line.strip() != ""]
    
    # 3 - segment the signal into 30s epochs
    n_epochs_signal = len(signal) // samples_per_epoch # whole epochs only
    n_epochs = min(n_epochs_signal, len(raw_labels)) # align to the shorter one

    # trim both to the same length (drop any leftlover partial epoch / extra label)
    signal = signal [: n_epochs * samples_per_epoch]
    labels = raw_labels[: n_epochs]

    # reshape the flat signal into a matrix: one row per epoch
    X_subject = signal.reshape(n_epochs, samples_per_epoch)

    # 4 - remap labels {0,1,2,3,5} -> {0,1,2,3,4}
    y_subject = np.array([LABEL_MAP[lab] for lab in labels])

    return X_subject, y_subject

# Main: Loop over all subjects and stack the results
def main():
    X_all, y_all, groups_all = [], [], []

    for sid in SUBJECTS:
        print(f"Processing Subject {sid}", end=" ")
        X_s, y_s = process_subject(sid)
        print(f"{X_s.shape[0]} epochs")

        X_all.append(X_s)
        y_all.append(y_s)
        groups_all.append(np.full(X_s.shape[0], sid)) # record which subject each epoch came from (needed for subject-wise CV)

    # concatenate every subject into one big array
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    groups = np.concatenate(groups_all, axis=0)

    print("\n=== Summary ===")
    print(f"X shape: {X.shape} (epochs x samples)")
    print(f"y shape: {y.shape}")
    print(f"groups shape: {groups.shape}")

    # class distribution - expect N2 to dominante, N1 to be rare
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    print("\nClass Distribution")
    for cls in range(5):
        count = np.sum(y == cls)
        pct = 100 * count / len(y)
        print(f"  {stage_names[cls]:5s} (label {cls}): {count:5d} epochs ({pct:.1f}%)")
    
    # save everything for instante reuse later
    np.savez_compressed(OUTPUT_FILE, X=X, y=y, groups=groups)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()