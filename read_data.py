# Purpose of this code: open a subject, understand the structure of the signal and the anotations.

import os
import numpy as np

DATA_DIRECTORY = "./ISRUC-Sleep-III/1" # change the subject's number here
SUBJECT_ID = 1

print("ISRUC SUBJECT ", SUBJECT_ID)

# List the files in the folder
print("\n[1] Files in the Folder")
for f in sorted(os.listdir(DATA_DIRECTORY)):
    PATH = os.path.join(DATA_DIRECTORY, f)
    SIZE = os.path.getsize(PATH) / (1024 * 1024)
    print(f"    {f:20s}     ({SIZE:.1f}) MB")

print("*"*60)

# Transforming .rec to .edf
import shutil
import tempfile
import mne

REC_PATH = os.path.join(DATA_DIRECTORY, f"{SUBJECT_ID}.rec")

with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
    tmp_edf = tmp.name
shutil.copy(REC_PATH, tmp_edf)

raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose=False)
os.remove(tmp_edf)

# Open the Signal
print(f"Sampling Rate: {raw.info['sfreq']} Hz")
print(f"Total Duration: {raw.n_times / raw.info['sfreq'] / 3600 :.2f} hours")
print(f"Number of Channels: {len(raw.ch_names)}")
print("Avaliable Channels: ")
for ch in raw.ch_names:
    print(f"    - {ch}")

print("*"*60)

# Read the Notes from the Specialist Number 1
print("Reading the Notes from the Specialist Number 1")
txt_path = os.path.join(DATA_DIRECTORY, f"{SUBJECT_ID}_1.txt")

with open(txt_path, "r") as fh:
    lines = [l.strip() for l in fh if l.strip() != ""]

print(f"Number of lines (annoted epochs): {len(lines)}")
print(f"First 10 lines: {lines[:10]}")
print(f"Unique Values Founded: {sorted(set(lines))}")

# in ISRUC, the typical encoding is: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = N5
# It does not have 4 because AASM merges S3+S4 in N3; the 5 = REM is inheritance of numeration
print("\nExpected Interpretation (AASM in ISRUC)")
print("0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = REM")

print("*"*60)

# Read the Notes from the Specialist Number 2
print("Reading the Notes from the Specialist Number 2")
txt_path = os.path.join(DATA_DIRECTORY, f"{SUBJECT_ID}_2.txt")

with open(txt_path, "r") as fh:
    lines = [l.strip() for l in fh if l.strip() != ""]

print(f"Number of lines (annoted epochs): {len(lines)}")
print(f"First 10 lines: {lines[:10]}")
print(f"Unique Values Founded: {sorted(set(lines))}")

# in ISRUC, the typical encoding is: 0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = N5
# It does not have 4 because AASM merges S3+S4 in N3; the 5 = REM is inheritance of numeration
print("\nExpected Interpretation (AASM in ISRUC)")
print("0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = REM")

print("*"*60)

# Sanity Check: number of epochs x signal duration
seg_duration = raw.n_times / raw.info["sfreq"]
expected_epochs = seg_duration / 30
print(f" Signal Duration: {seg_duration:.0f}s -> ~{expected_epochs:.0f} 30s epochs")
print(f"Noted Epochs: {len(lines)}")

if abs(expected_epochs - len(lines)) > 5:
    print("[Warning] Discrepancy between the signal epochs and the notes")
    print("Could have extra epochs in the beginnning/end to cut it of")
else:
    print("[Ok] Number of compatilble epochs")
