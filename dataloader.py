import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_KEYWORDS = {
    "amazon": "amazon.com",
    "youtube": "youtube.com",
    "reddit": "reddit.com",
    "weather": "weather.com",
    "wiki": "wikipedia.org",
    "wikipedia": "wikipedia.org",
}

def infer_label_from_filename(fname):
    fname = fname.lower()
    for keyword, label in LABEL_KEYWORDS.items():
        if keyword in fname:
            return label
    return None

def split_sequence(seq, window_size=300, stride=100):
    chunks = []
    for i in range(0, len(seq), stride):
        chunk = seq[i:i + window_size]
        if len(chunk) < window_size:
            # padding with (0, 0.0, 0.0)
            chunk += [(0, 0.0, 0.0)] * (window_size - len(chunk))
        chunks.append(chunk)
        if i + window_size >= len(seq):
            break
    return chunks

def load_dataset(csv_dir, window_size=300, stride=100, verbose=True):
    X, Y = [], []
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        label = infer_label_from_filename(fname)
        if label is None:
            if verbose:
                print(f"Skipping file (no label match): {fname}")
            continue

        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path)
            df = df[df["direction"].isin([-1, 1])]
            sequence = list(zip(df["direction"], df["delta_time"], df["length"]))

            chunks = split_sequence(sequence, window_size=window_size, stride=stride)
            X.extend(chunks)
            Y.extend([label] * len(chunks))
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed on {fname}: {e}")
            continue

    X = np.array(X, dtype=np.float32)  
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    if verbose:
        print(f"Loaded {len(X)} samples from `{csv_dir}`.")
        print(f"Detected labels: {list(label_encoder.classes_)}")

    return X, Y_encoded, label_encoder
