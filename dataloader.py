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

def load_dataset_homepage(csv_dir, window_size=300, stride=100, verbose=True):
    X, Y = [], []
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        if "homepage" not in fname.lower():
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


def load_dataset_subpage_by_site(csv_dir, target_site="amazon", window_size=300, stride=100, verbose=True):
    """
    Load only files belonging to a specific target website (e.g., amazon),
    and classify among its subpages (e.g., homepage, deal, pharm).
    """
    X, Y = [], []
    sub_labels = []

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue

        fname_lower = fname.lower()

        if target_site.lower() not in fname_lower:
            continue

        if "homepage" in fname_lower:
            sub_label = "homepage"
        elif "deal" in fname_lower:
            sub_label = "deal"
        elif "pharm" in fname_lower:
            sub_label = "pharm"
        elif "brooke" in fname_lower:
            sub_label = "brooke"
        elif "sun" in fname_lower:
            sub_label = "sun"
        elif "mild" in fname_lower:
            sub_label = "mild"
        elif "dogs" in fname_lower:
            sub_label = "dogs"
        elif "today" in fname_lower:
            sub_label = "today"
        elif "monthly" in fname_lower:
            sub_label = "monthly"
        elif "uva" in fname_lower:
            sub_label = "uva"
        elif "tj" in fname_lower:
            sub_label = "tj"
        else:
            if verbose:
                print(f"Skipping file (no subpage label found): {fname}")
            continue

        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path)
            df = df[df["direction"].isin([-1, 1])]
            sequence = list(zip(df["direction"], df["delta_time"], df["length"]))

            chunks = split_sequence(sequence, window_size=window_size, stride=stride)
            X.extend(chunks)
            Y.extend([sub_label] * len(chunks))
        except Exception as e:
            if verbose:
                print(f"[ERROR] Reading {fname}: {e}")
            continue

    X = np.array(X, dtype=np.float32)
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    if verbose:
        print(f"Loaded {len(X)} samples for {target_site}.")
        print(f"Subpage labels detected: {list(label_encoder.classes_)}")

    return X, Y_encoded, label_encoder


def load_dataset_sitepagelevel(csv_dir, window_size=300, stride=100, verbose=True):
    """
    Load all pages (homepage + subpages) across multiple websites,
    classify based on fixed site-page keywords in filename.
    Ignore trailing _b, _d, _m, _1, _2, etc.
    """
    X, Y = [], []

    target_labels = [
        "amazon_homepage", "amazon_deal", "amazon_pharm",
        "youtube_homepage", "youtube_sun", "youtube_brooke",
        "reddit_homepage", "reddit_mild", "reddit_dogs",
        "weather_homepage", "weather_today", "weather_monthly",
        "wikipedia_homepage", "wikipedia_uva", "wikipedia_tj"
    ]

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue

        fname_lower = fname.lower()

        matched_label = None
        for label in target_labels:
            if label in fname_lower:
                matched_label = label
                break 

        if matched_label is None:
            if verbose:
                print(f"Skipping unmatched file: {fname}")
            continue

        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path)
            df = df[df["direction"].isin([-1, 1])]
            sequence = list(zip(df["direction"], df["delta_time"], df["length"]))

            chunks = split_sequence(sequence, window_size=window_size, stride=stride)
            X.extend(chunks)
            Y.extend([matched_label] * len(chunks))
        except Exception as e:
            if verbose:
                print(f"[ERROR] Reading {fname}: {e}")
            continue

    X = np.array(X, dtype=np.float32)
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    if verbose:
        print(f"Loaded {len(X)} samples across 15 site-page classes.")
        print(f"Detected labels: {list(label_encoder.classes_)}")

    return X, Y_encoded, label_encoder
