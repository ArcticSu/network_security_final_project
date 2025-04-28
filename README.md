# Network Security Final Project: Website Fingerprinting using Machine Learning

This project explores website fingerprinting attacks on Tor traffic using various machine learning models. The input data consists of packet-level features observed at the entry guard node, and the task is to identify which website was visited.

## Task Overview

- **Input (X)**: Sequences of packets represented by tuples `(direction, deltaT, size)`
  - `direction`: 1 for outgoing (user → guard), -1 for incoming (guard → user)
  - `deltaT`: timestamp difference (relative to first packet)
  - `size`: packet size
- **Output (Y)**: Visited website (`amazon.com`, `reddit.com`, etc.)

## Project Structure

| File / Folder        | Description |
|----------------------|-------------|
| `Filtered-Data/`     | Original `.pcapng` traffic data collected from Tor browser |
| `processed_file/`    | Preprocessed CSV files containing parsed packet sequences per visit |
| `dataloader.py`      | Dataset loading utilities with sliding window support |
| `extract_info.py`    | Parses `.pcapng` files to extract `(deltaT, direction, size)` and saves as CSV |
| `read_file.py`       | Preview tool for inspecting `.pcapng` packet summaries |
| `model_lstm.py`      | LSTM-based sequence classifier for website prediction |
| `model_gru.py`       | GRU-based variant (simplified recurrent architecture) |
| `model_svm.py`       | Flattened statistical feature classifier using Support Vector Machine |
| `model_trans.py`     | **Transformer-based classifier (best performance)** for sequence modeling |
