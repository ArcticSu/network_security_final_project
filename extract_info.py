import os
import pyshark
import requests
import pandas as pd

def is_guard_node(ip_address):
    url = f"https://onionoo.torproject.org/details?search={ip_address}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if not data.get('relays'):
            return False
        for relay in data['relays']:
            if 'Guard' in relay.get('flags', []):
                return True
        return False
    except:
        return False

def process_pcapng(file_path, output_csv_path):
    cap = pyshark.FileCapture(file_path)
    results = []
    baseline_time = None
    guard_cache = {}

    for i, packet in enumerate(cap):
        try:
            pkt_time = float(packet.sniff_timestamp)
            if baseline_time is None:
                baseline_time = pkt_time
            delta_t = pkt_time - baseline_time

            if not hasattr(packet, 'ip'):
                continue

            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            length = int(packet.length)

            if src_ip in guard_cache:
                src_is_guard = guard_cache[src_ip]
            else:
                src_is_guard = is_guard_node(src_ip)
                guard_cache[src_ip] = src_is_guard

            if dst_ip in guard_cache:
                dst_is_guard = guard_cache[dst_ip]
            else:
                dst_is_guard = is_guard_node(dst_ip)
                guard_cache[dst_ip] = dst_is_guard

            if src_is_guard and not dst_is_guard:
                direction = -1
            elif dst_is_guard and not src_is_guard:
                direction = 1
            else:
                direction = 0

            results.append([delta_t, direction, length])

        except Exception as e:
            print(f"[WARN] Skipping packet {i} in {file_path}: {e}")
            continue


    cap.close()

    df = pd.DataFrame(results, columns=['delta_time', 'direction', 'length'])
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} rows to {output_csv_path}")

def process_all_files_in_folder(folder_path):
    parent_dir = os.path.dirname(folder_path)
    output_dir = os.path.join(parent_dir, "processed_file")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pcapng"):
            file_path = os.path.join(folder_path, filename)
            output_csv_path = os.path.join(output_dir, filename.replace('.pcapng', '.csv'))
            print(f"Processing {filename}...")
            process_pcapng(file_path, output_csv_path)

if __name__ == "__main__":
    folder = r"D:\UVA\CS6501 Network Security and Privacy\final_project\Filtered-Data"
    process_all_files_in_folder(folder)
