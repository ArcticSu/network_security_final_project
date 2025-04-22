import requests
import argparse
from scapy.all import rdpcap, wrpcap, IP  # pip install scapy
import os

def is_guard_node(ip_address):
    url = f"https://onionoo.torproject.org/details?search={ip_address}"
    response = requests.get(url)
    data = response.json()

    if not data.get('relays'):
        return False

    for relay in data['relays']:
        if 'Guard' in relay.get('flags', []):
            return True
    return False


def read_pcap(pcap_file):

    packets = rdpcap(pcap_file)

    unique_ips = set()
    for pkt in packets:
        if IP in pkt:
            unique_ips.add(pkt[IP].src)
            unique_ips.add(pkt[IP].dst)

    print(f"Found {len(unique_ips)} unique IPs.")

    tor_guard_ips = set()
    print("Checking Tor guard nodes...")

    for ip in unique_ips:
        if is_guard_node(ip):
            tor_guard_ips.add(ip)

    print(f"Found {len(tor_guard_ips)} Tor guard nodes!")


    print("Filtering packets...")

    filtered_packets = []
    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            if src_ip in tor_guard_ips or dst_ip in tor_guard_ips:
                filtered_packets.append(pkt)

    base_dir = os.path.dirname(pcap_file)
    base_name = os.path.basename(pcap_file).replace(".pcapng", "")
    output_pcap = os.path.join(base_dir, f"{base_name}.pcapng")

    wrpcap(output_pcap, filtered_packets)
    print(f"Done! Filtered pcapng saved as {output_pcap}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_folder", help="Path to pcap folder")
    
    args = parser.parse_args()

    pcap_files = [os.path.join(args.pcap_folder, f) for f in os.listdir(args.pcap_folder) if f.endswith(".pcapng")]

    print(f"Found {len(pcap_files)} pcapng files in folder!")
    for pcap_file in pcap_files:
        print(f"Reading {pcap_file}...")
        read_pcap(pcap_file)


if __name__ == "__main__":
    main()
