# import pyshark

# file_path = r'D:\UVA\CS6501 Network Security and Privacy\final_project\Filtered-Data\amazon_deal_b.pcapng'

# cap = pyshark.FileCapture(file_path)

# for i, packet in enumerate(cap):
#     print(f"\n======= Packet {i+1} =======")
#     print(f"Timestamp: {packet.sniff_time}")
#     print(f"Protocol: {packet.highest_layer}")
    
#     if hasattr(packet, 'ip'):
#         print(f"IP: {packet.ip.src} → {packet.ip.dst}")

#     if hasattr(packet, 'tcp'):
#         print(f"TCP ports: {packet.tcp.srcport} → {packet.tcp.dstport}")
#     elif hasattr(packet, 'udp'):
#         print(f"UDP ports: {packet.udp.srcport} → {packet.udp.dstport}")

#     if hasattr(packet, 'http'):
#         print(f"HTTP Host: {packet.http.host}")
#         print(f"HTTP Request URI: {packet.http.request_uri}")

#     if hasattr(packet, 'tls'):
#         print("This is a TLS packet.")

#     if i >= 9:
#         break

# cap.close()

import pyshark

file_path = r'D:\UVA\CS6501 Network Security and Privacy\final_project\Filtered-Data\amazon_deal_b.pcapng'

cap = pyshark.FileCapture(file_path)

for i, packet in enumerate(cap):
    print(f"\n======= Packet {i+1} =======")
    
    print(f"Timestamp: {packet.sniff_time}")

    print(f"Protocol: {packet.highest_layer}")

    try:
        print(f"Packet Length: {packet.length} bytes")
    except AttributeError:
        print("Packet Length: [Unavailable]")

    if hasattr(packet, 'ip'):
        print(f"IP: {packet.ip.src} → {packet.ip.dst}")

    if hasattr(packet, 'tcp'):
        print(f"TCP Ports: {packet.tcp.srcport} → {packet.tcp.dstport}")
    elif hasattr(packet, 'udp'):
        print(f"UDP Ports: {packet.udp.srcport} → {packet.udp.dstport}")

    if hasattr(packet, 'http'):
        print(f"HTTP Host: {packet.http.host}")
        print(f"HTTP URI: {packet.http.request_uri}")

    if i >= 9:
        break

cap.close()
