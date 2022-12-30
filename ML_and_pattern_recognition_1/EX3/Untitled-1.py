import socket
hostname = socket.gethostname()
ipaddr = socket.gethostbyname(hostname)
address = 'Rautatienkatu 9, Tampere, Finland'
print(f"Address: {address} - Hostname: {hostname} - ipaddr: {ipaddr}")

