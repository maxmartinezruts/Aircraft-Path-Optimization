#!/usr/bin/env python

import socket


TCP_IP = '52.43.121.77'
TCP_PORT = 2329
BUFFER_SIZE = 1024
MESSAGE = "Hello, World!"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
print('connected')
s.send(MESSAGE.encode())
data = s.recv(BUFFER_SIZE)
s.close()

print("received data:", data)