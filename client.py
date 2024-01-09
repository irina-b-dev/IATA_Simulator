import socket
import threading

host = '127.0.0.1'
port = 5555

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

def receive_messages():
    while True:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            print(f"\nEve: {data}")
        except ConnectionError:
            print("Server disconnected.")
            break

receive_thread = threading.Thread(target=receive_messages)
receive_thread.start()

while True:
    message = input("Enter your command: ")
    client_socket.send(message.encode('utf-8'))
