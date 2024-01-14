import socket
import threading
import sys
import os

host = '127.0.0.1'
port = 5555

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))
shutdown_flag = threading.Event()

def receive_messages(client_socket):
    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                print("Server disconnected. Exiting.")
                sys.exit(0)
            else:
                decoded_data = data.decode('utf-8')
                if decoded_data.lower() == 'exit':
                    print("Server requested exit. Exiting.")
                    os._exit(0)
                print(f"Received data from server: {decoded_data}")

        except socket.error:
            print("Error receiving data from server.")
            sys.exit(1)

# receive_thread = threading.Thread(target=receive_messages)
# receive_thread.start()

receive_thread = threading.Thread(target=receive_messages, args=(client_socket,))
receive_thread.daemon = True  # Set as daemon thread
receive_thread.start()


# Continue with the client's main logic and user input
while True:
    user_input = input("Enter command: ")

    # Send user input to the server
    client_socket.send(user_input.encode('utf-8'))

    # Optionally, check for a specific exit command
    if user_input.lower() == 'exit':
        print("Exiting as per user request.")
        sys.exit(0)

