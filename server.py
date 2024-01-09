import socket
import threading
import argparse

from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate

host = '127.0.0.1'
port = 5555

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen()

print(f"Server listening on {host}:{port}")

clients = {}
N = 5
initial_qubits = [0,1,2,3,4]
lock = threading.Lock()


def distribute_qubits_to_clients(num_qubits):
    with lock:
        num_clients = len(clients)
        qubits_per_client = num_qubits // num_clients
        remaining_qubits = num_qubits % num_clients

        for client_socket in clients:
            clients[client_socket]["qubits"].extend(initial_qubits[:qubits_per_client])
            # clients[client_socket]["qubits"] = 
            initial_qubits[:qubits_per_client] = []

        # Distribute remaining qubits to the first 'remaining_qubits' clients
        for i, client_socket in enumerate(list(clients.keys())[:remaining_qubits]):
            clients[client_socket]["qubits"].append(initial_qubits[i])
        initial_qubits[:remaining_qubits] = []


def parse_gate_command(command_args , client_socket, server=False):

    parser = argparse.ArgumentParser(description="Process a gate command")
    parser.add_argument('gate_name', type=str, help='Name of the gate')
    parser.add_argument('--starting_qubit', type=int, required=True, help='Starting qubit number')
    parser.add_argument('--control', nargs='+', type=int, default=[], help='List of control qubits')
    

    args = parser.parse_args(command_args[1:])

    # Process the parsed command
    process_gate_command(args.starting_qubit, args.control, args.gate_name, client_socket,server=server)


def handle_client(client_socket, address):
    print(f"Accepted connection from {address}")

     # Initialize the qubit list for this client
    with lock:
        alias = f"Alice-{len(clients)}"
        clients[client_socket] = {"qubits": [], "alias": alias}

        welcome_message = f"Hello {alias}! You are now connected."
        client_socket.send(welcome_message.encode('utf-8'))
    while True:
        try:
            command = client_socket.recv(1024).decode('utf-8')

            if not command:
                print(f"Connection from {clients[client_socket]['alias']} : {address} closed")
                with lock:
                    del clients[client_socket]
                client_socket.close()
                break

            if command.lower() == "exit":
                print(f"Connection from {address} closed")
                with lock:
                    del clients[client_socket]
                client_socket.close()
                break

            # Split the command into a list of arguments
            command_args = command.split()
            print(command_args)
            if command_args[0].lower() == "gate":
                parse_gate_command(command_args,client_socket)
            elif command_args[0].lower() == "mine":
                client_str = ' '.join(map(str, clients[client_socket]["qubits"]))
                send_message_to_client(client_socket, client_str)
            else:
                print(f"Unknown command received from {clients[client_socket]['alias']} -a {address}: {command}")
        except Exception as e:
            print(f"Error processing command from {clients[client_socket]['alias']} -a {address}: {e}")


def process_gate_command(starting_qubit, control_qubits, gate_name, client_socket, server=False):
    if not server:
        print(f"Processing command from {clients[client_socket]['alias']}:")
    print("Starting qubit:", starting_qubit)
    print("Control qubits:", control_qubits)
    print("Gate name:", gate_name)
    
    # initializing list
    target_list = []
    if server:
        target_list = initial_qubits
    else: 
        target_list = clients[client_socket]["qubits"]
    
    # initializing test list 
    test_list = control_qubits
    test_list.append(starting_qubit)

    print(target_list)
    print(test_list)
    
    check_qubit_ownership = all(ele in target_list for ele in test_list)

    if check_qubit_ownership:
        # TODO apply_gate
        print("applying gate to system")
        send_message_to_client(client_socket, "gate applied!")
    elif server:
        print("Qubits already distributed")
    else:
        send_message_to_client(client_socket, 
                               "You do not have access to those qubits, talk to your local Eve about this \n If you are unsure about which qubits you own, use command \"mine\" ")




def send_message_to_client(client_socket, message):
    try:
        client_socket.send(message.encode('utf-8'))
    except socket.error:
        print("Error sending message to client.")


# Start a separate thread to handle messages typed in the server terminal
def read_server_messages():
    while True:
        server_message = input("S:Enter a command: ")
        process_server_message(server_message)


def process_server_message(message):

    command_args = message.split()

    if command_args[0].lower() == 'gate':
        print("passed here")
        parse_gate_command(command_args)
    elif command_args[0].lower() == "distribute_qubits":
        try:
            num_qubits = 0
            if len(command_args) < 2:
                num_qubits = len(initial_qubits)
            else:
                num_qubits = int(command_args[1])
            distribute_qubits_to_clients(num_qubits)
        except ValueError:
            print("Invalid number of qubits specified in distribute_qubits command.")    
    else:
                print(f"Unknown command")


    # Example: Process internal server messages here


message_thread = threading.Thread(target=read_server_messages)
message_thread.start()

while True:
    client_socket, client_address = server_socket.accept()
    with lock:
        clients[client_socket] = {"qubits": [], "alias": f"Alice-{len(clients)}"}

    client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
    client_handler.start()
