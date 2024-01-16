import socket
import threading
import argparse

from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate
import backend
import sys
import signal
import time
import re

import os
from convert_circuit import convert_IATA_to_qiskit


host = '127.0.0.1'
port = 5555

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen()

print(f"Server listening on {host}:{port}")

clients = {}
#N = 5
#initial_qubits = [0,1,2,3,4]
num_intial, system = backend.initial_interogation()
initial_qubits = []
for i in range(0, num_intial):
    initial_qubits.append(i)
noise_input = input("Should noise be applied? [y/n]")
if noise_input == "y":
    noise = True
else:
    noise = False 

lock = threading.Lock()
shutdown_flag = threading.Event()


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return(True) 
    return(False)


def is_power_of_two(n):
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def parse_complex_number(input_str):
    """Parse complex numbers in the format 'a + bj'."""
    real_part, imag_part = map(float, re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', input_str))
    return complex(real_part, imag_part)

def get_unitary_matrix():
    """Get a square unitary matrix with dimensions that are powers of 2 from the user."""
    # while True:
    try:
        matrix_name = input(f"Enter the name of your gate ")
        n = int(input(f"Enter the dimension of the {matrix_name} matrix (power of 2): "))
        if not is_power_of_two(n):
            raise ValueError("Please enter a dimension that is a power of 2.")
        
        # Input the matrix elements
        matrix = np.empty((n, n), dtype=complex)
        print(f"Enter the elements of the {matrix_name} matrix separated by spaces:")
        for i in range(n):
            row_input = input(f"Row {i+1}: ")
            elements = row_input.split()
            if len(elements) != n:
                raise ValueError(f"Please enter exactly {n} elements for row {i+1}.")
            matrix[i, :] = [parse_complex_number(e) for e in elements]
        
        # Check if the matrix is unitary
        product = np.dot(matrix, np.conj(matrix.T))
        identity_matrix = np.eye(n)
        if not np.allclose(product, identity_matrix):
            raise ValueError(f"The entered {matrix_name} matrix is not unitary.")
        
        return matrix_name, np.conj(np.conj(matrix)), n
    
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def show_circuit():
    qiskit_circuit = convert_IATA_to_qiskit(system)
    print(qiskit_circuit)
    pass
    

def show_probs(client_socket, qubit_array):

    target_list = clients[client_socket]["qubits"].copy()
    
    # initializing test list 
    test_list = qubit_array.copy()

    print(target_list)
    print(test_list)
    
    check_qubit_ownership = all(ele in target_list for ele in test_list)
    strqp = "\n"
    if check_qubit_ownership:
        for q in qubit_array:
            prob, measurment = system.produce_specific_measurement(q)
            prob100 = int(round(prob*100))
            lines = f"Q_{q}\t"
            for line in range(0,prob100):
                lines += "|"
            strprob = f" {prob100}% \n"
            lines  += strprob   
            strqp += lines

        send_message_to_client(client_socket, 
                            strqp) 

    else:
        send_message_to_client(client_socket, 
                            "You do not have access to those qubits, talk to your local Eve about this \n If you are unsure about which qubits you own, use command \"mine\" ") 



def show_probs_server(qubit_array):

    target_list = initial_qubits.copy()
    
    # initializing test list 
    test_list = qubit_array.copy()

    print(target_list)
    print(test_list)
    
    check_qubit_ownership = all(ele in target_list for ele in test_list)
    strqp = "\n"
    if check_qubit_ownership:
        for q in qubit_array:
            prob, measurment = system.produce_specific_measurement(q)
            prob100 = int(round(prob*100))
            lines = f"Q_{q}\t"
            for line in range(0,prob100):
                lines += "|"
            strprob = f" {prob100}% \n"
            lines  += strprob   
            strqp += lines

        
        print(strqp) 

    else:
       print("you do not have those qubits") 


def initialize_teleportation():
    socket_Alice = 0
    socket_Bob = 0
    found_alice = False
    while not found_alice:
        alice = input("Who is Alice:")
        # with lock:
        socket_Alice = get_socket_id_from_alias(alice)
        if not socket_Alice:
            print("receiver not found!")
        else:
            found_alice = True
    found_bob = False

    while not found_bob:
        bob = input("Who is Bob:")
        # with lock:
        socket_Bob = get_socket_id_from_alias(bob)
        if not socket_Bob:
            print("receiver not found!")
        else:
            found_bob = True

    nr_qubit = int(input("Which qubit to teleport:"))
    send_qubits_to_client(clients[socket_Alice]["alias"], [nr_qubit])
    
    qubit_Alice = int(input("Which qubit to send to Alice:"))
    send_qubits_to_client(clients[socket_Alice]["alias"], [qubit_Alice])
    
    qubit_Bob = int(input("Which qubit to send to Bob:"))
    send_qubits_to_client(clients[socket_Bob]["alias"], [qubit_Bob])

    print("entangleling...")
    entangle_for_teleportation(nr_qubit,qubit_Alice,qubit_Bob)

    
    pass

def apply_correction(client_socket, measurment_psi, measurment_qubitA, qubitB):
    if measurment_qubitA > 0:
        backend.apply_operations(target_list=system, starting_qubit=qubitB, gate_name="X")
    if measurment_psi > 0:
        backend.apply_operations(target_list=system, starting_qubit=qubitB, gate_name="Z")

    send_message_to_client(client_socket, 
                            f"Measuring your qubit...")
    prob, measurment_result_qubitB = system.produce_specific_measurement(qubitB)
    send_message_to_client(client_socket, 
                            f"Let's see what is the value of psi ...\n{measurment_result_qubitB}")
    

def entangle_for_teleportation(psi, qubitA, qubitB):
    backend.apply_operations(target_list=system, starting_qubit=qubitA, control_qubits=[], gate_name="H")
    backend.apply_operations(target_list=system, starting_qubit=qubitB, control_qubits=[qubitA], gate_name="X")
    backend.apply_operations(target_list=system, starting_qubit=qubitA, control_qubits=[psi], gate_name="X")
    backend.apply_operations(target_list=system, starting_qubit=psi, control_qubits=[], gate_name="H")
    pass

def measure_qubits_for_client(socket_client, qubit_array, collapse = False ):
    target_list = clients[socket_client]["qubits"].copy()

    # initializing test list 
    test_list = qubit_array.copy()

    print(target_list)
    print(test_list)

    check_qubit_ownership = all(ele in target_list for ele in test_list)

    if check_qubit_ownership:
        with lock:
            for qubit in qubit_array:
                if collapse:
                    measurement = system.collapse_measurement(qubit)
                    send_message_to_client(socket_client, 
                                f"Qubit {qubit} now collapsed into {measurement}")
                else:
                    prob, measurement = system.produce_specific_measurement(qubit)
                    send_message_to_client(socket_client, 
                                f"Measurement for {qubit} is {measurement}")

def measure_for_teleportation(psi, qubitA, socket_sender, socket_receiver):
    # TODO should collapse state just to change probabilities
    measurment_result_psi = system.collapse_measurement(psi)
    
    measurment_result_qubitA = system.collapse_measurement(qubitA)

    qubitB = int(clients[socket_receiver]["qubits"][0])
    
    send_message_to_client(socket_sender, 
                            f"Measured psi as {measurment_result_psi} and qubitA as {measurment_result_qubitA}")
    send_message_to_client(socket_receiver, 
                            f"You received measurement from Alice for psi as {measurment_result_psi} and qubitA as {measurment_result_qubitA}")
    send_message_to_client(socket_receiver, 
                            f"applying correction on your {qubitB} qubit")
    apply_correction(socket_receiver,measurment_result_psi,measurment_result_qubitA,qubitB)

    pass


def send_qubits_to_client(receiver_alias, qubit_array):
    
    # with lock:
    socket_receiver = get_socket_id_from_alias(receiver_alias)
    if not socket_receiver:
        print("receiver not found!")
        return
    target_list = initial_qubits.copy()
    # initializing test list 
    test_list = qubit_array.copy()

    check_qubit_ownership = all(ele in target_list for ele in test_list)

    if check_qubit_ownership:
        clients[socket_receiver]["qubits"].extend(qubit_array)
        for i in qubit_array:
            if i in initial_qubits:
                initial_qubits.remove(i)

        send_message_to_client(socket_receiver, 
                            f"You received {qubit_array} from Eve") 

    else:
        print("you do not have access to those qubits")

def send_qubits_to(sender_alias, receiver_alias, qubit_array):
    sender_socket = get_socket_id_from_alias(sender_alias)
    # with lock:
    socket_receiver = get_socket_id_from_alias(receiver_alias)
    if not socket_receiver:
        send_message_to_client(sender_socket, "receiver not found!")
        return
    
    target_list = clients[sender_socket]["qubits"].copy()
    
    # initializing test list 
    test_list = qubit_array.copy()

    print(target_list)
    print(test_list)
    
    check_qubit_ownership = all(ele in target_list for ele in test_list)

    if check_qubit_ownership:
        clients[socket_receiver]["qubits"].extend(qubit_array)
        for i in qubit_array:
            if i in clients[sender_socket]["qubits"]:
                clients[sender_socket]["qubits"].remove(i)

        send_message_to_client(socket_receiver, 
                            f"You received {qubit_array} from {sender_alias}") 

    else:
        send_message_to_client(sender_socket, 
                            "You do not have access to those qubits, talk to your local Eve about this \n If you are unsure about which qubits you own, use command \"mine\" ") 

        
def get_socket_id_from_alias(alias):
    with lock:
        for socket_id, client_info in clients.items():
            if client_info["alias"] == alias:
                return socket_id
    return None


def distribute_qubits_to_clients(num_qubits):
    with lock:
        num_clients = len(clients)
        if num_clients > 0:
            qubits_per_client = num_qubits // num_clients
            remaining_qubits = num_qubits % num_clients

            for client_socket in clients:
                clients[client_socket]["qubits"] = initial_qubits[:qubits_per_client]
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
    
    try :
        args = parser.parse_args(command_args[1:])
        print(f"args control {args.control}")
        print(f"args starting qubit {args.starting_qubit}")
        # Process the parsed command
        process_gate_command(args.starting_qubit, args.control, args.gate_name, client_socket,server=server, gate_matrix=[], name=-1)
        
    except (SystemExit, argparse.ArgumentError) as e:
        print(f"Error parsing command-line arguments: {e}")
        if not server:
            send_message_to_client(client_socket,f"Error parsing command-line arguments: {e} \n{parser.format_help()}")


def parse_send_command(command_args , client_socket, server=False):

    parser = argparse.ArgumentParser(description="Process a send command")
    parser.add_argument('qubits', nargs='+', type=int, help='list of qubits to send')
    parser.add_argument('--to', type=str, required=True, help='alias of the receiver')
    
    try :
        args = parser.parse_args(command_args[1:])

        # Process the parsed command
        send_qubits_to(clients[client_socket]['alias'], args.to, args.qubits)
    except (SystemExit, argparse.ArgumentError) as e:
        print(f"Error parsing command-line arguments: {e}")
        send_message_to_client(client_socket,f"Error parsing command-line arguments: {e} \n{parser.format_help()}")

def parse_send_to_client_command(command_args):

    parser = argparse.ArgumentParser(description="Process a send command")
    parser.add_argument('qubits', nargs='+', type=int, help='list of qubits to send')
    parser.add_argument('--to', type=str, required=True, help='alias of the receiver')
    
    try : 
        args = parser.parse_args(command_args[1:])

        # Process the parsed command
        send_qubits_to_client(args.to, args.qubits)
    except (SystemExit, argparse.ArgumentError) as e:
        print(f"Error parsing command-line arguments: {e}")
    
def parse_measure_command(command_args , client_socket, server=False):

    parser = argparse.ArgumentParser(description="Process measure command")
    parser.add_argument('qubits', nargs='+', type=int, help='list of qubits to measure')
    parser.add_argument('--collapse', type=str, default="false" , help='true or false')
    
    try : 
        args = parser.parse_args(command_args[1:])
        collapse = False
        if args.collapse.lower() == "true":
            measure_qubits_for_client(client_socket,args.qubits,collapse=True)
        else:
            measure_qubits_for_client(client_socket,args.qubits)

    except (SystemExit, argparse.ArgumentError) as e:
        print(f"Error parsing command-line arguments: {e}")
        send_message_to_client(client_socket,f"Error parsing command-line arguments: {e}\n{parser.format_help()}")


def parse_measure_and_send_command(command_args , client_socket, server=False):

    parser = argparse.ArgumentParser(description="Process measure and send command")
    parser.add_argument('--psi', type=int, help='qubit psi')
    parser.add_argument('--qubitA', type=int, help='qubit A')
    parser.add_argument('--to', type=str, required=True , help='alias of the receiver')
    
    try:
        args = parser.parse_args(command_args[1:])
        socket_receiver = get_socket_id_from_alias(args.to)
        if not socket_receiver:
            send_message_to_client(client_socket, "Alias for receiver not found!")
            return
        
        send_message_to_client(client_socket, f"Measuring and sending to {args.to}")
        measure_for_teleportation(args.psi, args.qubitA, client_socket,socket_receiver)

    except (SystemExit, argparse.ArgumentError) as e:
        print(f"Error parsing command-line arguments: {e}")
        send_message_to_client(client_socket,f"Error parsing command-line arguments: {e}\n{parser.format_help()}")

def parse_show_probs_command(command_args , client_socket, server=False):

    parser = argparse.ArgumentParser(description="Process show probabilities command")
    parser.add_argument('qubits', nargs='+', type=int, help='list of qubits to measure')
    try: 
        args = parser.parse_args(command_args[1:])
        if server :
            show_probs_server(args.qubits)
        else:
            show_probs(client_socket, args.qubits)
    except (SystemExit, argparse.ArgumentError) as e:
        print(f"Error parsing command-line arguments: {e}")
        if not server:
            send_message_to_client(client_socket,f"Error parsing command-line arguments: {e} \n{parser.format_help()}")


def handle_client(client_socket, address):
    print(f"Accepted connection from {address}")

     # Initialize the qubit list for this client
    with lock:
        alias = f"Alice-{len(clients)}"
        clients[client_socket] = {"qubits": [], "alias": alias}

        welcome_message = f"Hello {alias}! You are now connected. \n supported commands are: \ngate, mine, send --to, measure [--collapse], measure_and_send --to, show_prob, exit "
        client_socket.send(welcome_message.encode('utf-8'))
    while not shutdown_flag.is_set():
        try:
            command = client_socket.recv(1024).decode('utf-8')

            if not command:
                print(f"Connection from {clients[client_socket]['alias']} : {address} closed")
                with lock:
                    del clients[client_socket]
                client_socket.close()
                break

            elif command.lower() == "exit":
                print(f"Connection from {address} closed")
                with lock:
                    del clients[client_socket]
                client_socket.close()
                return

            # Split the command into a list of arguments
            command_args = command.split()

            print(command_args)
            if command_args[0].lower() == "gate":
                parse_gate_command(command_args,client_socket)

            elif command_args[0].lower() == "mine":
                client_str = ' '.join(map(str, clients[client_socket]["qubits"]))
                send_message_to_client(client_socket, client_str)

            elif command_args[0].lower() == "send":
                parse_send_command(command_args, client_socket)
                
            elif command_args[0].lower() == "measure":
                parse_measure_command(command_args,client_socket)
                
            elif command_args[0].lower() == "measure_and_send":
                parse_measure_and_send_command(command_args,client_socket)

            elif command_args[0].lower() == "show_prob":
                parse_show_probs_command(command_args,client_socket)
            else:
                print(f"Unknown command received from {clients[client_socket]['alias']} -a {address}: {command}")
        except Exception as e:
            print(f"Error processing command from {clients[client_socket]['alias']} -a {address}: {e}")
            send_message_to_client(client_socket, f"supported commands are: \ngate, mine, send --to, measure [--collapse], measure_and_send --to, show_prob, exit")



def process_gate_command(starting_qubit, control_qubits, gate_name, client_socket, server=False, gate_matrix=[], name=-1):
    if not server:
        print(f"Processing command from {clients[client_socket]['alias']}:")
    print("Starting qubit:", starting_qubit)
    print("Control qubits:", control_qubits)
    print("Gate name:", gate_name)
    
    # initializing list
    target_list = []
    if server:
        target_list = initial_qubits.copy()
    else: 
        target_list = clients[client_socket]["qubits"].copy()

    target_qubits = []
    gate = gates_map[gate_name][0]
    n_gate = gates_map[gate_name][1]
    
    for i in range(0,n_gate):
        target_qubits.append(starting_qubit+i)


    
    
    if common_member(control_qubits, target_qubits):
        if server:
            print("Target qubits cannot be the same as control qubits")
        else:
            send_message_to_client(client_socket, 
                                "Target qubits cannot be the same as control qubits")
        return
    

   

    # initializing test list 
    test_list = control_qubits.copy()
    test_list.extend(target_qubits)
    

    print(target_list)
    print(test_list)
    
    check_qubit_ownership = all(ele in target_list for ele in test_list)

    if check_qubit_ownership:
        # TODO apply_gate
        backend.apply_operations(target_list=system, starting_qubit=starting_qubit, control_qubits=control_qubits, gate_name=gate_name, gate_matrix=gate_matrix, noise=noise, name=name)
        if not server:
          send_message_to_client(client_socket, "applied gate")
        print("applying gate to system")
    elif server:
        print("You do not have access to those qubits")
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
    if len(command_args) > 0:
        if command_args[0].lower() == 'exit':
            sys.exit(0)
        elif command_args[0].lower() == 'gate':
            print("passed here")
            parse_gate_command(command_args, 0, server=True)
        elif command_args[0].lower() == 'mine':
            print(initial_qubits)
        
        elif command_args[0].lower() == 'send':
            parse_send_to_client_command(command_args)

        elif command_args[0].lower() == 'show_prob':
            parse_show_probs_command(command_args,0,True)

        elif command_args[0].lower() == 'show_circuit':
            show_circuit()

        elif command_args[0].lower() == 'new_gate':
            name, matrix, dimension = get_unitary_matrix()
            print(matrix)
            print(dimension)
            Gate.create_custom_user_gate(size=dimension, matrix_values=matrix,name=name)
        elif command_args[0].lower() == 'print_gate':
            print(gates_map[command_args[1]][0])
        
        elif command_args[0].lower() == 'initialize_teleportation':  
            initialize_teleportation()
        elif command_args[0].lower() == "distribute_qubits":
            num_qubits = 0
            print(len(command_args))
            if len(command_args) > 1:
                num_qubits = int(command_args[1])
            else:
                print(len(initial_qubits))
                num_qubits = len(initial_qubits)
                
            distribute_qubits_to_clients(num_qubits)  
        else:
            print(f"Unknown command,\n supported commands are: \ndistribute_qubits, gate, mine, send --to, initialize_teleportation, show_circuit exit ")


    # Example: Process internal server messages here

def server_shutdown(signum, frame):
    print("\nCtrl+C received. Closing all connections...")

    for client in clients:
            data = "exit"
            client.send(data.encode('utf-8'))

    time.sleep(1)
    os._exit(0)


# Set the signal handler for Ctrl+C
signal.signal(signal.SIGINT, server_shutdown)


message_thread = threading.Thread(target=read_server_messages)
message_thread.start()

while not shutdown_flag.is_set():

    client_socket, client_address = server_socket.accept()
    with lock:
        clients[client_socket] = {"qubits": [], "alias": f"Alice-{len(clients)}"}

    client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
    client_handler.start()

    
