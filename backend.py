from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate

# Interogate user for the number of qubits and types
def initial_interogation():
    nr_qubits = int(input("Enter number of qubits:"))
    qubits = NQubitSystem(nr_qubits)
    qubit_values = input("Enter qubit values:")
    qubit_values = qubit_values.split()
    for i in range(len(qubit_values)):
        qubit_values[i] = int(qubit_values[i])
    qubits.initialize_state(qubit_values)
    return nr_qubits, qubits

# n, q = initial_interogation()
# print(n)
# q.print_initial_qubits()

# take input from the server and use it to apply operations and everything 

def apply_operations(target_list, starting_qubit, control_qubits, gate_name, gate_matrix = [1], noise = False, name = -1):
    match gate_name:
        case "H":
            target_list.apply_H_gate(starting_qubit, noise)
        case "X":
            target_list.apply_X_gate(starting_qubit, noise)
        case "Y":
            target_list.apply_Y_gate(starting_qubit, noise)
        case "Z":
            target_list.apply_Z_gate(starting_qubit, noise)
        case "T":
            target_list.apply_T_gate(starting_qubit, noise)
        case "S":
            target_list.apply_S_gate(starting_qubit, noise)
        case "CNOT":
            target_list.apply_CNOT_gate(starting_qubit, noise)
        case "CNOT10":
            target_list.apply_CNOT10_gate(starting_qubit, noise)
        case "CH":
            target_list.apply_CH_gate(starting_qubit, noise)
        case "CY":
            target_list.apply_CY_gate(starting_qubit, noise)
        case "CZ":
            target_list.apply_CZ_gate(starting_qubit, noise)
        case "CT":
            target_list.apply_CT_gate(starting_qubit, noise)
        case "CS":
            target_list.apply_CS_gate(starting_qubit, noise)
        case "SWAP":
            target_list.apply_SWAP_gate(starting_qubit, noise)
        case "TOFFOLI":
            target_list.apply_TOFFOLI_gate(starting_qubit, noise)
        case "CU":
            gate_init = target_list.control_gate(control_qubits, starting_qubit, gate_matrix, name)
            target_list.apply_gate(gate_init, starting_qubit = starting_qubit)
        case "SWAPN":
            gate_init = target_list.swap_n_gate(control_qubit = control_qubits, target_qubit = starting_qubit, name=name)
            target_list.apply_gate(gate_init)
    target_list.print_state()

