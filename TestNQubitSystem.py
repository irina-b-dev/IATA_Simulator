from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate
from pytket import Circuit
# from pytket.backends import Simulator
from convert_circuit import convert_IATA_to_qiskit, convert_IATA_to_tket, convert_IATA_to_cirq, run_qiskit_circuit, run_cirq_circuit_on_qiskit
from qiskit import QuantumCircuit
from pytket.circuit.display import render_circuit_jupyter
import matplotlib.pyplot as plt
import cirq
from pytket.extensions.qiskit import tk_to_qiskit
from collections import Counter
    
def test_init():
    print("Init test 1! Initializing a 3-qubit system as [0,0,0]\n")
    quantum_system = NQubitSystem(n_qubits = 3)
    expected_state = [1.0] + [0.0] * (2 ** quantum_system.n_qubits - 1)
    assert np.allclose(quantum_system.state, expected_state)
    quantum_system.print_state()
    print("Init test 1 passed!\n")

def test_initialize_state():
    quantum_system = NQubitSystem(n_qubits = 3)

    print("Initialize test 1! Initializing a 3-qubit system as [0,1,1]\n")
    quantum_system.initialize_state([0,1,1])
    assert np.allclose(quantum_system.state, [0, 0, 0, 1, 0, 0, 0, 0])
    quantum_system.print_state()
    print("Initialize test 1 passed!\n")

    print("Initialize test 2! Initializing a 3-qubit system as [0,0,1]\n")
    quantum_system.initialize_state([0,0,1])
    assert np.allclose(quantum_system.state, [0, 1, 0, 0, 0, 0, 0, 0])
    quantum_system.print_state()
    print("Initialize test 2 passed!\n")  

def test_basic_gates():
    quantum_system = NQubitSystem(n_qubits = 3)
    print("Apply gate | Initialize a 3-qubit system as [0,1,1]\n")
    quantum_system.initialize_state([0,1,1])
    quantum_system.print_state()

    print("Apply gate | Test H-Gate | Apply H gate on qubit 0!\n")
    quantum_system.apply_H_gate(0)
    quantum_system.print_state()
    #https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    assert np.allclose(quantum_system.state, [0, 0, 0, 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)])

    print("Apply gate | Test H-Gate | Apply H gate on qubit 2!\n")
    quantum_system.apply_H_gate(2)
    quantum_system.print_state()
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    assert np.allclose(quantum_system.state, [0, 0, 0.5, -0.5, 0, 0, 0.5, -0.5])

    print("Apply gate | Test X-gate | Apply X gate on qubit 1!\n")
    quantum_system.apply_X_gate(1)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0.5, -0.5, 0, 0, 0.5, -0.5, 0, 0])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test Y-gate | Apply Y gate on qubit 1!\n")
    quantum_system.apply_Y_gate(1)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5j, -0.5j, 0, 0, 0.5j, -0.5j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B1%2C%22Y%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test Z-gate | Apply Z gate on qubit 2!\n")
    quantum_system.apply_Z_gate(2)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5j, 0.5j, 0, 0, 0.5j, 0.5j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test T-gate | Apply T gate on qubit 2!\n")
    quantum_system.apply_T_gate(2)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5j, -0.35355+0.35355j, 0, 0, 0.5j, -0.35355+0.35355j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test S-gate | Apply S gate on qubit 0!\n")
    quantum_system.apply_S_gate(0)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5j, -0.35355+0.35355j, 0, 0, -0.5, -0.35355-0.35355j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CNOT-gate | Apply CNOT gate on control-qubit 1 and target-qubit 2!\n")
    quantum_system.apply_CNOT_gate(1)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, -0.35355+0.35355j, 0.5j, 0, 0, -0.35355-0.35355j, -0.5])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CH-gate | Apply CH gate on control-qubit 0 and target-qubit 1!\n")
    quantum_system.apply_CH_gate(0)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, -0.35355+0.35355j, 0.5j, -0.25-0.25j, -0.35355, +0.25+0.25j, +0.35355])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CY-gate | Apply CY gate on control-qubit 1 and target-qubit 2!\n")
    quantum_system.apply_CY_gate(1)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5, -0.35355-0.35355j, -0.25-0.25j, -0.35355, -0.35355j, -0.25+0.25j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CZ-gate | Apply CZ gate on control-qubit 0 and target-qubit 1!\n")
    quantum_system.apply_CZ_gate(0)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5, -0.35355-0.35355j, -0.25-0.25j, -0.35355, +0.35355j, +0.25-0.25j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CT-gate | Apply CT gate on control-qubit 1 and target-qubit 2!\n")
    quantum_system.apply_CT_gate(1)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5, -0.5j, -0.25-0.25j, -0.35355, +0.35355j, +0.35355])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CS-gate | Apply CS gate on control-qubit 0 and target-qubit 1!\n")
    quantum_system.apply_CS_gate(0)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0.5, -0.5j, -0.25-0.25j, -0.35355, -0.35355, +0.35355j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test SWAP-gate | Apply SWAP gate on qubits 1 and 2!\n")
    quantum_system.apply_SWAP_gate(1)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0.5, 0, -0.5j, -0.25-0.25j, -0.35355, -0.35355, +0.35355j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%2C%5B%22Swap%22%2C%22Swap%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply gate | Test CNOT10-gate | Apply CNOT10 gate on control-qubit 1 and target-qubit 0!\n")
    quantum_system.apply_CNOT10_gate(0)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0.5, -0.35355, +0.35355j, -0.25-0.25j, -0.35355, 0, -0.5j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%2C%5B%22Swap%22%2C%22Swap%22%5D%2C%5B1%2C%22%E2%80%A2%22%2C%22X%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    
    print("Apply gate | Test TOFFOLI-gate | Apply TOFFOLI gate on control-qubits 0,1 and target-qubit 2!\n")
    quantum_system.apply_TOFFOLI_gate(0)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0.5, -0.35355, +0.35355j, -0.25-0.25j, -0.35355, -0.5j, 0])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%2C%5B%22Swap%22%2C%22Swap%22%5D%2C%5B1%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("Apply control gate | Test custom control-gate | Apply CS gate on control-qubit 0 and target-qubit 2!\n")
    target_qubit = 2
    control_qubit = [0]
    # starting_qubit = np.min([target_qubit,np.min(control_qubit)])
    starting_qubit = 0
    gate_name = "S"
    control_gate_name = f"Controlled-{gate_name}_Cq{control_qubit}_Tq{target_qubit}"
    gate = quantum_system.control_gate(control_qubits = control_qubit, target_qubit = target_qubit, gate_matrix = gates_map[gate_name][0], name=control_gate_name)
    quantum_system.apply_gate(gate, starting_qubit = starting_qubit)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0.5, -0.35355, +0.35355j, -0.25-0.25j, -0.35355j, -0.5j, 0])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%2C%5B%22Swap%22%2C%22Swap%22%5D%2C%5B1%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BD%22%2C1%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print(f"Checking qubit 0 state: {quantum_system.produce_specific_measurement(0)}\n")

    quantum_system.print_probabilities()
    print(f"After measurement: {quantum_system.produce_measurement()}")
    print("\nBasic gates tests successful!")
    quantum_system.print_all_gates_applied()
    return quantum_system

def test_import_export_gate():
    gate = gates_map["TOFFOLI"][0]
    file_path = 'tests/gate'
    Gate.export_gate(file_path, gate)
    toffoli_gate = Gate.import_gate(file_path)
    assert np.allclose(toffoli_gate, gates_map["TOFFOLI"][0])
    print("Test import/export successfull!")

def test_custom_gate():
    quantum_system = NQubitSystem(n_qubits = 3)
    quantum_system.initialize_state([0,1,1])
    quantum_system.print_state()
    quantum_system.print_initial_qubits()

    print("\nApply hadamard gate on qubit 0!\n")
    quantum_system.apply_H_gate(0)
    quantum_system.print_state()
    #https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    assert np.allclose(quantum_system.state, [0, 0, 0, 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)])
    quantum_system.print_all_gates_applied()

    print("\nApply CT gate on control-qubit 0 and target-qubit 2!\n")
    target_qubit = 2
    control_qubit = [0]
    gate_name = "T"
    starting_qubit = 0
    # np.min([target_qubit,np.min(control_qubit)])
    control_gate_name = f"Controlled-{gate_name}_Cq{control_qubit}_Tq{target_qubit}"
    gate = quantum_system.control_gate(control_qubits = control_qubit, target_qubit = target_qubit, gate_matrix = gates_map[gate_name][0], name=control_gate_name)
    quantum_system.apply_gate(gate, starting_qubit = starting_qubit)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0, 0, 1/np.sqrt(2), 0, 0, 0, 0.5+0.5j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%2C%5B%22Z%5E%C2%BC%22%2C1%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    quantum_system.print_all_gates_applied()

    print("\nApply H gate on qubit 1!\n")
    dim, gate = Gate.create_custom_gate(dim = 1, gates_list=["H", "H", "H"], name="Custom_H")
    assert np.allclose(gate, gates_map["H"][0])
    quantum_system.apply_gate(gate, n_gate = dim, starting_qubit = 1)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()
    assert np.allclose(quantum_system.state, [0, 0.5, 0, -0.5, 0, 0.35355+0.35355j, 0, -0.35355-0.35355j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%2C%5B%22Z%5E%C2%BC%22%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("\nApply I gate on qubit 2!\n")
    dim, gate = Gate.create_custom_gate(dim = 1, gates_list=["H", "H", "H", "H"], name = "Custom_I")
    assert np.allclose(gate, np.eye(2))
    quantum_system.apply_gate(gate, n_gate = dim, starting_qubit = 2)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()
    assert np.allclose(quantum_system.state, [0, 0.5, 0, -0.5, 0, 0.35355+0.35355j, 0, -0.35355-0.35355j])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%2C%5B%22Z%5E%C2%BC%22%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("\nApply H gate on qubit 0!\n")
    gate = Gate.create_custom_user_gate(size = 2, matrix_values=np.array([[1, 1], [1, -1]]) / np.sqrt(2), name = "Custom_H_2")
    assert np.allclose(gate, gates_map["H"][0])
    quantum_system.apply_gate(gate, n_gate = dim, starting_qubit = 0)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()
    assert np.allclose(quantum_system.state, [0, 0.60355+0.25j, 0, -0.60355-0.25j, 0, 0.10355-0.25j, 0, -0.10355+0.25j], atol=1e-6)
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%2C%5B%22Z%5E%C2%BC%22%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%5D%2C%5B1%2C1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print("\nApply Z and S gates on qubit 1!\n")
    dim, gate = Gate.create_custom_gate(dim = 1, gates_list=["Z", "S"], name = "Custom_Z_S")
    quantum_system.apply_gate(gate, n_gate = dim, starting_qubit = 1)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()
    assert np.allclose(quantum_system.state, [0, 0.60355+0.25j, 0, -0.25+0.60355j, 0, 0.10355-0.25j, 0, 0.25+0.10355j], atol=1e-6)

    print("\nApply Y and T gates on qubit 2!\n")
    gate = Gate.create_custom_user_gate(size = 2, matrix_values=np.dot(gates_map["T"][0], gates_map["Y"][0]), name = "Custom_Y_T")
    quantum_system.apply_gate(gate, n_gate = 1, starting_qubit = 2)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()
    assert np.allclose(quantum_system.state, [0.25-0.60355j, 0, 0.60355+0.25j, 0, -0.25-0.10355j, 0, 0.10355-0.25j, 0], atol=1e-6)
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%2C%5B%22Z%5E%C2%BC%22%2C1%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%5D%2C%5B1%2C1%2C%22H%22%5D%2C%5B1%2C%22Z%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Y%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    print("\nCustom gates tests successful!")

    return quantum_system

def test_import_export_circuit_custom():
    quantum_system = test_custom_gate()
    json_file = "tests/circuit_custom.json"

    quantum_system.export_circuit(json_file)
    quantum_system_imported = NQubitSystem.import_circuit(json_file)
    quantum_system_imported.print_state()
    assert np.allclose(quantum_system_imported.state, [0.25-0.60355j, 0, 0.60355+0.25j, 0, -0.25-0.10355j, 0, 0.10355-0.25j, 0], atol=1e-6)
    quantum_system_imported.print_all_gates_applied()
    print("Import/Export test successful!")

def test_import_export_circuit_basic():
    quantum_system = test_basic_gates()
    json_file = "tests/circuit_basic.json"

    quantum_system.export_circuit(json_file)
    quantum_system_imported = NQubitSystem.import_circuit(json_file)
    quantum_system_imported.print_state()
    assert np.allclose(quantum_system_imported.state, [0, 0.5, -0.35355, +0.35355j, -0.25-0.25j, -0.35355j, -0.5j, 0], atol=1e-6)
    quantum_system_imported.print_all_gates_applied()
    print("Import/Export test successful!")
    

def test_noise():
    quantum_system = NQubitSystem(n_qubits = 3)
    quantum_system.initialize_state([0,1,1])
    quantum_system.print_state()

    quantum_system.apply_H_gate(0, noise = True)
    quantum_system.print_state()
    #https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    if np.allclose(quantum_system.state, [0, 0, 0, 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)]):
        print("Quantum noise NOT applied!")
    else:
        print("Quantum noise applied!")

def test_lab3_circuit():
    quantum_system = NQubitSystem(n_qubits = 4)
    quantum_system.initialize_state([0,1,1,0])
    quantum_system.print_state()
    quantum_system.print_initial_qubits()

    quantum_system.apply_H_gate(3)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_X_gate(1)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_S_gate(3)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_SWAP_gate(2)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_CNOT10_gate(1)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_CNOT10_gate(0)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_H_gate(0)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_CNOT_gate(0)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_S_gate(0)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    quantum_system.apply_CNOT_gate(1)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()

    
    target_qubit = 0
    control_qubit = [1]
    gate_name = "H"
    starting_qubit = 0
    # np.min([target_qubit,np.min(control_qubit)])
    control_gate_name = f"Controlled-{gate_name}_Cq{control_qubit}_Tq{target_qubit}"
    gate = quantum_system.control_gate(control_qubits = control_qubit, target_qubit = target_qubit, gate_matrix = gates_map[gate_name][0], name=control_gate_name)
    quantum_system.apply_gate(gate,starting_qubit = starting_qubit)
    quantum_system.print_state()


    target_qubit = 0
    control_qubit = 2
    swap_gate_name = f"Swap-{gate_name}_Cq{control_qubit}_Tq{target_qubit}"
    gate = quantum_system.swap_n_gate(control_qubit = control_qubit, target_qubit = target_qubit, name=swap_gate_name)
    quantum_system.apply_gate(gate)
    quantum_system.print_state()
    

    quantum_system.apply_H_gate(1)
    quantum_system.print_state()
    quantum_system.print_all_gates_applied()


def test_density_matrix():

    quantum_system = NQubitSystem(n_qubits=3)
    quantum_system.initialize_state([0, 1, 1])
    quantum_system.apply_H_gate(0)
    quantum_system.print_state()
    quantum_system.plot_density_matrix()
    print(quantum_system.calculate_density_matrix())

def test_convert_IATA_to_tket():
    # Create a 00 + 11 pair in IATA + TKET
    IATA_circuit = NQubitSystem(n_qubits = 2)
    IATA_circuit.initialize_state([0,0])
    IATA_circuit.print_state()

    print("Apply H gate on qubit 0!\n")
    IATA_circuit.apply_H_gate(0)
    IATA_circuit.print_state()

    # Convert IATA to TKET
    tket_circuit = convert_IATA_to_tket(IATA_circuit)

    print("Apply CNOT gate on qubit control 0 and target 1!\n")
    tket_circuit.CX(0, 1)

    # Run TKET by using qiskit backend
    # Convert TKET to Qiskit
    qiskit_circuit = tk_to_qiskit(tket_circuit)
    qiskit_circuit.measure_all()
    run_qiskit_circuit(qc = qiskit_circuit, execution_type = 0)

def test_convert_IATA_to_qiskit():
    # Create a 00 + 11 pair in IATA + Qiskit
    IATA_circuit = NQubitSystem(n_qubits = 2)
    IATA_circuit.initialize_state([0,0])
    IATA_circuit.print_state()

    print("Apply H gate on qubit 0!\n")
    IATA_circuit.apply_H_gate(0)
    IATA_circuit.print_state()

    qiskit_circuit = convert_IATA_to_qiskit(IATA_circuit)

    print("Apply CNOT gate on qubit control 0 and target 1!\n")
    qiskit_circuit.cx(0, 1)
    print(qiskit_circuit)
    qiskit_circuit.measure_all()
    run_qiskit_circuit(qc = qiskit_circuit, execution_type = 0)

def test_convert_IATA_to_cirq(backend):
    # Create a 00 + 11 pair in IATA + Qiskit
    IATA_circuit = NQubitSystem(n_qubits = 2)
    IATA_circuit.initialize_state([0,0])
    IATA_circuit.print_state()

    print("Apply H gate on qubit 0!\n")
    IATA_circuit.apply_H_gate(0)
    IATA_circuit.print_state()

    cirq_circuit = convert_IATA_to_cirq(IATA_circuit)
    print(cirq_circuit)
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circuit.append(cirq.CNOT(q0, q1))
    print("Apply CNOT gate on qubit control 0 and target 1!\n")
    cirq_circuit.append(cirq.measure(q0, q1, key='result'))
    print(cirq_circuit)

    # Run circuit using local Cirq local simulator
    if backend == 0:
        simulator = cirq.Simulator()
        result = simulator.run(cirq_circuit, repetitions=1024)
        counts = result.histogram(key='result')
        total_counts = dict(Counter({f'{i:02b}': count for i, count in counts.items()}))

        # Print the results
        print("\nResults:")
        print(total_counts)

        # Plotting the results
        plt.bar(total_counts.keys(), total_counts.values())
        plt.xlabel('State')
        plt.ylabel('Counts')
        plt.title('Measurement Results')
        plt.show()

    # Run circuit using Qiskit simulators
    elif backend == 1:
        run_cirq_circuit_on_qiskit(circuit = cirq_circuit, qubits = (q0,q1), execution_type = 2)

def test_run_qiskit_circuit():
    n_qubits = 2
    n_bits = n_qubits

    # Initialize a Quantum Circuit
    qc = QuantumCircuit(n_qubits, n_bits)

    # Create Entanglement
    qc.h(0)
    qc.cx(0, 1)

    # Specify which qubits to measure
    qc.measure([0,1], [0,1])

    run_qiskit_circuit(qc = qc, execution_type = 0)

def test_basic_circuit_in_qiskit():
    IATA_circuit = test_basic_gates()
    IATA_circuit.print_probabilities()
    qiskit_circuit = convert_IATA_to_qiskit(IATA_circuit)
    print(qiskit_circuit)
    qiskit_circuit.measure_all()
    run_qiskit_circuit(qc = qiskit_circuit, execution_type = 0)

def test_custom_circuit_in_qiskit():
    IATA_circuit = test_custom_gate()
    IATA_circuit.print_probabilities()
    qiskit_circuit = convert_IATA_to_qiskit(IATA_circuit)
    print(qiskit_circuit)
    qiskit_circuit.measure_all()
    run_qiskit_circuit(qc = qiskit_circuit, execution_type = 0)

if __name__ == "__main__":
    # test_init()
    # test_initialize_state()
    # test_basic_gates()
    # test_import_export_gate()
    # test_custom_gate()
    # test_lab3_circuit()
    # test_import_export_circuit_custom()
    # test_import_export_circuit_basic()
    # test_noise()
    # test_density_matrix()

    # test_run_qiskit_circuit()
    # test_convert_IATA_to_tket()
    # test_convert_IATA_to_qiskit()
    # test_convert_IATA_to_cirq(backend = 1)

    test_basic_circuit_in_qiskit()
    #test_custom_circuit_in_qiskit()