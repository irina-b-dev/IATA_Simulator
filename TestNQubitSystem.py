from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate

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
    control_qubit = 0
    gate_name = "S"
    control_gate_name = f"Controlled-{gate_name}_Cq{control_qubit}_Tq{target_qubit}"
    gate = quantum_system.control_gate(control_qubit = control_qubit, target_qubit = target_qubit, gate_matrix = gates_map[gate_name][0], name=control_gate_name)
    quantum_system.apply_gate(gate)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0.5, -0.35355, +0.35355j, -0.25-0.25j, -0.35355j, -0.5j, 0])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%2C%5B%22Swap%22%2C%22Swap%22%5D%2C%5B1%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BD%22%2C1%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print(f"Checking qubit 0 state: {quantum_system.produce_specific_measurement(0)}\n")

    quantum_system.print_probabilities()
    print(f"After measurement: {quantum_system.produce_measurement()}")
    print("\nBasic gates tests successful!")
    quantum_system.print_all_gates_applied()

def test_import_export_gate():
    gate = gates_map["TOFFOLI"][0]
    Gate.export_gate('test', gate)
    toffoli_gate = Gate.import_gate('test')
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
    control_qubit = 0
    gate_name = "T"
    control_gate_name = f"Controlled-{gate_name}_Cq{control_qubit}_Tq{target_qubit}"
    gate = quantum_system.control_gate(control_qubit = control_qubit, target_qubit = target_qubit, gate_matrix = gates_map[gate_name][0], name=control_gate_name)
    quantum_system.apply_gate(gate)
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

    print("\nCustom gates tests successful!")

    return quantum_system

def test_export_circuit():
    quantum_system = test_custom_gate()
    quantum_system.export_circuit("circuit.json")
    quantum_system_imported = NQubitSystem.import_circuit("circuit.json")
    
def test_import_circuit():
    pass

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

if __name__ == "__main__":
    #test_init()
    #test_initialize_state()
    #test_basic_gates()
    #test_import_export_gate()
    #test_custom_gate()
    #test_export_circuit()
    test_import_circuit()
    #test_noise()
