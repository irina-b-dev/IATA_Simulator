from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates
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

def test_apply_gate():
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

    print("Apply control gate | Test Control-gate | Apply CS gate on control-qubit 0 and target-qubit 2!\n")

    gate = quantum_system.control_gate(control_qubit = 0, target_qubit = 2, gate_matrix = gates["S"][0])
    quantum_system.apply_gate(gate)
    quantum_system.print_state()
    assert np.allclose(quantum_system.state, [0, 0.5, -0.35355, +0.35355j, -0.25-0.25j, -0.35355j, -0.5j, 0])
    # https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B%22H%22%2C%22X%22%2C%22H%22%5D%2C%5B%22Z%22%2C%22Y%22%2C%22Z%5E%C2%BD%22%5D%2C%5B%22Z%5E%C2%BC%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22H%22%2C%22%E2%80%A2%22%5D%2C%5B%22Y%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BC%22%2C%22%E2%80%A2%22%5D%2C%5B1%2C%22Z%5E%C2%BD%22%2C%22%E2%80%A2%22%5D%2C%5B%22Swap%22%2C%22Swap%22%5D%2C%5B1%2C%22%E2%80%A2%22%2C%22X%22%5D%2C%5B%22X%22%2C%22%E2%80%A2%22%2C%22%E2%80%A2%22%5D%2C%5B%22Z%5E%C2%BD%22%2C1%2C%22%E2%80%A2%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D

    print(f"Checking qubit 0 state: {quantum_system.produce_specific_measurement(0)}\n")

    quantum_system.print_probabilities()
    print(f"After measurement: {quantum_system.produce_measurement()}")
    #quantum_system.plot_state_probabilities()

def test_import_export_gate():
    gate = gates["TOFFOLI"][0]
    Gate.export_gate('test', gate)
    toffoli_gate = Gate.import_gate('test')
    assert np.allclose(toffoli_gate, gates["TOFFOLI"][0])
    print("Test import/export successfull!")

def test_custom_gate():
    quantum_system = NQubitSystem(n_qubits = 3)
    quantum_system.initialize_state([0,1,1])
    quantum_system.print_state()

    quantum_system.apply_H_gate(0)
    quantum_system.print_state()
    #https://algassert.com/quirk#circuit=%7B%22cols%22%3A%5B%5B1%2C1%2C%22H%22%5D%5D%2C%22init%22%3A%5B1%2C1%5D%7D
    assert np.allclose(quantum_system.state, [0, 0, 0, 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)])

    dim, gate = Gate.create_custom_gate()
    assert np.allclose(gate, gates["H"][0])
    print("Test custom gate successfull!")

    gate = Gate.create_custom_user_gate()
    assert np.allclose(gate, gates["H"][0])
    print("Test custom user gate successfull!")

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
    #test_apply_gate()
    #test_import_export_gate()
    #test_custom_gate()
    test_noise()