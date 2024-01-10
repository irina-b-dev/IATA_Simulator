from NQubitSystem import NQubitSystem
from pytket import Circuit, OpType
from pytket.circuit import Unitary1qBox, Unitary2qBox
import numpy as np
import re
from qiskit import QuantumCircuit
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.extensions.cirq import tk_to_cirq
import cirq

common_gates = ["X", "Y", "Z", "H", "S", "T", "CNOT", "CH",
                "CY", "CZ", "CT", "CS", "SWAP", "CNOT10", "TOFFOLI"]


def apply_gate_tket(tket_circuit, gate_name,  qubits_affected):
    if gate_name == "X":
        tket_circuit.X(qubits_affected[0])
    elif gate_name == "Y":
        tket_circuit.Y(qubits_affected[0])
    elif gate_name == "Z":
        tket_circuit.Z(qubits_affected[0])
    elif gate_name == "H":
        tket_circuit.H(qubits_affected[0])
    elif gate_name == "S":
        tket_circuit.S(qubits_affected[0])
    elif gate_name == "T":
        tket_circuit.T(qubits_affected[0])
    elif gate_name == "CNOT":
        tket_circuit.CX(qubits_affected[0], qubits_affected[1])
    elif gate_name == "CH":
        tket_circuit.CH(qubits_affected[0], qubits_affected[1])
    elif gate_name == "CY":
        tket_circuit.CY(qubits_affected[0], qubits_affected[1])
    elif gate_name == "CZ":
        tket_circuit.CZ(qubits_affected[0], qubits_affected[1])
    elif gate_name == "CT":
        angle = np.pi / 4
        tket_circuit.add_gate(OpType.CRz, angle, [
                              qubits_affected[0], qubits_affected[1]])
    elif gate_name == "CS":
        tket_circuit.CS(qubits_affected[0], qubits_affected[1])
    elif gate_name == "SWAP":
        tket_circuit.SWAP(qubits_affected[0], qubits_affected[1])
    elif gate_name == "CNOT10":
        # CNOT with control and target swapped
        tket_circuit.CX(qubits_affected[1], qubits_affected[0])
    elif gate_name == "TOFFOLI":
        tket_circuit.CCX(qubits_affected[0],
                         qubits_affected[1], qubits_affected[2])


def apply_custom_gate(tket_circuit, custom_gate, qubits_affected):
    if len(qubits_affected) == 1:
        custom_1q_gate = Unitary1qBox(custom_gate)
        tket_circuit.add_gate(custom_1q_gate, qubits_affected)
    elif len(qubits_affected) == 2:
        custom_2q_gate = Unitary2qBox(custom_gate)
        tket_circuit.add_gate(custom_2q_gate, qubits_affected)


def apply_controlled_gate(tket_circuit, gate_string):
    # Parse the gate_string
    match = re.match(r"Controlled-(\w+)_Cq(\d+)_Tq(\d+)", gate_string)
    if not match:
        raise ValueError("Invalid gate string format.")

    gate, control_qubit, target_qubit = match.groups()

    # Convert qubit indices to integers
    control_qubit = int(control_qubit)
    target_qubit = int(target_qubit)

    # Apply the controlled gate to the circuit
    if gate in common_gates and gate != "T":
        # Dynamically call the gate method
        getattr(tket_circuit, f'C{gate}')(control_qubit, target_qubit)
    elif gate == "T":
        angle = np.pi / 4
        tket_circuit.add_gate(OpType.CRz, angle, [
                              control_qubit, target_qubit])
    else:
        raise ValueError(f"Unsupported gate '{gate}'.")


def convert_to_tket(IATA_circuit):
    n = IATA_circuit.n_qubits
    tket_circuit = Circuit(n)
    dummy_index = IATA_circuit.index
    for i in range(0, n, 1):
        if dummy_index % 2 == 1:
            tket_circuit.X(i)
        dummy_index = dummy_index/2
    for gate_applied in IATA_circuit.gates_applied:
        idx, gate_name, qubits_affected, single_gate, system_gate = gate_applied
        if (gate_name in common_gates):
            apply_gate_tket(tket_circuit, gate_name, qubits_affected)
        elif gate_name[:10] == "Controlled":
            apply_controlled_gate(tket_circuit, gate_name)
        else:
            apply_custom_gate(tket_circuit, single_gate, qubits_affected)
    return tket_circuit


def convert_to_qiskit(IATA_circuit):
    tket_circuit = convert_to_tket(IATA_circuit)
    return tk_to_qiskit(tket_circuit)


def convert_to_cirq(IATA_circuit):
    tket_circuit = convert_to_tket(IATA_circuit)
    return tk_to_cirq(tket_circuit)
