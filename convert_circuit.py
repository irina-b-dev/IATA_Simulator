from NQubitSystem import NQubitSystem
from pytket import Circuit, OpType
from pytket.circuit import Unitary1qBox, Unitary2qBox
import numpy as np
IATA_circuit = NQubitSystem.import_circuit("tests/circuit_custom.json")
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
        else:
            print(gate_name)
            print(single_gate)

    return tket_circuit


tket_circuit = convert_to_tket(IATA_circuit)
print(tket_circuit)
