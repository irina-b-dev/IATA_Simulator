from NQubitSystem import NQubitSystem
from convert_circuit import convert_IATA_to_qiskit
from convert_circuit import convert_IATA_to_tket
from convert_circuit import convert_IATA_to_cirq
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import cirq
json_file = "tests/circuit_basic.json"
IATA_circuit = NQubitSystem.import_circuit(json_file)
tket_circuit = convert_IATA_to_tket(IATA_circuit)
print(tket_circuit)
qiskit_circuit = convert_IATA_to_qiskit(IATA_circuit)
print(qiskit_circuit)
cirq_circuit = convert_IATA_to_cirq(IATA_circuit)
print(cirq_circuit)
