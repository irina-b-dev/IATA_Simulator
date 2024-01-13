import numpy as np
from constants import gates_map
from scipy.linalg import norm
import matplotlib.pyplot as plt
import random
import json
import qiskit.quantum_info as qi

# The n-qubit system's state is as an array of 2^n coefficients (one for each possible value of the qubits).


class NQubitSystem:
    def is_valid_state(self, tolerance=1e-10):
        # Ensure it's a column vector (shape is (n, 1) or (n,))
        if len(self.state.shape) > 2 or (len(self.state.shape) == 2 and self.state.shape[1] != 1):
            return False

        # Check if the sum of the squares of the absolute values is approximately 1
        return np.isclose(np.sum(np.abs(self.state)**2), 1, atol=tolerance)

    def print_state(self):
        print(f"======= {self.n_qubits}-qubit system's state =======")
        for i, prob in enumerate(self.state):
            binary_string = format(i, f"0{self.n_qubits}b")
            print(f"{binary_string}: {prob:.6f}")
        print("\n")

    def print_probabilities(self):
        print(f"======= {self.n_qubits}-qubit system's probabilities =======")
        probabilities = np.abs(self.state)**2
        for i, prob in enumerate(probabilities):
            binary_string = format(i, f"0{self.n_qubits}b")
            print(f"{binary_string}: {prob:.6f}")
        print("\n")

    def print_all_gates_applied(self):
        if len(self.gates_applied) == 0:
            return
        print("Gates applied:")
        for gate_applied in self.gates_applied:
            idx, gate_name, qubits_affected, single_gate, system_gate = gate_applied
            print(f"Step {idx}: {gate_name} on qubits {qubits_affected}")
        print("")

    def complex_encoder(self, z):
        return {"real": float(z.real), "imag": float(z.imag)}

    def complex_decoder(self, d):
        return complex(d["real"], d["imag"])

    def export_circuit(self, file_path):
        if len(self.gates_applied) == 0:
            return
        circuit = {
            "initial_state": self.initial_state,
            "gates_applied": []
        }
        for gate_applied in self.gates_applied:
            idx, gate_name, qubits_affected, single_gate, system_gate = gate_applied

            single_gate = np.array([[self.complex_encoder(z)
                                   for z in row] for row in single_gate]).tolist()
            system_gate = np.array([[self.complex_encoder(z)
                                   for z in row] for row in system_gate]).tolist()

            circuit["gates_applied"].append({
                "idx": idx,
                "gate_name": gate_name,
                "qubits_affected": qubits_affected,
                "single_gate": single_gate,
                "system_gate": system_gate
            })
        circuit_json = json.dumps(circuit, indent=4)
        with open(file_path, "w") as json_file:
            json_file.write(circuit_json)

        print(f"Saved JSON data to {file_path}")

    @staticmethod
    def import_circuit(file_path):
        with open(file_path, "r") as json_file:
            circuit = json.load(json_file)
            initial_state = circuit["initial_state"]
            quantum_system = NQubitSystem(n_qubits=len(initial_state))
            quantum_system.initialize_state(initial_state)
            gates_applied = circuit["gates_applied"]
            for gate_applied in gates_applied:
                idx = gate_applied["idx"]
                gate_name = gate_applied["gate_name"]
                qubits_affected = gate_applied["qubits_affected"]
                single_gate = gate_applied["single_gate"]
                single_gate = np.array(
                    [[quantum_system.complex_decoder(d) for d in row] for row in single_gate])
                system_gate = gate_applied["system_gate"]
                system_gate = np.array(
                    [[quantum_system.complex_decoder(d) for d in row] for row in system_gate])
                quantum_system.apply_full_gate(
                    idx, gate_name, qubits_affected, single_gate, system_gate)

        return quantum_system

    # Initialize the state as |0...0>
    def __init__(self, n_qubits):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer")
        self.index = 0
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0  # Initialize to |000...0>
        assert self.is_valid_state()
        self.gates_applied = []

    # The function receives an array of desired value for each qubit, e.g. `[0,1,0,0]` for a 4-qubit system and sets the state accordingly.
    def initialize_state(self, qubit_values):
        assert len(qubit_values) == self.n_qubits
        # Creates a string from the array, e.g. [0,1,1] -> "011" then converts from binary to int to get the position
        self.index = int(''.join(map(str, qubit_values)), 2)
        self.state = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.state[self.index] = 1.0
        assert self.is_valid_state()
        self.initial_state = qubit_values

    def print_initial_qubits(self):
        print(
            f"Initial qubits (1st qubit starts from the left): {self.initial_state}")

    def quantum_noise(self):
        p = float(input("Probability for noise:"))
        target = random.randrange(self.n_qubits)
        if random.random() < p:
            self.apply_X_gate(target, False)
        if random.random() < p:
            self.apply_Z_gate(target, False)

    def apply_full_gate(self, idx, gate_name, qubits_affected, single_gate, system_gate, noise=False):
        self.state = np.dot(system_gate, self.state)

        if noise == True:
            self.quantum_noise()

        assert self.is_valid_state()
        self.gates_applied.append(
            (idx, gate_name, qubits_affected, single_gate, system_gate))

    # Apply general gate to the state
    def apply_gate(self, gate, n_gate=-1, starting_qubit=0, noise=False):
        assert 0 <= starting_qubit <= self.n_qubits - n_gate

        if n_gate == -1:
            n_gate = int(np.log2([len(gate)])[0])
            # self.n_qubits

        I = np.eye(2)
        gate_matrix = 1
        qubit = 0

        # Construct gate matrix for n_gate qubits
        while (qubit < self.n_qubits):
            if qubit == starting_qubit:
                gate_matrix = np.kron(gate_matrix, gate)
                qubit = qubit + n_gate
            else:
                gate_matrix = np.kron(gate_matrix, I)
                qubit += 1

        # Update the state by applying the gate matrix
        self.state = np.dot(gate_matrix, self.state)

        if noise == True:
            self.quantum_noise()

        assert self.is_valid_state()

        qubits_affected = [starting_qubit + i for i in range(n_gate)]
        gate_name = [key for key, value in gates_map.items(
        ) if gate.shape == value[0].shape and np.all(value[0] == gate)][0]
        # self.gates_applied.append((len(self.gates_applied), gate_name, gate, qubits_affected, gate_matrix))
        self.gates_applied.append(
            (len(self.gates_applied)+1, gate_name, qubits_affected, gate, gate_matrix))

    def apply_H_gate(self, target_qubit, noise=False):
        gate = gates_map["H"][0]
        n_gate = gates_map["H"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_X_gate(self, target_qubit, noise=False):
        gate = gates_map["X"][0]
        n_gate = gates_map["X"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_Y_gate(self, target_qubit, noise=False):
        gate = gates_map["Y"][0]
        n_gate = gates_map["Y"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_Z_gate(self, target_qubit, noise=False):
        gate = gates_map["Z"][0]
        n_gate = gates_map["Z"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_T_gate(self, target_qubit, noise=False):
        gate = gates_map["T"][0]
        n_gate = gates_map["T"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_S_gate(self, target_qubit, noise=False):
        gate = gates_map["S"][0]
        n_gate = gates_map["S"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CNOT_gate(self, target_qubit, noise=False):
        gate = gates_map["CNOT"][0]
        n_gate = gates_map["CNOT"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CH_gate(self, target_qubit, noise=False):
        gate = gates_map["CH"][0]
        n_gate = gates_map["CH"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CY_gate(self, target_qubit, noise=False):
        gate = gates_map["CY"][0]
        n_gate = gates_map["CY"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CZ_gate(self, target_qubit, noise=False):
        gate = gates_map["CZ"][0]
        n_gate = gates_map["CZ"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CT_gate(self, target_qubit, noise=False):
        gate = gates_map["CT"][0]
        n_gate = gates_map["CT"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CS_gate(self, target_qubit, noise=False):
        gate = gates_map["CS"][0]
        n_gate = gates_map["CS"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_SWAP_gate(self, target_qubit, noise=False):
        gate = gates_map["SWAP"][0]
        n_gate = gates_map["SWAP"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_CNOT10_gate(self, target_qubit, noise=False):
        gate = gates_map["CNOT10"][0]
        n_gate = gates_map["CNOT10"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_TOFFOLI_gate(self, target_qubit, noise=False):
        gate = gates_map["TOFFOLI"][0]
        n_gate = gates_map["TOFFOLI"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def plot_state_probabilities(self):
        data = {}
        for i in range(self.n_qubits):
            key = str(i)
            value = self.probabilities[i]
            data[key] = value
        courses = list(data.keys())
        values = list(data.values())

        fig = plt.figure(figsize=(8, 4))

        # creating the bar plot
        plt.bar(courses, values, color='maroon',
                width=0.4)

        plt.xlabel("qubit number")
        plt.ylabel("probabilities")
        plt.title("probabilities of measuring 0 in computational basis")
        plt.show()

    def produce_measurement(self):
        projectors = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

        def project(i, j, self):
            shape_tuple = ()
            for q in range(self.n_qubits):
                shape_tuple = shape_tuple + (2,)

            modified_state = np.reshape(self.state, shape_tuple)
            projected = np.tensordot(projectors[j], modified_state, (1, i))
            return np.moveaxis(projected, 0, i)

        measurements = np.zeros(self.n_qubits, dtype=int)
        self.probabilities = np.zeros(self.n_qubits, dtype=float)
        for i in range(self.n_qubits):
            projected = project(i, 0, self)
            # print(projected)
            norm_projected = norm(projected.flatten())
            # measurements = np.zeros(self.n_qubits, dtype=int)
            print("No qubit {}. Probability to be 0: {}".format(
                i, norm_projected**2))
            self.probabilities[i] = norm_projected**2
            if np.random.random() < norm_projected**2:  # Sample according to probability distribution
                # print(projected/norm_projected)
                measurement_result = 0
            else:
                projected = project(i, 1, self)
                # print(projected/norm(projected))
                measurement_result = 1
            measurements[i] = measurement_result

        return measurements.tolist()

    def produce_measurement_2(self):
        # Calculate the probabilities of each state
        probabilities = np.abs(self.state) ** 2

        # Randomly select a state based on these probabilities
        measured_state_index = np.random.choice(
            len(self.state), p=probabilities)

        # Convert the state index to binary representation and then to a list of qubits
        # Format the binary string to have the same length as the number of qubits
        binary_state = format(measured_state_index,
                              '0' + str(self.n_qubits) + 'b')

        # Convert the binary string to a list of integers (0s and 1s)
        # measured_qubits = [int(bit) for bit in binary_state]
        return binary_state

    def plot_state_probabilities_2(self):
        # Compute the probabilities for each basis state
        probabilities = np.abs(self.state)**2

        # Generate labels for the x-axis representing the binary states
        binary_states = [format(i, '0' + str(self.n_qubits) + 'b')
                         for i in range(2**self.n_qubits)]

        # Create a bar chart
        plt.bar(binary_states, probabilities)

        # Set labels and title
        plt.xlabel('Binary States')
        plt.ylabel('Probabilities')
        plt.title('Probabilities for a {}-qubit System'.format(self.n_qubits))

        # Show the bar chart
        plt.show()

    def produce_specific_measurement(self, qubit):
        projectors = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

        def project(i, j, self):
            shape_tuple = ()
            for q in range(self.n_qubits):
                shape_tuple = shape_tuple + (2,)

            modified_state = np.reshape(self.state, shape_tuple)
            projected = np.tensordot(projectors[j], modified_state, (1, i))
            return np.moveaxis(projected, 0, i)

        projected = project(qubit, 0, self)
        norm_projected = norm(projected.flatten())
        print("No qubit {}. Probability to be 0: {}".format(
            qubit, norm_projected**2))

        if np.random.random() < norm_projected**2:
            measurement_result = 0
        else:
            projected = project(qubit, 1, self)
            measurement_result = 1

        return measurement_result

    def apply_control(self, control_qubit, target_qubit, gate_matrix):

        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])

        M1 = np.eye(1)
        M2 = np.eye(1)

        if (target_qubit > control_qubit):
            M1 = np.kron(P0, np.eye(
                2 ** (target_qubit - control_qubit - 1 + int(np.log2([len(gate_matrix)])[0]))))

            # M1 = np.kron(np.eye(2**(control_qubit)), M1)
            # M1 = np.kron(M1, np.eye(2** (self.n_qubits - target_qubit - 1)))

            M2 = np.kron(
                np.kron(P1, np.eye(2 ** (target_qubit - control_qubit - 1))), gate_matrix)
            # - int(np.log2([len(gate_matrix)])[0])
            # M2 = np.kron(np.eye(2**(control_qubit)), M2)
            # M2 = np.kron(M2, np.eye(2** (self.n_qubits - target_qubit - 1)))

        else:
            M1 = np.kron(np.eye(2 ** (control_qubit - target_qubit -
                         1 + int(np.log2([len(gate_matrix)])[0]))), P0)
            # M1 = np.kron(np.eye(2**(target_qubit)), M1)
            # M1 = np.kron(M1, np.eye(2** (self.n_qubits - control_qubit - 1)))

            M2 = np.kron(gate_matrix,
                         np.kron(np.eye(2 ** (control_qubit - target_qubit - 1)), P1))
            # M2 = np.kron(np.eye(2**(target_qubit)), M2)
            # M2 = np.kron(M2, np.eye(2** (self.n_qubits - control_qubit - 1)))

        controlled_gate = M1 + M2

        return controlled_gate

    def control_gate(self, control_qubits, target_qubit, gate_matrix, name=-1):

        control_qubits = np.sort(control_qubits)
        index_target = np.searchsorted(control_qubits, target_qubit)
        index_target2 = np.searchsorted(
            control_qubits, target_qubit + int(np.log2([len(gate_matrix)])[0]) - 1)
        assert index_target == index_target2, "Qubit overlap (target qubit same as control qubit)"

        qubits_before_target = control_qubits[:index_target]
        qubits_after_target = control_qubits[index_target:]

        controlled_gate = gate_matrix
        num_qubits_a = len(qubits_after_target)
        for i in range(0, num_qubits_a):
            controlled_gate = self.apply_control(
                qubits_after_target[i], target_qubit, controlled_gate)

        num_qubits_b = len(qubits_before_target)
        for i in range(0, num_qubits_b):
            control_qb = qubits_before_target[num_qubits_b - i - 1]
            controlled_gate = self.apply_control(
                control_qb, target_qubit, controlled_gate)
            target_qubit = control_qb

        print(controlled_gate)
        if name != -1:
            gates_map[name] = (controlled_gate, int(
                np.log2(len(controlled_gate))))

        qubits_affected = np.append(control_qubits, target_qubit)

        return controlled_gate

    def swap_n_gate(self, control_qubit, target_qubit, name=-1):

        M1 = np.eye(1)
        M2 = np.eye(1)

        if (target_qubit < control_qubit):
            aux = target_qubit
            target_qubit = control_qubit
            control_qubit = aux

        # M1 eye
        M1 = np.eye(2 ** (self.n_qubits))

        # M2 X gates_map["X"][0]
        M2_1 = np.kron(np.eye(2 ** control_qubit), gates_map["X"][0])
        M2_2 = np.kron(
            np.eye(2 ** (target_qubit - control_qubit - 1)), gates_map["X"][0])
        M2_3 = np.kron(M2_1, M2_2)
        M2 = np.kron(M2_3, np.eye(2**(self.n_qubits - target_qubit - 1)))

        # M3 Y gates_map["Y"][0]
        M3_1 = np.kron(np.eye(2 ** control_qubit), gates_map["Y"][0])
        M3_2 = np.kron(
            np.eye(2 ** (target_qubit - control_qubit - 1)), gates_map["Y"][0])
        M3_3 = np.kron(M3_1, M3_2)
        M3 = np.kron(M3_3, np.eye(2**(self.n_qubits - target_qubit - 1)))

        # M4 Z gates_map["Z"][0]
        M4_1 = np.kron(np.eye(2 ** control_qubit), gates_map["Z"][0])
        M4_2 = np.kron(
            np.eye(2 ** (target_qubit - control_qubit - 1)), gates_map["Z"][0])
        M4_3 = np.kron(M4_1, M4_2)
        M4 = np.kron(M4_3, np.eye(2**(self.n_qubits - target_qubit - 1)))

        swap_gate = M1 + M2 + M3 + M4
        swap_gate = swap_gate * 0.5
        print(swap_gate)
        if name != -1:
            gates_map[name] = (swap_gate, int(np.log2(len(swap_gate))))

        return swap_gate

    def import_to_qiskit(self):
        pass
        """
        circuit = json.load(json_file)
        initial_state = circuit["initial_state"]
        quantum_system = NQubitSystem(n_qubits = len(initial_state))
        quantum_system.initialize_state(initial_state)
        gates_applied = circuit["gates_applied"]
        for gate_applied in gates_applied:
            idx = gate_applied["idx"]
            gate_name = gate_applied["gate_name"]
            qubits_affected = gate_applied["qubits_affected"]
            single_gate = gate_applied["single_gate"]
            single_gate = np.array([[quantum_system.complex_decoder(d) for d in row] for row in single_gate])
            system_gate = gate_applied["system_gate"]
            system_gate = np.array([[quantum_system.complex_decoder(d) for d in row] for row in system_gate])
            quantum_system.apply_full_gate(idx, gate_name, qubits_affected, single_gate, system_gate)
        """

    def calculate_density_matrix(self):
        density_matrix = np.tensordot(self.state, self.state.conj(), axes=0)
        return density_matrix

    def plot_density_matrix(self):
        density_matrix = np.tensordot(self.state, self.state.conj(), axes=0)
        density_matrix = abs(density_matrix)
        plt.imshow(density_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('Density matrix')
        plt.show()
