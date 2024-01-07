import numpy as np
from constants import gates
from scipy.linalg import norm
import matplotlib.pyplot as plt
import random

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

    # Initialize the state as |0...0>
    def __init__(self, n_qubits):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer")
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype = complex)
        self.state[0] = 1.0  # Initialize to |000...0>
        assert self.is_valid_state()

    # The function receives an array of desired value for each qubit, e.g. `[0,1,0,0]` for a 4-qubit system and sets the state accordingly.
    def initialize_state(self, qubit_values):
        assert len(qubit_values) == self.n_qubits
        index = int(''.join(map(str, qubit_values)), 2) # Creates a string from the array, e.g. [0,1,1] -> "011" then converts from binary to int to get the position
        self.state = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.state[index] = 1.0
        assert self.is_valid_state()

    def quantum_noise(self):
            p = float(input("Probability for noise:"))
            target = random.randrange(self.n_qubits)
            if random.random() < p:
                self.apply_X_gate(target, False)
            if random.random() < p:
                self.apply_Z_gate(target, False)

    # Apply general gate to the state
    def apply_gate(self, gate, n_gate = -1, starting_qubit = 0, noise = False):
        assert 0 <= starting_qubit <= self.n_qubits - n_gate

        if n_gate == -1:
            n_gate = self.n_qubits

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

    def apply_H_gate(self, target_qubit, noise):
        gate = gates["H"][0]
        n_gate = gates["H"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def apply_X_gate(self, target_qubit, noise):
        gate = gates["X"][0]
        n_gate = gates["X"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_Y_gate(self, target_qubit, noise):
        gate = gates["Y"][0]
        n_gate = gates["Y"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_Z_gate(self, target_qubit, noise):
        gate = gates["Z"][0]
        n_gate = gates["Z"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_T_gate(self, target_qubit, noise):
        gate = gates["T"][0]
        n_gate = gates["T"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
    
    def apply_S_gate(self, target_qubit, noise):
        gate = gates["S"][0]
        n_gate = gates["S"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
    
    def apply_CNOT_gate(self, target_qubit, noise):
        gate = gates["CNOT"][0]
        n_gate = gates["CNOT"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
    
    def apply_CH_gate(self, target_qubit, noise):
        gate = gates["CH"][0]
        n_gate = gates["CH"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_CY_gate(self, target_qubit, noise):
        gate = gates["CY"][0]
        n_gate = gates["CY"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_CZ_gate(self, target_qubit, noise):
        gate = gates["CZ"][0]
        n_gate = gates["CZ"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_CT_gate(self, target_qubit, noise):
        gate = gates["CT"][0]
        n_gate = gates["CT"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
        
    def apply_CS_gate(self, target_qubit, noise):
        gate = gates["CS"][0]
        n_gate = gates["CS"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
    
    def apply_SWAP_gate(self, target_qubit, noise):
        gate = gates["SWAP"][0]
        n_gate = gates["SWAP"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
    
    def apply_CNOT10_gate(self, target_qubit, noise):
        gate = gates["CNOT10"][0]
        n_gate = gates["CNOT10"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)
    
    def apply_TOFFOLI_gate(self, target_qubit, noise):
        gate = gates["TOFFOLI"][0]
        n_gate = gates["TOFFOLI"][1]
        self.apply_gate(gate, n_gate, target_qubit, noise)

    def plot_state_probabilities(self):
        data = {}
        for i in range(self.n_qubits):
            key = str(i)
            value = self.probabilities[i]
            data[key] = value
        courses = list(data.keys())
        values = list(data.values())
      
        fig = plt.figure(figsize = (8, 4))
     
        # creating the bar plot
        plt.bar(courses, values, color ='maroon', 
            width = 0.4)
     
        plt.xlabel("qubit number")
        plt.ylabel("probabilities")
        plt.title("probabilities of measuring 0 in computational basis")
        plt.show()

    def produce_measurement(self):
        projectors=[np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ]
        
        def project(i,j,self):
            shape_tuple = ()
            for q in range(self.n_qubits):
                shape_tuple = shape_tuple + (2,)
            
            modified_state=np.reshape(self.state, shape_tuple)
            projected=np.tensordot(projectors[j],modified_state,(1,i))
            return np.moveaxis(projected,0,i)
        
        measurements = np.zeros(self.n_qubits, dtype=int)
        self.probabilities = np.zeros(self.n_qubits, dtype=float)
        for i in range(self.n_qubits):
            projected=project(i,0,self) 
            #print(projected)
            norm_projected=norm(projected.flatten())
            #measurements = np.zeros(self.n_qubits, dtype=int)
            print("No qubit {}. Probability to be 0: {}".format(i, norm_projected**2))
            self.probabilities[i] = norm_projected**2
            if np.random.random()<norm_projected**2: # Sample according to probability distribution
                #print(projected/norm_projected)
                measurement_result = 0
            else:
                projected=project(i,1,self)
                #print(projected/norm(projected))
                measurement_result = 1
            measurements[i] = measurement_result
        
        return measurements.tolist()

    def produce_measurement_2(self):
        # Calculate the probabilities of each state
        probabilities = np.abs(self.state) ** 2

        # Randomly select a state based on these probabilities
        measured_state_index = np.random.choice(len(self.state), p=probabilities)

        # Convert the state index to binary representation and then to a list of qubits
        # Format the binary string to have the same length as the number of qubits
        binary_state = format(measured_state_index, '0' + str(self.n_qubits) + 'b')

        # Convert the binary string to a list of integers (0s and 1s)
        #measured_qubits = [int(bit) for bit in binary_state]
        return binary_state

    def plot_state_probabilities_2(self):
        # Compute the probabilities for each basis state
        probabilities = np.abs(self.state)**2

        # Generate labels for the x-axis representing the binary states
        binary_states = [format(i, '0' + str(self.n_qubits) + 'b') for i in range(2**self.n_qubits)]

        # Create a bar chart
        plt.bar(binary_states, probabilities)

        # Set labels and title
        plt.xlabel('Binary States')
        plt.ylabel('Probabilities')
        plt.title('Probabilities for a {}-qubit System'.format(self.n_qubits))

        # Show the bar chart
        plt.show()

    def produce_specific_measurement(self, qubit):
        projectors=[np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ]
        
        def project(i,j,self):
            shape_tuple = ()
            for q in range(self.n_qubits):
                shape_tuple = shape_tuple + (2,)
            
            modified_state=np.reshape(self.state, shape_tuple)
            projected=np.tensordot(projectors[j],modified_state,(1,i))
            return np.moveaxis(projected,0,i)
        
        projected=project(qubit,0,self) 
        norm_projected=norm(projected.flatten())
        print("No qubit {}. Probability to be 0: {}".format(qubit, norm_projected**2))

        if np.random.random()<norm_projected**2: 
            measurement_result = 0
        else:
            projected=project(qubit,1,self)
            measurement_result = 1
        
        return measurement_result

    def control_gate(self, control_qubit, target_qubit, gate_matrix):
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])

        M1 = np.eye(1)
        M2 = np.eye(1)

        if(target_qubit>control_qubit):
            M1 = np.kron(P0, np.eye(2 ** (target_qubit - control_qubit)))
            M2 = np.kron(np.kron(P1, np.eye(2 ** (target_qubit - control_qubit) - len(gate_matrix))), gate_matrix)
            
        else:
            M1 = np.kron(np.eye(2 ** (control_qubit - target_qubit)), P0)
            M2 = np.kron(gate_matrix, np.kron(np.eye(2 ** (control_qubit - target_qubit -1) - len(gate_matrix)), P1)) 

        controlled_gate = M1 + M2
        #print(controlled_gate)
        return controlled_gate