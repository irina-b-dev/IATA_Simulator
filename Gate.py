# Gates class
import numpy as np
import matplotlib.pyplot as plt
from constants import gates_map

class Gate:
    def __init__(self, dim):
        self.dim = dim
        self.gate = np.zeros([dim, dim], dtype=float)
    def initialize_gate(self, gate_values):
        assert np.shape(gate_values) == (self.dim, self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                self.gate[i][j] = gate_values[i][j]
    def compute_transpose(self):
        L, W = np.shape(self.gate)
        matrix_new = np.zeros([W, L], dtype=complex)
        for i in range(L):
            for j in range(W):
                matrix_new[j][i] = self.gate[i][j]
        return matrix_new
    def compute_adjoint(self):
        return np.transpose(np.conj(self.gate))
    def add_gates(gate_list):
        return np.sum(gate_list, axis=0)
    def multiply_gates(gates):
        L, W = gates[0].shape
        first = np.eye(L)
        for i in range(len(gates)):
            first = np.dot(first, gates[i])
        return first
    def multiply_constant(self, constant):
        return constant*self.gate
    def compute_rank(self):
        return np.linalg.matrix_rank(self.gate)
    def is_valid_gate(self):
        bec = 0
        if np.allclose(self.multiply_gates([self.gate, self.compute_adjoint(self.gate)]), np.eye(len(self.gate))):
            bec = 1
        return bec
    def get_eigen(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.gate)
        return eigenvalues, eigenvectors
    def plot_gate_representation(self):
        eigenvalues, eigenvectors = self.get_eigen(self.gate)
        for i in range(len(eigenvectors)):
            plt.quiver(0, 0, eigenvectors[i][0], eigenvectors[i][1], color='r', units='xy', scale=1)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Eigenvectors plot')
        plt.show()

    @staticmethod    
    def export_gate(file_name, gate):
        with open(file_name, 'wb') as f:
            np.save(f,gate)

    @staticmethod
    def import_gate(file_name):
        with open(file_name, 'rb') as f:
            gate = np.load(f)
            return gate

    @staticmethod
    def create_custom_gate(dim=None, gates_list=None, name=None):
        if dim is None:
            dim = int(input("Enter no qubits affected:"))

        if gates_list is None:
            gates_input = input("Enter list of gates separated by space:")
            gates_list = gates_input.split()

        init_gate = np.eye(2 ** dim)
        for gate_name in gates_list:
            if gate_name not in gates_map:
                print(f"Invalid gate name: {gate_name}")
                return None, None
            gate = gates_map[gate_name][0]
            init_gate = np.dot(init_gate, gate)

        if name != None:
            gates_map[name] = (init_gate, int(np.log2(len(init_gate))))

        return dim, init_gate

    @staticmethod
    def create_custom_user_gate(size=None, matrix_values=None, name=None):
        if size is None:
            size = int(input("Enter the size of the square matrix: "))

        matrix = []

        if matrix_values is None:
            # Get user input if matrix_values is not provided
            for i in range(size):
                row_str = input(f"Enter values for row {i + 1} separated by space: ")
                row_values = list(map(float, row_str.split()))
                
                if len(row_values) != size:
                    print("Invalid input. Number of columns must match the size of the square matrix.")
                    return None

                matrix.append(row_values)
        else:
            # Use provided matrix_values
            if len(matrix_values) != size or any(len(row) != size for row in matrix_values):
                print("Invalid input. Size of the matrix or rows does not match the specified size.")
                return None

            matrix = matrix_values

        matrix_array = np.array(matrix)

        if name != None:
            gates_map[name] = (matrix_array, int(np.log2(len(matrix_array))))

        return matrix_array