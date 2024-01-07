# Gates class
import numpy as np
import matplotlib.pyplot as plt
from constants import gates

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
    def create_custom_gate(dim = -1, gates_list = -1):
        if dim == -1:
            dim = int(input("Enter no qubits affected:"))
        if gates_list == -1: 
            gates_list = input("Enter list of gates separated by space:")
        gates_list = gates_list.split()
        init_gate = np.eye(2**dim)
        for i in range(len(gates_list)):
            print(gates[gates_list[i]])
            gate = gates[gates_list[i]][0]
            init_gate = np.dot(init_gate, gate)
        return dim, init_gate 

    @staticmethod
    def create_custom_user_gate():
        size = int(input("Enter the size of the square matrix: "))

        # Initialize an empty list to store the matrix
        matrix = []

        # Loop to input each row of the square matrix
        for i in range(size):
            row_str = input(f"Enter values for row {i + 1} separated by space: ")
            row_values = list(map(float, row_str.split()))
            
            # Ensure the number of columns matches the size
            if len(row_values) != size:
                print("Invalid input. Number of columns must match the size of the square matrix.")
                return None

            matrix.append(row_values)

        # Convert the list of lists to a NumPy array
        matrix_array = np.array(matrix)

        return matrix_array