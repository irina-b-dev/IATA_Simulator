from NQubitSystem import NQubitSystem

quantum_system = NQubitSystem(n_qubits=3)
quantum_system.initialize_state([0, 1, 1])
quantum_system.apply_H_gate(0)
quantum_system.print_state()
quantum_system.plot_density_matrix()
print(quantum_system.calculate_density_matrix())
