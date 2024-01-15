from NQubitSystem import NQubitSystem
import numpy as np
from constants import gates_map
from Gate import Gate

def samplequantumgenerator():
    qubit = NQubitSystem(1)
    qubit.apply_H_gate(0, False)
    bit = qubit.produce_measurement()
    return bit[0]

# q = samplequantumgenerator()
# print(q)

def sample2quantumgenerator():
    qubit = NQubitSystem(2)
    qubit.apply_H_gate(0, False)
    qubit.apply_H_gate(1, False)
    bit = qubit.produce_measurement()
    str_bit = ''.join(str(e) for e in bit)
    return str_bit

def samplenrquantumgenerator(nr_q):
    qubit = NQubitSystem(nr_q)
    for i in range(nr_q):
        qubit.apply_H_gate(i, False)
    bit = qubit.produce_measurement()
    str_bit = ''.join(str(e) for e in bit)
    return str_bit

def samplerandominrange(max):
    output = max + 1
    while output > max:
        bit_list = []
        for i in range(0, max.bit_length()):
            bit = samplequantumgenerator()
            bit_list.append(bit)
        output = int("".join(str(x) for x in bit_list), 2)
    return output
