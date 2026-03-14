from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
import qiskit.circuit.library as qulib
from qiskit.qasm3 import dump, dumps, Exporter
import numpy as np
import random
from typing import List, Union

class DynamicCircuitBuilder():
    """ Dynamic circuit version of the molecule generation ansatz. It utilizes less qubits. """
    def __init__(self, num_heavy_atom:int, temperature:float=0.2, remove_bond_disconnection:bool=True, chemistry_constraint:bool=True):
        self.num_heavy_atom = num_heavy_atom
        self.temperature = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.num_qubits = 4 + (num_heavy_atom-1) * 2
        self.num_clbits = num_heavy_atom * (num_heavy_atom + 1)
        self.qubits = QuantumRegister(self.num_qubits)
        self.clbits = ClassicalRegister(self.num_clbits, name="c")
        self.length_all_weight_vector = int(8 + (self.num_heavy_atom - 2)*(self.num_heavy_atom + 3) * 3 / 2)

    def initialize_quantum_circuit(self):
        self.qc = QuantumCircuit(self.qubits, self.clbits)

    def softmax_temperature(self, weight_vector):
        weight_vector /= self.temperature
        exps = np.exp(weight_vector)
        return exps / np.sum(exps)

    def controlled_ry(self, control:int, target:int, digit:float):
        self.qc.cry(np.pi*digit, control, target)

    def get_classical_register_by_name(self, name: str):
        for cr in self.qc.cregs:
            if cr.name == name:
                return cr
        else:
            raise ValueError(f"The name {name} is not found in classical registers.")

    def build_two_atoms(self, weight_vector: Union[List[float], np.ndarray]):
        assert len(weight_vector) == 8 # length of weight vector should be 8
        self.qc.ry(np.pi * weight_vector[0], 0)
        self.qc.x(1)
        self.qc.ry(np.pi * weight_vector[2], 2)
        self.qc.ry(np.pi * weight_vector[4], 3)
        self.qc.cx(0, 1)
        self.controlled_ry(1, 2, weight_vector[3])
        self.qc.cx(2, 3)
        self.controlled_ry(0, 1, weight_vector[1])
        self.qc.cx(1, 2)
        self.controlled_ry(2, 3, weight_vector[5])

        # measure atom 1 state:
        self.qc.measure(self.qubits[0:2], self.clbits[0:2])
        # measure atom 2 state and save:
        self.qc.measure(self.qubits[2:4], self.clbits[2:4])

        # Add a new atom existence ClassicalRegister:
        atom_existence_CR = ClassicalRegister(bits=self.clbits[2:4], name="atom_2_existence")
        self.qc.add_register(atom_existence_CR)
        with self.qc.if_test((atom_existence_CR, 0)) as else_:
            pass
        with else_:
            self.qc.ry(np.pi * weight_vector[6], 4)
            self.qc.x(5)
            self.qc.cx(4,5)
            self.controlled_ry(4, 5, weight_vector[7])

        self.qc.measure(self.qubits[4:6], self.clbits[4:6])

    def reset_previous_atom_bond_circuit(self, heavy_idx):
        reset_qubits_index = list(range(2, 2*heavy_idx))
        start_clbit = (heavy_idx - 2)**2 + (heavy_idx - 2)
        reset_clbits_index = list(range(start_clbit, start_clbit+(heavy_idx - 1)*2))
        for qubit_index, clbit_index in zip(reset_qubits_index, reset_clbits_index):
            with self.qc.if_test((self.clbits[clbit_index], 1)):
                self.qc.x(qubit_index)                

    def build_atom_type_circuit(self, heavy_atom_number: int, weight_vector: Union[List[float], np.ndarray]):
        assert len(weight_vector) == 3
        qubit_1_index = 2
        qubit_2_index = 3
        clbit_1_index = (heavy_atom_number - 1)**2 + (heavy_atom_number - 1)
        clbit_2_index = clbit_1_index + 1

        register_name = f"atom_{heavy_atom_number-1}_existence"
        atom_existence_CR = self.get_classical_register_by_name(register_name)
        with self.qc.if_test((atom_existence_CR, 0)) as else_:
            pass
        with else_:
            self.qc.ry(np.pi * weight_vector[0], qubit_1_index)
            self.qc.ry(np.pi * weight_vector[1], qubit_2_index)
            # self.qc.cx(qubit_1_index, qubit_2_index)
            self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[2])
        self.qc.measure(self.qubits[[qubit_1_index,qubit_2_index]], self.clbits[[clbit_1_index,clbit_2_index]])
        # create new atom existence register
        atom_existence_CR = ClassicalRegister(bits=self.clbits[[clbit_1_index,clbit_2_index]], name=f"atom_{heavy_atom_number}_existence")
        self.qc.add_register(atom_existence_CR)

    def build_bond_type_circuit(self, heavy_atom_number: int, fixed_weight_vector: Union[List[float], np.ndarray],
                                flexible_weight_vector: Union[List[float], np.ndarray]):
        assert len(fixed_weight_vector) == heavy_atom_number-1
        assert len(flexible_weight_vector) == 2*(heavy_atom_number-1)
        qubit_start_index = 4
        qubit_end_index = qubit_start_index + 2*(heavy_atom_number - 1)
        clbit_start_index = (heavy_atom_number)**2 - heavy_atom_number + 2
        clbit_end_index = clbit_start_index + 2*(heavy_atom_number - 1)

        register_name = f"atom_{heavy_atom_number}_existence"
        atom_existence_CR = self.get_classical_register_by_name(register_name)
        with self.qc.if_test((atom_existence_CR, 0)) as else_:
            pass
        with else_:
            for i in range(heavy_atom_number-1):
                self.qc.ry(np.pi * fixed_weight_vector[i], qubit_start_index+2*i+1)
                self.controlled_ry(qubit_start_index+2*i+1, qubit_start_index+2*i, flexible_weight_vector[2*i]) # < 0.5
                self.controlled_ry(qubit_start_index+2*i, qubit_start_index+2*i+1, flexible_weight_vector[2*i+1]) # > 0.5
            self.qc.measure(self.qubits[qubit_start_index:qubit_end_index], self.clbits[clbit_start_index:clbit_end_index])
            if self.remove_bond_disconnection:
                bond_disconnection_CR = ClassicalRegister(bits=self.clbits[clbit_start_index:clbit_end_index], name=f"bond_{heavy_atom_number}_connection")
                self.qc.add_register(bond_disconnection_CR)
                with self.qc.if_test((bond_disconnection_CR, 0)):
                    self.qc.x(qubit_end_index-1)
                    self.qc.measure(self.qubits[qubit_end_index-1], self.clbits[clbit_end_index-1])
    
    def generate_quantum_circuit(self, all_weight_vector:Union[List[float], np.ndarray]=None, random_seed=0):
        random.seed(random_seed)
        self.initialize_quantum_circuit()
        # (1) generate weight vector
        if (type(all_weight_vector) == np.ndarray) or (type(all_weight_vector) == list):
            assert len(all_weight_vector) == self.length_all_weight_vector
            self.all_weight_vector = all_weight_vector
        else:
            self.all_weight_vector = np.array([random.random() for _ in range(self.length_all_weight_vector)])
            if self.chemistry_constraint and (self.num_heavy_atom>=3):
                used_part = 8
                for heavy_idx in range(3, self.num_heavy_atom+1):
                    used_part += 3 # atom type weight vector
                    num_fixed = heavy_idx-1
                    num_flexible = 2*num_fixed
                    bond_type_fixed_part = self.all_weight_vector[used_part: used_part+num_fixed]
                    self.all_weight_vector[used_part: used_part+num_fixed] = self.softmax_temperature(bond_type_fixed_part)
                    bond_type_flexible_part = self.all_weight_vector[used_part+num_fixed: used_part+num_fixed+num_flexible]
                    bond_type_flexible_part *= 0.5
                    bond_type_flexible_part += np.array([0, 0.5]*(heavy_idx-1))
                    self.all_weight_vector[used_part+num_fixed: used_part+num_fixed+num_flexible] = bond_type_flexible_part
                    used_part += num_fixed + num_flexible
        # (2) start to construct the quantum circuit
        self.build_two_atoms(self.all_weight_vector[0:8])
        used_part = 8
        for heavy_idx in range(3, self.num_heavy_atom+1):
            num_fixed = heavy_idx-1
            num_flexible = 2*num_fixed
            atom_type_weight_vector = self.all_weight_vector[used_part: used_part+3]
            bond_type_fixed_part = self.all_weight_vector[used_part+3: used_part+3+num_fixed]
            bond_type_flexible_part = self.all_weight_vector[used_part+3+num_fixed: used_part+3+num_fixed+num_flexible]
            used_part += 3+num_fixed+num_flexible

            self.reset_previous_atom_bond_circuit(heavy_idx)
            self.build_atom_type_circuit(heavy_idx, atom_type_weight_vector)
            self.build_bond_type_circuit(heavy_idx, bond_type_fixed_part, bond_type_flexible_part)

        return self.qc

if __name__=="__main__":
    qc_generator = DynamicCircuitBuilder(num_heavy_atom=5)
    dqc = qc_generator.generate_quantum_circuit()
    # print(dqc.draw(output="text"))
    # f = open("my_file.txt", 'w')
    # dump(dqc, f)
    # f.close()
    exp = Exporter(allow_aliasing=True)
    print(exp.dumps(dqc))

