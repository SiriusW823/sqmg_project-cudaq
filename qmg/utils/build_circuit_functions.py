from qiskit import QuantumCircuit
import qiskit.circuit.library as qulib
import numpy as np
import random
from typing import List, Union

class CircuitBuilder():
    """ This normal circuit does not support the function of conditinal weight or molecular structure generation. """
    def __init__(self, num_heavy_atom:int, temperature:float=0.2, remove_bond_disconnection:bool=True, chemistry_constraint:bool=True):
        self.num_heavy_atom = num_heavy_atom
        self.temperature = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.num_qubits = num_heavy_atom*(num_heavy_atom+1)
        self.num_ancilla_qubits = num_heavy_atom - 1
        self.length_all_weight_vector = int(8 + (self.num_heavy_atom - 2)*(self.num_heavy_atom + 3) * 3 / 2)

    def initialize_quantum_circuit(self):
        self.qc = QuantumCircuit(self.num_qubits+self.num_ancilla_qubits, self.num_qubits)

    def softmax_temperature(self, weight_vector):
        weight_vector /= self.temperature
        exps = np.exp(weight_vector)
        return exps / np.sum(exps)

    def controlled_ry(self, control:int, target:int, digit:float):
        self.qc.cry(np.pi*digit, control, target)

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
        # store the |00> state in ancilla qubit (fourth qubit).
        self.qc.x(2)
        self.qc.x(3)
        self.qc.x(4)
        self.qc.ccx(2,3,4) 
        self.qc.x(2)
        self.qc.x(3)
        # bond type (2-1) circuit. If ancilla is |0>, then bond 2-1 circuit must output |00>.
        self.qc.cx(4, 5)
        self.qc.cx(5, 6)
        self.controlled_ry(4, 5, weight_vector[6])
        self.qc.cx(5, 6)
        self.controlled_ry(5, 6, weight_vector[7])

    def build_bond_type_circuit(self, heavy_atom_number: int, fixed_weight_vector: Union[List[float], np.ndarray],
                                flexible_weight_vector: Union[List[float], np.ndarray]):
        num_target_qubit = (heavy_atom_number - 1) * 2
        assert len(fixed_weight_vector)*2 == num_target_qubit == len(flexible_weight_vector)
        # ancilla qubit indicating the existence of heavy_atom_number
        ancilla_qubit_index = 2*heavy_atom_number + (heavy_atom_number-1)**2 - 1
        for i in range(2*(heavy_atom_number-1)):
            self.qc.cx(ancilla_qubit_index+i, ancilla_qubit_index+i+1)
        for i in range((heavy_atom_number-1)):
            self.controlled_ry(ancilla_qubit_index+num_target_qubit-2*i-1, ancilla_qubit_index+num_target_qubit-2*i, 1-fixed_weight_vector[-1-i])
            self.qc.cx(ancilla_qubit_index+num_target_qubit-2*i-2, ancilla_qubit_index+num_target_qubit-2*i-1)
            self.controlled_ry(ancilla_qubit_index+num_target_qubit-2*i, ancilla_qubit_index+num_target_qubit-2*i-1, flexible_weight_vector[-2-2*i])
            self.controlled_ry(ancilla_qubit_index+num_target_qubit-2*i-1, ancilla_qubit_index+num_target_qubit-2*i, flexible_weight_vector[-1-2*i])
    
    def build_atom_type_circuit(self, heavy_atom_number: int, weight_vector: Union[List[float], np.ndarray]):
        assert len(weight_vector) == 3
        ancilla_qubit_index = 2*(heavy_atom_number-1) + (heavy_atom_number-2)**2 - 1
        qubit_1_index = ancilla_qubit_index + 2*(heavy_atom_number - 2) + 1
        qubit_2_index = qubit_1_index + 1
        self.qc.cx(ancilla_qubit_index, qubit_1_index)
        self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[1])
        self.controlled_ry(ancilla_qubit_index, qubit_1_index, weight_vector[0])
        self.qc.cx(qubit_2_index, qubit_1_index)
        self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[2])
        
        self.qc.x(qubit_1_index)
        self.qc.x(qubit_2_index)
        self.qc.x(qubit_2_index+1)
        self.qc.ccx(qubit_1_index,qubit_2_index,qubit_2_index+1) # store the |00> state in ancilla qubit.
        self.qc.x(qubit_1_index)
        self.qc.x(qubit_2_index)

    def build_removing_bond_disconnection_circuit(self, heavy_atom_number: int):
        ancilla_qubit_index = 2*(heavy_atom_number) + (heavy_atom_number-1)**2 - 1
        mcx_gate = qulib.XGate().control(2*(heavy_atom_number-1))
        control_qubits_index_list = list(range(ancilla_qubit_index+1, ancilla_qubit_index+1+2*(heavy_atom_number-1)))
        for id_ in [ancilla_qubit_index]+control_qubits_index_list:
            self.qc.x(id_)
        self.qc.append(mcx_gate, control_qubits_index_list+[ancilla_qubit_index])
        for id_ in control_qubits_index_list:
            self.qc.x(id_)
        self.qc.cx(ancilla_qubit_index, control_qubits_index_list[-1])
        
    def measure(self):
        effective_qubit_index = list(range(self.num_qubits + self.num_ancilla_qubits))
        for j in range(2, self.num_heavy_atom+1):
            ancilla_qubit_number = 2*j + (j-1)**2 - 1
            effective_qubit_index.remove(ancilla_qubit_number)
        self.effective_qubit_index = effective_qubit_index
        self.qc.measure(self.effective_qubit_index, list(range(self.num_qubits)))
    
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
        if self.num_heavy_atom - 2 > 0:
            used_part = 8
            for heavy_idx in range(3, self.num_heavy_atom+1):
                num_fixed = heavy_idx-1
                num_flexible = 2*num_fixed
                atom_type_weight_vector = self.all_weight_vector[used_part: used_part+3]
                bond_type_fixed_part = self.all_weight_vector[used_part+3: used_part+3+num_fixed]
                bond_type_flexible_part = self.all_weight_vector[used_part+3+num_fixed: used_part+3+num_fixed+num_flexible]
                used_part += 3+num_fixed+num_flexible
                self.build_atom_type_circuit(heavy_idx, atom_type_weight_vector)
                if (heavy_idx >= 4) and self.remove_bond_disconnection:
                    self.build_removing_bond_disconnection_circuit(heavy_idx-1)
                self.build_bond_type_circuit(heavy_idx, bond_type_fixed_part, bond_type_flexible_part)
            else:
                if self.remove_bond_disconnection:
                    self.build_removing_bond_disconnection_circuit(heavy_idx)
        self.measure()
        return self.qc

if __name__=="__main__":
    qc_generator = CircuitBuilder(num_heavy_atom=3)
    qc = qc_generator.generate_quantum_circuit()
    print(qc.draw(output="text"))

