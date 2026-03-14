from rdkit import Chem
import numpy as np
import random
from typing import List

class ConditionalWeightsGenerator():
    """ Generate the corresponding weights in dynamic quantum circuit, based on the provided Molecule SMARTS representation. """
    def __init__(self, num_heavy_atom:int, smarts=None, disable_connectivity_position:List[int]=[]):
        self.num_heavy_atom = num_heavy_atom
        self.length_all_weight_vector = int(8 + (num_heavy_atom - 2) * (num_heavy_atom + 3) * 3 / 2)
        self.parameters_value = np.zeros(self.length_all_weight_vector)
        self.parameters_indicator = np.zeros(self.length_all_weight_vector)
        self.atom_type_to_idx = {"C": 1, "O": 2, "N": 3}  # only supports C, O, N atoms now.
        self.bond_type_to_idx = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            None: 0
        }
        self.qubits_per_type_atom = int(np.ceil(np.log2(len(self.atom_type_to_idx) + 1))) # How many qubits required for describing the quantum state of atom type
        self.qubits_per_type_bond = int(np.ceil(np.log2(len(self.bond_type_to_idx)))) # How many qubits required for describing the quantum state of bond type
        self.smarts = smarts
        self.disable_connectivity_position = disable_connectivity_position
        self.mapnum_atom_dict = {}
        self.mapnum_bond_dict = {}
        if smarts:
            self.mol = Chem.MolFromSmiles(smarts)
            self.num_fixed_atoms = self.mol.GetNumAtoms()
            Chem.Kekulize(self.mol, clearAromaticFlags=True)
            self._initialize_maps()
            self._generate_constrained_parameters()
        else:
            self.num_fixed_atoms = 0
            self.used_part = 8

    def _initialize_maps(self):
        for atom in self.mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if not map_num:
                raise ValueError(f"The atom mapping number should be given in the SMARTS: {atom.GetSmarts()}.")
            self.mapnum_atom_dict.update({map_num: atom})
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            a1, a2 = min(a1, a2), max(a1, a2)
            self.mapnum_bond_dict.update({(a1, a2): bond.GetBondType()})

        if not sorted(self.mapnum_atom_dict.keys()) == list(range(1, len(self.mapnum_atom_dict.keys()) + 1)):
            raise ValueError("The atom mapping number provided should be a continuous positive integer.")
        if not self._check_two_atoms_connected(1, 2):
            raise ValueError("The atom mapping numbers 1 and 2 should be connected.")

    @staticmethod
    def _decimal_to_binary(x, padding_length=2):
        bit = "0" * (padding_length - 1) + bin(x)[2:]
        return bit[-padding_length:]  # -2 means we only take 4 possible states

    def _check_two_atoms_connected(self, map_num_1, map_num_2):
        connect_num_list = []
        atom_1 = self.mapnum_atom_dict[map_num_1]
        for bond in atom_1.GetBonds():
            connect_num_list += [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()]
        return map_num_2 in connect_num_list

    def _set_initial_two_atoms_parameters(self):
        # Atom type 1 and 2
        atom_state_1 = self._decimal_to_binary(self.atom_type_to_idx[self.mapnum_atom_dict[1].GetSymbol()])
        if atom_state_1 != "01":
            self.parameters_value[0] = 1.
            if atom_state_1 == "11":
                self.parameters_value[1] = 1.
        self.parameters_indicator[[0, 1]] = 1

        atom_state_2 = self._decimal_to_binary(self.atom_type_to_idx[self.mapnum_atom_dict[2].GetSymbol()])
        self.parameters_value[2] = (int(atom_state_1[-1]) + int(atom_state_2[0])) % 2
        self.parameters_value[4] = (self.parameters_value[2] + int(atom_state_2[-1])) % 2
        self.parameters_indicator[[2, 3, 4, 5]] = 1

        # Bond 2-1
        bond_state_2_1 = self._decimal_to_binary(self.bond_type_to_idx[self.mapnum_bond_dict[(1, 2)]])
        self.parameters_value[6] = int(bond_state_2_1[0])
        self.parameters_value[7] = int(bond_state_2_1[1])
        self.parameters_indicator[[6, 7]] = 1

    def _process_remaining_atoms(self):
        used_part = 8
        # starts with number 3 atom, since the first two atoms have been determined by previous circuit
        for map_num in range(3, len(self.mapnum_atom_dict.keys()) + 1):
            atom_state = self._decimal_to_binary(self.atom_type_to_idx[self.mapnum_atom_dict[map_num].GetSymbol()])
            self.parameters_value[used_part] = int(atom_state[0])
            self.parameters_value[used_part + 1] = int(atom_state[1])
            self.parameters_indicator[[used_part, used_part + 1, used_part + 2]] = 1
            used_part += 3

            for previous_atom_map in range(1, map_num):
                bond_type = self.mapnum_bond_dict.get((previous_atom_map, map_num), None)
                bond_state = self._decimal_to_binary(self.bond_type_to_idx[bond_type])
                first_gate_index = used_part + previous_atom_map - 1
                second_gate_index = used_part + (map_num - 1) + 2 * (previous_atom_map - 1)
                third_gate_index = used_part + (map_num - 1) + 2 * (previous_atom_map - 1) + 1
                self.parameters_value[first_gate_index] = int(bool(int(bond_state[0]) + int(bond_state[1])))
                self.parameters_value[second_gate_index] = int(bond_state[0])
                self.parameters_value[third_gate_index] = 1 - int(bond_state[1])
                self.parameters_indicator[[first_gate_index, second_gate_index, third_gate_index]] = 1
            used_part += 3 * (map_num - 1)
        self.used_part = used_part

    def _apply_disable_connectivity(self):
        for map_num in self.disable_connectivity_position:
            fixed_part = self.used_part
            for f_idx in range(self.num_fixed_atoms + 1, self.num_heavy_atom + 1):
                fixed_part += 3  # atom 3 weights
                first_gate_index = fixed_part + map_num - 1
                second_gate_index = fixed_part + (f_idx - 1) + 2 * (map_num - 1)
                third_gate_index = fixed_part + (f_idx - 1) + 2 * (map_num - 1) + 1
                self.parameters_indicator[[first_gate_index, second_gate_index, third_gate_index]] = 1
                fixed_part += (f_idx - 1) * 3

    def _generate_constrained_parameters(self):
        self._set_initial_two_atoms_parameters()
        self._process_remaining_atoms()
        self._apply_disable_connectivity()
        return self.parameters_value, self.parameters_indicator
    
    def softmax_temperature(self, weight_vector, temperature):
        weight_vector /= temperature
        exps = np.exp(weight_vector)
        return exps / np.sum(exps)

    def generate_conditional_random_weights(self, random_seed:int=0, chemistry_constraint:bool=True, temperature:float=0.2):
        random.seed(random_seed)
        random_weight_vector = np.array([random.random() for _ in range(self.length_all_weight_vector)])
        random_weight_vector = random_weight_vector * (1 - self.parameters_indicator) + self.parameters_value
        if chemistry_constraint:
            fixed_part = self.used_part
            for f_idx in range(self.num_fixed_atoms + 1, self.num_heavy_atom + 1):
                fixed_part += 3  # atom 3 weights
                first_gate_index_list = [fixed_part + i for i in range(f_idx-1)]
                constrained_first_gate_index_list = [idx for idx in first_gate_index_list if not self.parameters_indicator[idx]]
                random_weight_vector[constrained_first_gate_index_list] = self.softmax_temperature(random_weight_vector[constrained_first_gate_index_list],
                                                                                                   temperature)

                second_gate_index_list = [fixed_part + (f_idx-1) + 2*i for i in range(f_idx-1)]
                constrained_second_gate_index_list = [idx for idx in second_gate_index_list if not self.parameters_indicator[idx]]
                random_weight_vector[constrained_second_gate_index_list]*= 0.5

                third_gate_index_list = [fixed_part + (f_idx-1) + 2*i+1 for i in range(f_idx-1)]
                constrained_third_gate_index_list = [idx for idx in third_gate_index_list if not self.parameters_indicator[idx]]
                random_weight_vector[constrained_third_gate_index_list] *= 0.5
                random_weight_vector[constrained_third_gate_index_list] += 0.5

                fixed_part += (f_idx - 1) * 3
        else:
            pass # do nothing
        
        return random_weight_vector
    
    def apply_chemistry_constraint(self, random_weight_vector: np.array, temperature:float=0.2):
        new_random_weight_vector = np.copy(random_weight_vector)
        fixed_part = self.used_part
        for f_idx in range(max(self.num_fixed_atoms, 2) + 1, self.num_heavy_atom + 1):
            fixed_part += 3  # atom 3 weights
            first_gate_index_list = [fixed_part + i for i in range(f_idx-1)]
            constrained_first_gate_index_list = [idx for idx in first_gate_index_list if not self.parameters_indicator[idx]]
            new_random_weight_vector[constrained_first_gate_index_list] = self.softmax_temperature(new_random_weight_vector[constrained_first_gate_index_list],
                                                                                                temperature)

            second_gate_index_list = [fixed_part + (f_idx-1) + 2*i for i in range(f_idx-1)]
            constrained_second_gate_index_list = [idx for idx in second_gate_index_list if not self.parameters_indicator[idx]]
            new_random_weight_vector[constrained_second_gate_index_list]*= 0.5

            third_gate_index_list = [fixed_part + (f_idx-1) + 2*i+1 for i in range(f_idx-1)]
            constrained_third_gate_index_list = [idx for idx in third_gate_index_list if not self.parameters_indicator[idx]]
            new_random_weight_vector[constrained_third_gate_index_list] *= 0.5
            new_random_weight_vector[constrained_third_gate_index_list] += 0.5

            fixed_part += (f_idx - 1) * 3

        return new_random_weight_vector

if __name__ == "__main__":
    # cwg = ConditionalWeightsGenerator(num_heavy_atom=5, smarts="[O:1]1[C:2][C:3]1", disable_connectivity_position = [1])
    # print(cwg.generate_conditional_random_weights(random_seed=1))
    smarts = "[C:1][C:2](=[O:3])[C:4]"
    disable_connectivity_position = [2, 3]
    cwg = ConditionalWeightsGenerator(num_heavy_atom=6, smarts=smarts, disable_connectivity_position = disable_connectivity_position)
    print(cwg.generate_conditional_random_weights(random_seed=1))