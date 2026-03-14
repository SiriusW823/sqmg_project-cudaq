from qiskit import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np
from typing import List, Union
from collections import Counter

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .utils import MoleculeQuantumStateGenerator, CircuitBuilder, DynamicCircuitBuilder, ConditionalWeightsGenerator

def get_token(file_path):
    with open(file_path) as f:
        data = f.read()
    token = data.strip()
    return token

class MoleculeGenerator():
    def __init__(self, num_heavy_atom:int, all_weight_vector:Union[List[float], np.ndarray]=None, backend_name:str="qiskit_aer",
                 temperature:float=0.2, dynamic_circuit:bool=True, remove_bond_disconnection:bool=True, chemistry_constraint:bool=True):
        self.num_heavy_atom = num_heavy_atom
        self.all_weight_vector = all_weight_vector
        self.backend_name = backend_name
        self._build_backend()
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.temperature = temperature
        self.dynamic_circuit = dynamic_circuit
        self.num_qubits = num_heavy_atom*(num_heavy_atom+1)
        self.num_ancilla_qubits = num_heavy_atom - 1
        self.data_generator = MoleculeQuantumStateGenerator(heavy_atom_size=num_heavy_atom, ncpus=1, sanitize_method="strict")

    def _build_backend(self):
        """ Construct the backend based on the user choice. """
        if self.backend_name == "qiskit_aer":
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
        elif self.backend_name in ["ibm", "ibmq"]:
            from qiskit_ibm_runtime import QiskitRuntimeService
            my_token = get_token("../docs/ibmq_tokens.txt")
            service = QiskitRuntimeService(channel="ibm_quantum", token=my_token)
            self.backend = service.least_busy(simulator=False, operational=True)
            print("Using IBM Quantum backend:", self.backend)
        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        
    def generate_quantum_circuit(self, random_seed):
        if self.dynamic_circuit:
            self.qc = DynamicCircuitBuilder(self.num_heavy_atom, self.temperature, 
                                 self.remove_bond_disconnection, self.chemistry_constraint).generate_quantum_circuit(self.all_weight_vector, random_seed)
        else:
            self.qc = CircuitBuilder(self.num_heavy_atom, self.temperature, 
                                 self.remove_bond_disconnection, self.chemistry_constraint).generate_quantum_circuit(self.all_weight_vector, random_seed)
        
    def update_weight_vector(self, all_weight_vector):
        self.all_weight_vector = all_weight_vector

    def sample_molecule(self, num_sample, random_seed:int=0):
        self.generate_quantum_circuit(random_seed)
        self.transpiled_qc = self.pm.run(self.qc)
        self.sampler = Sampler(mode=self.backend)
        self.sampler.options.default_shots = num_sample
        job = self.sampler.run([self.transpiled_qc])
        if self.backend_name in ["ibm", "ibmq"]:
            print(f">>> Job ID: {job.job_id()}")
            return
        results = job.result()
        # if self.dynamic_circuit:
        #     string_results = results[0].data.c0.get_bitstrings()
        # else:
        string_results = results[0].data.c.get_bitstrings()
        counts = dict(Counter(string_results))

        smiles_dict = {}
        num_valid_molecule = 0
        for key, value in counts.items():
            if self.dynamic_circuit:
                key = "".join(key.split())
            smiles = self.data_generator.QuantumStateToSmiles(self.data_generator.post_process_quantum_state(key))
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + value
            if smiles:
                num_valid_molecule += value
        validity = num_valid_molecule / num_sample
        uniqueness = (len(smiles_dict.keys()) - 1) / num_valid_molecule
        return smiles_dict, validity, uniqueness
    
if __name__ == "__main__":
    num_heavy_atom = 5
    random_seed = 3
    cwg = ConditionalWeightsGenerator(num_heavy_atom, smarts="[O:1]1[C:2][C:3]1", disable_connectivity_position = [1])
    random_weight_vector = cwg.generate_conditional_random_weights(random_seed)
    mg = MoleculeGenerator(num_heavy_atom, all_weight_vector=random_weight_vector, dynamic_circuit=True) 
    smiles_dict, validity, diversity = mg.sample_molecule(20000)
    print(smiles_dict)
    print("Validity: {:.2f}%".format(validity*100))
    print("Diversity: {:.2f}%".format(diversity*100))