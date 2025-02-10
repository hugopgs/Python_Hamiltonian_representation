from dataclasses import dataclass
import numpy as np
from scipy import integrate
from typing import Callable, List, Tuple,Union
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, RZGate, RXGate

@dataclass
class Hamiltonian:
    nqubits: int
    terms: List[Tuple[str, List[int], Callable[[float], float]]]

    def __post_init__(self):
        print("The number of qubit:" + str(self.nqubits))
        print("Number of terms in the Hamiltonian:" + str(len(self.terms)))
        

    def get_term(self, t):
        return [(term[0], term[1], term[2](t)) for term in self.terms]

    def coefs(self, t: float):
        return [term[2](t) for term in self.terms]

    def l1_norm(self, T: float):
        fn = lambda t: np.linalg.norm(self.coefs(t), 1)
        return integrate.quad(fn, 0, T, limit=100)[0]

    def __len__(self):
        return len(self.terms)

    def get_matrix(self):
        import time 
        from functools import reduce
        X=np.array([[0,1],[1,0]])
        Y=np.array([[0,1.j],[-1.j,0]])
        Z=np.array([[1,0],[0,-1]])
        I=np.array([[1,0],[0,1]])
        res=[]
        for term in self.terms:
            tmp=[I]*self.nqubits
            if term[0]=='Z':
                tmp[term[1][0]]=Z
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0]=='X':
                tmp[term[1][0]]=X
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0]=='Y':
                tmp[term[1][0]]=Y
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0]=='XX':
                tmp[term[1][0]]=X
                tmp[term[1][1]]=X
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0]=='YY':
                tmp[term[1][0]]=Y
                tmp[term[1][1]]=Y
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0]=='ZZ':
                tmp[term[1][0]]=Z
                tmp[term[1][1]]=Z
                res.append(term[2](1)*reduce(np.kron, tmp))
        matrix=np.zeros((2**self.nqubits,2**self.nqubits), dtype=('complex128'))        
        for mat in res:
            matrix+=mat
        return matrix
    
    def gen_quantum_circuit(self, t: float, init_state: Union[np.ndarray, list, QuantumCircuit]=None)->QuantumCircuit:
        """Generate a qiskit quantum circuite given a list of gates.

        Args:
            gates (tuple[str, list[int], float]): The list of gates to generate the circuit from, in the form of : ("Gate", [nq1, nq2], parameters)
            exemple: ("XX",[2,3], 1) gate XX on qubit 2, 3 with parameter 1
            nq (int): total number of qubit
            init_state (QuantumCircuit, optional): A Quantum circuit to put at the beginning of the circuit. Defaults to None.

        Returns:
            QuantumCircuit: The quantum circuit representation of the given gates
        """
        nq=self.nqubits
        circ = QuantumCircuit(nq)
        if isinstance(init_state,(np.ndarray, list)):
            circ.initialize(init_state, [i for i in range(nq)], normalize=True)
            
        for pauli, qubits, coef in self.get_term(t):
            circ.append(self.__rgate(pauli, coef), qubits)
            
        if  isinstance(init_state,QuantumCircuit):
            circ = init_state.compose(circ,[i for i in range(nq)])
        return circ
    
    def __rgate(self, pauli, r):
        return {
            "X": RXGate(r),
            "Z": RZGate(r),
            "XX": RXXGate(r),
            "YY": RYYGate(r),
            "ZZ": RZZGate(r),
        }[pauli]
