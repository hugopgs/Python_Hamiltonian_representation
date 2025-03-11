from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import integrate
from typing import Callable, List, Tuple
from itertools import product
from qiskit import QuantumCircuit, transpile
from typing import Union
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
        def fn(t): return np.linalg.norm(self.coefs(t), 1)
        return integrate.quad(fn, 0, T, limit=100)[0]

    def __len__(self):
        return len(self.terms)

    def matrix(self):
        from functools import reduce
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, 1.j], [-1.j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.array([[1, 0], [0, 1]])
        res = []
        for term in self.terms:
            tmp = [I]*self.nqubits
            if term[0] == 'Z':
                tmp[term[1][0]] = Z
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0] == 'X':
                tmp[term[1][0]] = X
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0] == 'Y':
                tmp[term[1][0]] = Y
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0] == 'XX':
                tmp[term[1][0]] = X
                tmp[term[1][1]] = X
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0] == 'YY':
                tmp[term[1][0]] = Y
                tmp[term[1][1]] = Y
                res.append(term[2](1)*reduce(np.kron, tmp))
            if term[0] == 'ZZ':
                tmp[term[1][0]] = Z
                tmp[term[1][1]] = Z
                res.append(term[2](1)*reduce(np.kron, tmp))
        matrix = np.zeros((2**self.nqubits, 2**self.nqubits),
                          dtype=('complex128'))
        for mat in res:
            matrix += mat
        return matrix

    def gen_quantum_circuit(self, T: float, init_state: Union[np.ndarray, list, QuantumCircuit] = None, trotter_step: int = 1000, serialize=False) -> QuantumCircuit:
        """Generate a qiskit quantum circuite given a list of gates.
        Args:
            gates (tuple[str, list[int], float]): The list of gates to generate the circuit from, in the form of : ("Gate", [nq1, nq2], parameters)
            exemple: ("XX",[2,3], 1) gate XX on qubit 2, 3 with parameter 1
            nq (int): total number of qubit
            init_state (QuantumCircuit, optional): A Quantum circuit to put at the beginning of the circuit. Defaults to None.

        Returns:
            QuantumCircuit: The quantum circuit representation of the given gates
        """
        nq = self.nqubits
        circ = QuantumCircuit(nq)
        steps = np.linspace(0, T, trotter_step, endpoint=True)
        if isinstance(init_state, (np.ndarray, list)):
            circ.initialize(init_state, [i for i in range(nq)], normalize=True)
            if serialize:
                circ = transpile(circ, basis_gates=['rx', 'ry', 'rz', 'cx'])

        for step in steps:
            for pauli, qubits, coef in self.get_term(step):
                circ.append(self.__rgate(pauli, 2*coef*T/trotter_step), qubits)

        if isinstance(init_state, QuantumCircuit):
            circ = init_state.compose(circ, [i for i in range(nq)])
        if serialize:
            qasm = self.serialize_circuit(circ)
            return qasm
        else:
            return circ

    def __rgate(self, pauli, r):
        return {
            "X": RXGate(r),
            "Z": RZGate(r),
            "XX": RXXGate(r),
            "YY": RYYGate(r),
            "ZZ": RZZGate(r),
        }[pauli]

    def serialize_circuit(circuit: QuantumCircuit) -> str:
        """Serialize a QuantumCircuit into JSON."""
        from qiskit.qasm2 import dumps
        qasm_string = dumps(circuit)  # Convert circuit to OpenQASM 2.0
        gate_definitions = """
        gate rxx(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
        gate ryy(theta) a, b { ry(-pi/2) a; ry(-pi/2) b; cx a, b; rz(theta) b; cx a, b; ry(pi/2) a; ry(pi/2) b; }
        gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
        """
        qasm_lines = qasm_string.split("\n")
        qasm_lines.insert(2, gate_definitions.strip())
        qasm = "".join(qasm_lines)
        return qasm

    def eigenvalues(self) -> np.ndarray:
        """Return the real part of the eigenvalues of a matrix , i.e the eigenenergies of the Hamiltonian from is matrix representation
        Args:
            matrix (np.ndarray): a square matrix that represant an Hamiltonian 

        Returns:
        np.ndarray: the eigenenergies of the Hamiltonian
        """
        matrix = self.matrix()
        eigval_list = np.linalg.eig(matrix)[0]
        eigval_list[np.abs(eigval_list) < 1.e-11] = 0
        return eigval_list.real

    def energy_gap(self):     
        """Calculate the energy gap betweend different energy level.
        Remove double energy level.

        Args:
            Energies (list[float]): list of energies to calculate energy gap.

        Returns:
            list[float]: energy gap
        """
        rnd=4
        Energies=self.eigenvalues()
        Energies[np.abs(Energies) < 1.e-11] = 0
        Energies_no_double = []
        for energie in Energies:
            if np.round(energie, rnd) not in np.round(Energies_no_double, rnd):
                Energies_no_double.append(energie)
        res = []
        for i in range(len(Energies_no_double)-1):
            for k in range(i+1, len(Energies_no_double)):
                res.append(Energies_no_double[i]-Energies_no_double[k])
        res = np.abs(res)
        res= np.round(res, rnd)
        res_no_double = []
        for gap in res:
            gap = np.round(gap, rnd)
            if gap not in res_no_double:
                res_no_double.append(gap.tolist())

        res_no_double = np.array(res_no_double)
        res_no_double[np.abs(res_no_double) < 1.e-11] = 0
        return np.sort(res_no_double)