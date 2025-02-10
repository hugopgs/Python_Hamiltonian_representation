from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Tuple
from Hamiltonian.Hamiltonian import Hamiltonian
from qiskit import QuantumCircuit
@dataclass
class Ising_Hamil(Hamiltonian):
     nqubits: int
     terms: List[Tuple[str, List[int], Callable[[float], float]]]
    
     def __init__(self, n, J, d:float = None, transverse:bool=True):
          terms = [("ZZ", [k, (k + 1)%n], lambda t: -1*J*t)for k in range(n)]
          self.J=J
          self.d=d
          self.transverse=transverse
          if self.d<1.e-11:
               self.d=0
          if transverse:
               terms += [("X", [k], lambda t: -1*d*t) for k in range(n)]
          else:
               terms += [("Z", [k], lambda t: -1*d*t) for k in range(n)]
          super().__init__(n, terms)  

     
     
               
