from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import integrate
from typing import Callable, List, Tuple
from itertools import product
from Hamiltonian.Hamiltonian import Hamiltonian

@dataclass
class Spin_Chain_Hamil(Hamiltonian):
    nqubits: int
    terms: List[Tuple[str, List[int], Callable[[float], float]]]
    
    def __init__(self, n, freqs, func=None):
        if func is not None:
            def J(t):
                return func(t)
        else:
            def J(t):
                return np.cos(20 * t * np.pi)

        terms = [
            (gate, [k, (k + 1) % n], J)
            for k, gate in product(range(n), ["XX", "YY", "ZZ"])
        ]
        terms += [("Z", [k], lambda t, k=k: freqs[k]) for k in range(n)]
        
        super().__init__(n, terms)  
   