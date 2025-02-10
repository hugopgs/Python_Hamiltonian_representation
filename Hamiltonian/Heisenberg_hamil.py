from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import integrate
from typing import Callable, List, Tuple
from itertools import product
from Hamiltonian.Hamiltonian import Hamiltonian
import math as math



class Heisenberg_Hamil(Hamiltonian):
    """ Heisenberg Hamiltonian defined as : H_{Heisenberg}=\sum_{n=0}^{numQs-1}(jx.X_nX_{n+1}+jy.Y_nY_{n+1}+jz.Z_nZ_{n+1}) 
    Args:
        Hamiltonian (_type_): _description_
    """

    
    def __init__(self, n:int, jx:float,jy:float,jz: float, boundarie_conditions:bool=False):
        def Jx(t):
            return 1*jx*t
        def Jy(t):
            return 1*jy*t
        def Jz(t):
            return 1*jz*t
        if boundarie_conditions:
             terms = [
                (gate, [k, (k + 1)%n], Jx if gate == "XX" else Jy if gate == "YY" else Jz)
                for k, gate in product(range(n), ["XX", "YY", "ZZ"])
            ]
        else:
            terms = [
                (gate, [k, k + 1], Jx if gate == "XX" else Jy if gate == "YY" else Jz)
                for k, gate in product(range(n-1), ["XX", "YY", "ZZ"])
            ]


        super().__init__(n, terms)  
    
    def all_dimer_coverings(self,n):
        """
        Generate all perfect matchings (dimer coverings) of the set {0,1,...,n-1}.
        Each matching is returned as a list of pairs (i, j) with i < j.
        """
        
        if n == 0:
            yield []
        else:
            first = 0
            for partner in range(1, n):
                pair = (first, partner)
                others = [x for x in range(n) if x not in pair]
                
                for rest in self.all_dimer_coverings(len(others)):
                    mapping = {}
                    c = 0
                    for x in range(n):
                        if x not in pair:
                            mapping[c] = x
                            c += 1
                    
                    mapped_rest = [(mapping[a], mapping[b]) for (a, b) in rest]
                    yield [pair] + mapped_rest

    def ground_state(self):
        """
        Builds and returns the resonating-valence-bond (RVB) state for n qubits
        (n even) summed over all dimer coverings, as a NumPy array of length 2^n
        in the computational (Z) basis.
        
        The returned state is normalized.
        """
        n=self.nqubits
        if n % 2 != 0:
            raise ValueError("n must be even to form perfect dimer coverings.")
        
        dim = 2**n
        psi = np.zeros(dim, dtype=np.complex128)
        pair_factor = (1.0 / math.sqrt(2))**(n//2)
        
        for D in self.all_dimer_coverings(n):
            for b in range(dim):
                csign = 1  
                valid = True
                for (i, j) in D:
                    bit_i = (b >> i) & 1
                    bit_j = (b >> j) & 1
                    if bit_i == 0 and bit_j == 1:
                        csign *= +1
                    elif bit_i == 1 and bit_j == 0:
                        csign *= -1
                    else:
                        valid = False
                        break
                if valid:
                    psi[b] += pair_factor * csign
        
        norm = np.linalg.norm(psi)
        if norm > 1e-15:
            psi /= norm
        
        return psi

      
    
    
    