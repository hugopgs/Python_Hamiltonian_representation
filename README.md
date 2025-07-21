# Quantum Hamiltonian representation

## Overview

This repository provides a framework for constructing quantum Hamiltonians using Qiskit. The implementation allows for easy definition of time-dependent Hamiltonians, including Ising, Heisenberg, and general spin chain models. The `Hamiltonian` class serves as the base, while specialized classes extend it to represent specific models. Additionally, the framework enables conversion of these Hamiltonians into quantum circuits for execution on quantum simulators or real quantum hardware.

## Hamiltonian Class

The `Hamiltonian` class is the core structure for defining a quantum Hamiltonian. It allows users to specify:

- The number of qubits (`nqubits`)
- A list of terms in the Hamiltonian, where each term consists of:
  - A Pauli operator (e.g., `X`, `Y`, `Z`, `XX`, `YY`, `ZZ`)
  - The qubits on which the operator acts
  - A time-dependent function defining the coefficient of the term

### Key Methods:

- `get_term(t)`: Returns the terms of the Hamiltonian evaluated at time `t`.
- `coefs(t)`: Extracts the coefficients of the Hamiltonian terms at time `t`.
- `l1_norm(T)`: Computes the integral of the L1 norm of the coefficients over time.
- `get_matrix()`: Constructs the Hamiltonian matrix representation.
- `gen_quantum_circuit(t, init_state)`: Generates a Qiskit quantum circuit that implements the Hamiltonian evolution at time `t`.

## Types of Hamiltonians

### 1. Ising Hamiltonian (`Ising_Hamil`)

The Ising model describes a system of interacting spins with nearest-neighbor interactions. The Hamiltonian is given by:

$H = -J \sum_{i} Z_i Z_{i+1} - d \sum_{i} X_i- h \sum_{i} Z_i$

where:

- $J$ controls the interaction strength between neighboring spins.
- $d$ and $h$ represents the strength of an external field either transverse or longitudinal.

### 2. Heisenberg Hamiltonian

The Heisenberg model includes interactions in all three Pauli bases:

$H = -\sum_{i} (J_x X_i X_{i+1} +J_y Y_i Y_{i+1} +J_z Z_i Z_{i+1})$

where $J_x,J_y,J_z$ defines the interactions strength. This model describes quantum magnetism and spin chain dynamics.

### 3. Spin Chain Hamiltonian

A more general spin chain model allows different types of interactions and external fields:

$H = \sum_{i,j} J_{ij} P_i P_j + \sum_i h_i P_i$

where:

- $J_{ij}$ defines interaction strengths.
- $h_i$ represents local field strengths.
- $P$ can be any Pauli operator ($X$, $Y$, or $Z$).

## Usage

1. Install the necessary dependencies:
   ```sh
   pip install qiskit scipy numpy
   ```
2. Create an instance of a Hamiltonian:
   ```python
   from ising_hamiltonian import Ising_Hamil
   H = Ising_Hamil(n=4, transverse=1.0, longitudinal=0.5, fully_connected=False, bundarie_conditions=True)
   ```
3. Generate the Hamiltonian matrix:
   ```python
   matrix = H.get_matrix()
   ```
4. Create a qiskit quantum circuit representation:
   ```python
   qc = H.gen_quantum_circuit(t=1.0)
   print(qc)
   ```

## License

This project is released under the MIT License.

