import dimod
import neal
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from collections import Counter
import math
import pandas as pd
from itertools import chain
import time
import networkx as nx

def bounded_coefficient_encoding(kappa_x, mu_x):
    """
    Bounded-Coefficient Encoding Algorithm
    
    Inputs:
        kappa_x: upper bound on the integer variable x
        mu_x: upper bound on the coefficients of the encoding (must be <= kappa_x)
    
    Output:
        c_x: integer encoding coefficients (as a list)
    """
    # Validate inputs
    if mu_x > kappa_x:
        raise ValueError("mu_x must be <= kappa_x")
    
    # Check the condition for the simple case
    if kappa_x < 2**(math.floor(math.log2(mu_x)) + 1):
        # Simple case: return the basic encoding
        log_mu_x = math.floor(math.log2(mu_x))
        
        # Create the encoding vector
        c_x = []
        
        # Add powers of 2: 2^0, 2^1, ..., 2^(floor(log(mu_x)) - 1)
        for i in range(log_mu_x):
            c_x.append(2**i)
        
        # Add the final term: kappa_x - sum(2^(i-1)) for i=1 to floor(log(mu_x))
        sum_powers = sum(2**(i-1) for i in range(1, log_mu_x + 1))
        c_x.append(kappa_x - sum_powers)
        
        return c_x
    
    else:
        # Complex case
        # Compute rho = floor(log(mu_x)) + 1
        rho = math.floor(math.log2(mu_x)) + 1
        
        # Compute nu = kappa_x - sum(2^(i-1)) for i=1 to rho
        nu = kappa_x - sum(2**(i-1) for i in range(1, rho + 1))
        
        # Compute eta = floor(nu / mu_x)
        eta = math.floor(nu / mu_x)
        
        # Create the encoding vector
        c_x = []
        
        # Add coefficients based on the piecewise function
        # For i = 1, ..., rho: c_i = 2^(i-1)
        for i in range(1, rho + 1):
            c_x.append(2**(i-1))
        
        # For i = rho + 1, ..., rho + eta: c_i = mu_x
        for i in range(rho + 1, rho + eta + 1):
            c_x.append(mu_x)
        
        # For i = rho + eta + 1 (if nu - eta*mu_x != 0): c_i = nu - eta*mu_x
        if nu - eta * mu_x != 0:
            c_x.append(nu - eta * mu_x)
        
        return c_x

def verify_encoding(c_x, kappa_x, mu_x):
    """
    Verify that the encoding can represent all values from 0 to kappa_x
    and that all coefficients are <= mu_x
    """
    # Check coefficient bounds
    for coeff in c_x:
        if coeff > mu_x:
            return False, f"Coefficient {coeff} exceeds mu_x = {mu_x}"
    
    # Check that we can represent kappa_x
    max_representable = sum(c_x)
    if max_representable < kappa_x:
        return False, f"Cannot represent kappa_x = {kappa_x}, max representable = {max_representable}"
    
    return True, "Encoding is valid"

def test_algorithm_1():
    """Test the algorithm with various inputs"""
    test_cases = [
        (10, 5),    # Simple case
        (100, 20),  # Complex case
        (15, 8),    # Another test
        (7, 3),     # Small values
    ]
    
    for kappa_x, mu_x in test_cases:
        print(f"\nTesting with kappa_x = {kappa_x}, mu_x = {mu_x}")
        try:
            c_x = bounded_coefficient_encoding(kappa_x, mu_x)
            print(f"Encoding coefficients: {c_x}")
            print(f"Number of coefficients: {len(c_x)}")
            print(f"Sum of coefficients: {sum(c_x)}")
            
            is_valid, message = verify_encoding(c_x, kappa_x, mu_x)
            print(f"Verification: {message}")
            
        except Exception as e:
            print(f"Error: {e}")

def find_upper_bounds_coefficients(kappa, q, Q, epsilon_l, epsilon_c, n):
    """
    Algorithm 2: Finding the Upper Bounds on the Coefficients of the Encoding
    
    Inputs:
        kappa: parameter kappa
        q: parameter q
        Q: matrix Q (assumed to be n x n)
        epsilon_l: parameter epsilon_l
        epsilon_c: parameter epsilon_c
        n: number of variables
    
    Output:
        mu_x: list of upper bounds mu^(x_i) for i = 1, 2, ..., n
    """
    
    # Compute Q*kappa + q
    Q_kappa_plus_q = []
    for i in range(n):
        sum_val = q[i] if i < len(q) else 0  # Handle case where q might be shorter
        for j in range(n):
            if i < len(Q) and j < len(Q[i]):
                sum_val += Q[i][j] * kappa[j] if j < len(kappa) else Q[i][j] * kappa[0]
        Q_kappa_plus_q.append(sum_val)
    
    # Set m_l = min_i {|[Q*kappa + q]_i|}
    m_l = min(abs(val) for val in Q_kappa_plus_q)
    
    # Set m_c = min_{i,j} {|Q_{i,j}|, |Q_{j,i}|}
    m_c = float('inf')
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            if i < len(Q) and j < len(Q[i]):
                m_c = min(m_c, abs(Q[i][j]))
            if j < len(Q) and i < len(Q[j]):
                m_c = min(m_c, abs(Q[j][i]))
    
    if m_c == float('inf'):
        m_c = 1  # Default value if Q is empty
    
    # Initialize mu^(x_i) for all variables
    mu_x = []
    for i in range(n):
        # Initialize mu^(x_i) = floor(min{m_l / |epsilon_l|, sqrt(m_l / |Q_{i,i}| * epsilon_c)})
        term1 = m_l / abs(epsilon_l) if epsilon_l != 0 else float('inf')
        
        # For the second term, we need Q_{i,i}
        Q_ii = Q[i][i] if i < len(Q) and i < len(Q[i]) else 1
        term2 = math.sqrt(m_l / (abs(Q_ii) * abs(epsilon_c))) if Q_ii != 0 and epsilon_c != 0 else float('inf')
        
        mu_x_i = math.floor(min(term1, term2))
        mu_x.append(max(1, mu_x_i))  # Ensure at least 1
    
    # Main iterative loop
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        # Check if any mu^(x_i) * mu^(x_j) > m_c / |Q_{i,j}| * epsilon_c
        found_violation = False
        violating_pairs = []
        
        for i in range(n):
            for j in range(n):
                if i != j and i < len(Q) and j < len(Q[i]):
                    Q_ij = Q[i][j]
                    if Q_ij != 0:
                        threshold = m_c / (abs(Q_ij) * abs(epsilon_c)) if epsilon_c != 0 else float('inf')
                        if mu_x[i] * mu_x[j] > threshold:
                            violating_pairs.append((i, j, mu_x[i] * mu_x[j] - threshold))
                            found_violation = True
        
        if not found_violation:
            break
        
        # Find i, j = arg max_{i,j} {mu^(x_i) * mu^(x_j) - m_c / |Q_{i,j}| * epsilon_c}
        best_i, best_j = max(violating_pairs, key=lambda x: x[2])[:2]
        
        # Compute xi_i and xi_j
        xi_i = (kappa[best_i] if best_i < len(kappa) else kappa[0]) / (mu_x[best_i] - 1) + (kappa[best_j] if best_j < len(kappa) else kappa[0]) / mu_x[best_j]
        xi_j = (kappa[best_i] if best_i < len(kappa) else kappa[0]) / mu_x[best_i] + (kappa[best_j] if best_j < len(kappa) else kappa[0]) / (mu_x[best_j] - 1)
        
        # Update mu values based on comparison
        if xi_i < xi_j:
            mu_x[best_i] = mu_x[best_i] - 1
        else:
            mu_x[best_j] = mu_x[best_j] - 1
        
        # Ensure mu values don't go below 1
        mu_x[best_i] = max(1, mu_x[best_i])
        mu_x[best_j] = max(1, mu_x[best_j])
        
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"Warning: Algorithm reached maximum iterations ({max_iterations})")
    
    return mu_x

def create_test_case():
    """Create a simple test case for the algorithm"""
    n = 3
    kappa = [10, 15, 20]
    q = [5, -3, 8]
    Q = [
        [2, 1, -1],
        [1, 3, 2],
        [-1, 2, 4]
    ]
    epsilon_l = 0.1
    epsilon_c = 0.05
    
    return kappa, q, Q, epsilon_l, epsilon_c, n

def test_algorithm_2():
    """Test the upper bounds algorithm"""
    print("Testing Algorithm 2: Finding Upper Bounds on Coefficients")
    print("=" * 60)
    
    kappa, q, Q, epsilon_l, epsilon_c, n = create_test_case()
    
    print(f"Input parameters:")
    print(f"kappa = {kappa}")
    print(f"q = {q}")
    print(f"Q = {Q}")
    print(f"epsilon_l = {epsilon_l}")
    print(f"epsilon_c = {epsilon_c}")
    print(f"n = {n}")
    print()
    
    try:
        mu_x = find_upper_bounds_coefficients(kappa, q, Q, epsilon_l, epsilon_c, n)
        
        print(f"Results:")
        print(f"Upper bounds mu^(x_i): {mu_x}")
        
        # Display results for each variable
        for i in range(n):
            print(f"mu^(x_{i+1}) = {mu_x[i]}")
        
        print(f"\nTotal coefficient bound budget: {sum(mu_x)}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

def validate_bounds(mu_x, kappa, Q, epsilon_c):
    """Validate that the computed bounds satisfy the constraints"""
    n = len(mu_x)
    violations = []
    
    for i in range(n):
        for j in range(n):
            if i != j and i < len(Q) and j < len(Q[i]):
                Q_ij = Q[i][j]
                if Q_ij != 0:
                    # Check if mu^(x_i) * mu^(x_j) * |Q_{i,j}| * epsilon_c is reasonable
                    product = mu_x[i] * mu_x[j] * abs(Q_ij) * abs(epsilon_c)
                    if product > 1000:  # Arbitrary threshold for demonstration
                        violations.append((i, j, product))
    
    return violations