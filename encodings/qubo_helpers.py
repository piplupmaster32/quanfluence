import numpy as np
import scipy.linalg as la
import math

def generate_quip(n=5, kappa=50, sparsity=0.5, seed=42):
    """Generate a convex QUIP instance with known optimal solution"""
    np.random.seed(seed)
    U2 = np.array([-2, -1, 0, 1, 2])
    Q_initial = np.random.choice(U2, size=(n, n))
    Q_initial = (Q_initial + Q_initial.T) / 2
    sparsity_mask = np.random.random((n, n)) < sparsity
    Q_initial = Q_initial * sparsity_mask
    
    for i in range(n):
        if Q_initial[i, i] == 0:
            Q_initial[i, i] = np.random.choice([1, 2])
    
    eigenvalues = la.eigvals(Q_initial)
    lambda_min = np.min(eigenvalues)
    lambda_add = np.ceil((abs(min(lambda_min, 0))) + np.random.random())
    Q = Q_initial + lambda_add * np.eye(n)
    
    eigenvalues_final = la.eigvals(Q)
    x_star = np.zeros(n, dtype=int)
    num_nonzero = np.random.randint(1, n)
    nonzero_indices = np.random.choice(n, size=num_nonzero, replace=False)
    for i in nonzero_indices:
        x_star[i] = np.random.randint(1, kappa + 1)
    
    q = -2 * Q @ x_star
    kappa_vec = np.full(n, kappa, dtype=int)
    
    objective_at_x_star = x_star.T @ Q @ x_star + q.T @ x_star
    gradient_at_x_star = 2 * Q @ x_star + q
    
    print(f"Gradient at x* (should be ~0): {gradient_at_x_star}")
    return Q, q, kappa_vec, x_star

def create_binary_encoder(kappa):
    """Create binary encoding parameters for given upper bound"""
    width = math.floor(math.log2(kappa)) + 1
    coefficients = [2**i for i in range(width)]
    return width, coefficients

def encode_integer(value, width, coefficients):
    """Encode single integer to binary using coefficients"""
    binary = []
    remaining = value
    for coeff in reversed(coefficients):
        if remaining >= coeff:
            binary.append(1)
            remaining -= coeff
        else:
            binary.append(0)
    binary.reverse()
    return binary

def decode_binary(binary, coefficients):
    """Decode binary representation back to integer"""
    return sum(bit * coeff for bit, coeff in zip(binary, coefficients))

def encode_vector(integer_vector, kappa):
    """Encode full integer vector to binary"""
    width, coefficients = create_binary_encoder(kappa)
    
    binary_vector = []
    for val in integer_vector:
        if val < 0 or val > kappa:
            raise ValueError(f"Value {val} outside bounds [0, {kappa}]")
        binary_repr = encode_integer(val, width, coefficients)
        binary_vector.extend(binary_repr)
    
    return binary_vector, width, coefficients

def decode_vector(binary_vector, n_variables, width, coefficients):
    """Decode binary vector back to integers"""
    integer_vector = []
    
    for i in range(n_variables):
        start = i * width
        end = start + width
        var_binary = binary_vector[start:end]
        integer_val = decode_binary(var_binary, coefficients)
        integer_vector.append(integer_val)
    
    return integer_vector

def create_binary_encoding_matrix(n_variables, width, coefficients):
    """Create encoding matrix C where x = C @ y"""
    total_binary_vars = width * n_variables
    C = np.zeros((n_variables, total_binary_vars))
    
    for i in range(n_variables):
        start = i * width
        end = start + width
        C[i, start:end] = coefficients
    
    return C

def convert_to_qubo(Q_binary, q_binary):
    """
    Convert quadratic problem min y^T Q_binary y + q_binary^T y to QUBO format
    QUBO format: min y^T Q_qubo y where linear terms are on the diagonal
    """
    n = Q_binary.shape[0]
    Q_qubo = Q_binary.copy()
    
    # Add linear terms to diagonal
    for i in range(n):
        Q_qubo[i, i] += q_binary[i]
    
    return Q_qubo

def find_coefficient_upper_bounds(Q, q, kappa_vec, ell=0.01, c=0.01):
    """
    Algorithm 2: Finding the Upper Bounds on the Coefficients of the Encoding
    Based on precision requirements and resilience conditions.
    """
    n = len(kappa_vec)
    
    # Compute Qκ + q (equation before (20))
    Qkappa_plus_q = Q @ kappa_vec + q
    
    # Minimum absolute values for linear and quadratic terms
    ml = np.min([abs(val) for val in Qkappa_plus_q if abs(val) > 1e-10])
    if ml == 0:
        ml = 1e-6  # Avoid division by zero
    
    mc = np.min([abs(Q[i,i]) for i in range(n)] + 
                [abs(Q[i,j]) for i in range(n) for j in range(i+1,n) if abs(Q[i,j]) > 1e-10])
    if mc == 0:
        mc = 1e-6
    
    print(f"ml (min linear coeff magnitude): {ml:.6f}")
    print(f"mc (min quadratic coeff magnitude): {mc:.6f}")
    
    # Initialize μ_xi using inequalities (22) and (23)
    mu_x = np.zeros(n)
    for i in range(n):
        # From equation (22): μ_xi ≤ ml / (|[Qκ + q]_i| * ell)
        linear_bound = ml / (abs(Qkappa_plus_q[i]) * ell) if abs(Qkappa_plus_q[i]) > 1e-10 else np.inf
        
        # From equation (23): μ_xi ≤ sqrt(mc / (|Qii| * c))
        quadratic_bound = np.sqrt(mc / (abs(Q[i,i]) * c)) if abs(Q[i,i]) > 1e-10 else np.inf
        
        mu_x[i] = min(linear_bound, quadratic_bound)
        
    print(f"Initial μ bounds: {mu_x}")
    
    # Iteratively decrease μ_xi to satisfy inequality (24): μ_xi * μ_xj ≤ mc / (|Qij| * c)
    max_iterations = 100
    for iteration in range(max_iterations):
        violated = False
        max_violation = 0
        violating_pair = (-1, -1)
        
        # Check all pairs
        for i in range(n):
            for j in range(i+1, n):
                if abs(Q[i,j]) > 1e-10:
                    required_product = mc / (abs(Q[i,j]) * c)
                    current_product = mu_x[i] * mu_x[j]
                    
                    if current_product > required_product:
                        violation = current_product - required_product
                        if violation > max_violation:
                            max_violation = violation
                            violating_pair = (i, j)
                            violated = True
        
        if not violated:
            break
            
        # Greedily decrease μ_xi or μ_xj for the most violating pair
        i, j = violating_pair
        
        # Choose which one to decrease based on width estimate (κ/μ)
        width_i_decrease = kappa_vec[i] / (mu_x[i] - 1) if mu_x[i] > 1 else np.inf
        width_j_decrease = kappa_vec[j] / (mu_x[j] - 1) if mu_x[j] > 1 else np.inf
        combined_width_i = width_i_decrease + kappa_vec[j] / mu_x[j]
        combined_width_j = kappa_vec[i] / mu_x[i] + width_j_decrease
        
        if combined_width_i < combined_width_j and mu_x[i] > 1:
            mu_x[i] = max(1, mu_x[i] - 1)
        elif mu_x[j] > 1:
            mu_x[j] = max(1, mu_x[j] - 1)
        else:
            # Both are at minimum, break to avoid infinite loop
            break
    
    # Ensure all μ_xi are at least 1 and integers
    mu_x = np.maximum(1, np.floor(mu_x)).astype(int)
    
    print(f"Final μ bounds after constraint satisfaction: {mu_x}")
    return mu_x

def bounded_coefficient_encoding(kappa_x, mu_x):
    
    """
    Algorithm 1: Bounded-Coefficient Encoding
    
    Args:
        kappa_x: upper bound on the integer variable x
        mu_x: upper bound on the coefficients of the encoding
    
    Returns:
        c_x: integer encoding coefficients
    """
    print(f"  Encoding variable with κ={kappa_x}, μ={mu_x}")
    
    # Check if we should use binary encoding (equation 5)
    if kappa_x < 2**(math.floor(math.log2(mu_x)) + 1):
        print(f"    Using binary encoding (κ < 2^⌊log(μ)⌋+1)")
        # Binary encoding with adjustment for κ
        log_kappa = math.floor(math.log2(kappa_x))
        binary_coeffs = [2**i for i in range(log_kappa + 1)]
        
        # Adjust last coefficient to exactly reach κ
        if sum(binary_coeffs[:-1]) < kappa_x:
            binary_coeffs[-1] = kappa_x - sum(binary_coeffs[:-1])
        
        return binary_coeffs
    
    else:
        print(f"    Using bounded-coefficient encoding")
        # Bounded-coefficient encoding (equation 6)
        rho = math.floor(math.log2(mu_x)) + 1
        nu = kappa_x - sum(2**(i-1) for i in range(1, rho + 1))
        eta = math.floor(nu / mu_x)
        
        print(f"    ρ={rho}, ν={nu}, η={eta}")
        
        # Build coefficient vector
        c_x = []
        
        # First ρ coefficients: 2^(i-1) for i = 1, ..., ρ
        for i in range(1, rho + 1):
            c_x.append(2**(i-1))
        
        # Next η coefficients: μ_x
        for i in range(eta):
            c_x.append(mu_x)
        
        # Last coefficient if needed: ν - η*μ_x
        remainder = nu - eta * mu_x
        if remainder != 0:
            c_x.append(remainder)
        
        print(f"    Coefficients: {c_x}")
        return c_x

def encode_integer_bounded(value, coefficients):
    
    """Encode single integer using bounded coefficient encoding (greedy algorithm)"""
    binary = []
    remaining = value
    
    # Go through coefficients from largest to smallest (greedy)
    for coeff in reversed(coefficients):
        if remaining >= coeff:
            binary.append(1)
            remaining -= coeff
        else:
            binary.append(0)
    
    # Reverse to match coefficient order
    binary.reverse()
    return binary

def decode_integer_bounded(binary, coefficients):
    """Decode binary representation back to integer using bounded coefficients"""
    return sum(bit * coeff for bit, coeff in zip(binary, coefficients))

def create_bounded_encoding_system(kappa_vec, mu_vec):
    """Create the complete bounded coefficient encoding system for all variables"""
    n_variables = len(kappa_vec)
    
    print(f"\n=== CREATING BOUNDED COEFFICIENT ENCODING SYSTEM ===")
    
    # Get coefficients for each variable
    all_coefficients = []
    widths = []
    
    for i in range(n_variables):
        coeffs = bounded_coefficient_encoding(kappa_vec[i], mu_vec[i])
        all_coefficients.append(coeffs)
        widths.append(len(coeffs))
        print(f"Variable x[{i}]: κ={kappa_vec[i]}, μ={mu_vec[i]}, width={len(coeffs)}, coeffs={coeffs}")
    
    total_binary_vars = sum(widths)
    print(f"Total binary variables: {total_binary_vars}")
    
    return all_coefficients, widths, total_binary_vars

def encode_vector_bounded(integer_vector, kappa_vec, mu_vec):
    """Encode integer vector using bounded coefficient encoding"""
    all_coefficients, widths, total_binary_vars = create_bounded_encoding_system(kappa_vec, mu_vec)
    
    binary_vector = []
    for i, val in enumerate(integer_vector):
        if val < 0 or val > kappa_vec[i]:
            raise ValueError(f"Value {val} outside bounds [0, {kappa_vec[i]}]")
        
        binary_repr = encode_integer_bounded(val, all_coefficients[i])
        binary_vector.extend(binary_repr)
    
    return binary_vector, all_coefficients, widths

def decode_vector_bounded(binary_vector, all_coefficients, widths):
    """Decode binary vector back to integers using bounded coefficient encoding"""
    integer_vector = []
    start_idx = 0
    
    for i, width in enumerate(widths):
        end_idx = start_idx + width
        var_binary = binary_vector[start_idx:end_idx]
        integer_val = decode_integer_bounded(var_binary, all_coefficients[i])
        integer_vector.append(integer_val)
        start_idx = end_idx
    
    return integer_vector

def create_bounded_encoding_matrix(n_variables, all_coefficients, widths):
    """Create encoding matrix C for bounded coefficient encoding where x = C @ y"""
    total_binary_vars = sum(widths)
    C = np.zeros((n_variables, total_binary_vars))
    
    start_idx = 0
    for i in range(n_variables):
        end_idx = start_idx + widths[i]
        C[i, start_idx:end_idx] = all_coefficients[i]
        start_idx = end_idx
    
    return C

def convert_to_qubo(Q_binary, q_binary):
    """Convert quadratic problem to QUBO format"""
    n = Q_binary.shape[0]
    Q_qubo = Q_binary.copy()
    
    # Add linear terms to diagonal
    for i in range(n):
        Q_qubo[i, i] += q_binary[i]
    
    return Q_qubo