import numpy as np
import dimod

class ILPtoQUBOConverter:
    def __init__(self, c, A, b, var_bounds, penalty=1000):
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.var_bounds = var_bounds
        self.penalty = penalty
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.binary_mapping = {}
        self.bqm = dimod.BinaryQuadraticModel('BINARY')  # Using dimod's BQM

    def _bounded_coefficient_encoding(self, U, var_name):
        """Same as before"""
        if U == 0:
            return {}

        coeffs = []
        d = 1
        s_i = 1
        current_sum = 0

        while current_sum + s_i < U:
            coeffs.append(s_i)
            current_sum += s_i
            s_i = d * s_i + 1

        if U - current_sum > 0:
            coeffs.append(U - current_sum)

        encoding = {f"{var_name}_{i}": s for i, s in enumerate(coeffs)}
        return encoding

    def convert(self):
        # 1. Handle variable bounds and create binary mappings
        shifted_c = self.c.copy()
        shifted_A = self.A.copy()
        shifted_b = self.b.copy()

        for i in range(self.num_vars):
            lower_bound, upper_bound = self.var_bounds[i]
            if lower_bound != 0:
                shifted_b -= self.A[:, i] * lower_bound
                if len(shifted_c.shape) > 1:
                    shifted_c[i] -= 2 * np.dot(self.c[i,:], lower_bound)
                else:
                    shifted_c[i] -= lower_bound * self.c[i]

            U = upper_bound - lower_bound
            self.binary_mapping[f'y_{i}'] = self._bounded_coefficient_encoding(U, f'y_{i}')

        # 2. Add objective to BQM
        for i in range(self.num_vars):
            if self.c[i] != 0:
                for bin_var, coeff in self.binary_mapping[f'y_{i}'].items():
                    self.bqm.add_variable(bin_var, self.c[i] * coeff)

        # 3. Add constraints as penalties
        for i in range(self.num_constraints):
            constraint_expr = {}
            
            # LHS of constraint
            for j in range(self.num_vars):
                if self.A[i, j] != 0:
                    for bin_var, coeff in self.binary_mapping[f'y_{j}'].items():
                        constraint_expr[bin_var] = constraint_expr.get(bin_var, 0) + self.A[i, j] * coeff
            
            # Introduce slack variable
            min_lhs = 0 
            max_lhs = 0
            for var, coeff in constraint_expr.items():
                if coeff > 0:
                    max_lhs += coeff
                else:
                    min_lhs += coeff
            
            max_slack = self.b[i] - min_lhs
            
            slack_var_name = f's_{i}'
            self.binary_mapping[slack_var_name] = self._bounded_coefficient_encoding(int(max_slack), slack_var_name)

            for bin_var, coeff in self.binary_mapping[slack_var_name].items():
                constraint_expr[bin_var] = constraint_expr.get(bin_var, 0) + coeff

            # Constant part of the constraint
            constant = -self.b[i]

            # Add penalty terms to BQM
            # First collect all variables in the constraint
            vars_in_constraint = [v for v in constraint_expr.keys() if v != 'constant']
            
            # Add linear terms
            for var in vars_in_constraint:
                coeff = constraint_expr[var]
                self.bqm.add_variable(var, self.penalty * coeff * (2 * constant + coeff))
            
            # Add quadratic terms
            for j, var1 in enumerate(vars_in_constraint):
                for var2 in vars_in_constraint[j+1:]:
                    coeff1 = constraint_expr[var1]
                    coeff2 = constraint_expr[var2]
                    self.bqm.add_interaction(var1, var2, 2 * self.penalty * coeff1 * coeff2)

        return self.bqm, self.binary_mapping
    
def convert_dict_to_matrix(qubo_dict, binary_map):
    """
    Converts a QUBO dictionary to a square Q matrix.

    Args:
        qubo_dict (dict): The QUBO dictionary from the converter.
        binary_map (dict): The mapping of integer to binary variables.

    Returns:
        np.array: The final Q matrix.
        list: The ordered list of binary variable names corresponding
              to the rows/columns of the Q matrix.
    """
    # 1. Get a sorted, unique list of all binary variables
    all_binary_vars = set()
    for var_mappings in binary_map.values():
        for bin_var in var_mappings.keys():
            all_binary_vars.add(bin_var)

    # Also check the QUBO dict itself in case of variables not in the map
    # (like constants, though we handle them)
    for term in qubo_dict.keys():
        for var in term:
            if var != 'constant':
                 all_binary_vars.add(var)


    sorted_vars = sorted(list(all_binary_vars))
    var_to_index = {var: i for i, var in enumerate(sorted_vars)}
    num_vars = len(sorted_vars)

    # 2. Initialize an empty Q matrix
    Q = np.zeros((num_vars, num_vars))

    # 3. Populate the Q matrix from the dictionary
    for term, coeff in qubo_dict.items():
        if len(term) == 1 and term[0] != 'constant':
            # This is a linear term, goes on the diagonal
            idx = var_to_index[term[0]]
            Q[idx, idx] = coeff
        elif len(term) == 2:
            # This is a quadratic term
            idx1 = var_to_index[term[0]]
            idx2 = var_to_index[term[1]]
            # To create a symmetric Q matrix, we add half to (i,j) and half to (j,i)
            # For an upper-triangular matrix, you would just do Q[idx1, idx2] = coeff
            Q[idx1, idx2] += coeff / 2.0
            Q[idx2, idx1] += coeff / 2.0

    return Q, sorted_vars