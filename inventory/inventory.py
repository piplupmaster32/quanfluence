import numpy as np
import math
import dimod
from neal import SimulatedAnnealingSampler

def bounded_coefficient_encoding(kappa_x, mu_x):
    """Bounded-Coefficient Encoding Algorithm"""
    
    if kappa_x < 2**(math.floor(math.log2(mu_x)) + 1):
        # Use binary encoding
        log_kappa = math.floor(math.log2(kappa_x))
        binary_coeffs = [2**i for i in range(log_kappa + 1)]
        
        if sum(binary_coeffs[:-1]) < kappa_x:
            binary_coeffs[-1] = kappa_x - sum(binary_coeffs[:-1])
        
        return binary_coeffs
    else:
        # Use bounded-coefficient encoding
        rho = math.floor(math.log2(mu_x)) + 1
        nu = kappa_x - sum(2**(i-1) for i in range(1, rho + 1))
        eta = math.floor(nu / mu_x)
        
        c_x = []
        for i in range(1, rho + 1):
            c_x.append(2**(i-1))
        for i in range(eta):
            c_x.append(mu_x)
        
        remainder = nu - eta * mu_x
        if remainder != 0:
            c_x.append(remainder)
        
        return c_x

def create_inventory_bqm(S, A, R, C, D, mu_x_x, mu_x_s, penalty_strength=1000):
    """
    Create BQM for inventory management problem
    
    Parameters:
    - S: selling prices (n_products,)
    - A: recipe matrix (n_raw_materials, n_products) 
    - R: raw material upper limits (n_raw_materials,)
    - C: raw material costs (n_raw_materials,)
    - D: demand limits (n_products,)
    - penalty_strength: strength of constraint penalties
    
    Returns:
    - bqm: Binary Quadratic Model
    - encoding_info: Information about variable encoding
    """
    
    n_products = len(S)
    n_raw_materials = len(R)
    
    # Calculate objective coefficients
    P = C @ A - S  # Cost of raw materials minus selling price
    
    print(f"Problem size: {n_products} products, {n_raw_materials} raw materials")
    print(f"Objective coefficients P: {P}")
    print(f"Demand limits D: {D}")
    print(f"Raw material limits R: {R}")
    
    # Step 1: Encode production variables x1, x2, ..., xn with bounds D1, D2, ..., Dn
    x_coefficients = []
    x_widths = []
    
    for i in range(n_products):
        coeffs = bounded_coefficient_encoding(int(D[i]), mu_x_x[i])
        x_coefficients.append(coeffs)
        x_widths.append(len(coeffs))
        print(f"x{i+1} encoding: {len(coeffs)} binary variables, coefficients: {coeffs}")
    
    # Step 2: Encode slack variables s1, s2, ..., sm with bounds R1, R2, ..., Rm
    s_coefficients = []
    s_widths = []
    
    for i in range(n_raw_materials):
        coeffs = bounded_coefficient_encoding(int(R[i]), mu_x_s[i])
        s_coefficients.append(coeffs)
        s_widths.append(len(coeffs))
        print(f"s{i+1} encoding: {len(coeffs)} binary variables, coefficients: {coeffs}")
    
    # Step 3: Create variable mapping
    total_x_vars = sum(x_widths)
    total_s_vars = sum(s_widths)
    total_vars = total_x_vars + total_s_vars
    
    print(f"Total binary variables: {total_vars} ({total_x_vars} for x, {total_s_vars} for s)")
    
    # Create mapping from (variable_type, index, bit) to binary variable index
    var_map = {}
    current_idx = 0
    
    # Map x variables
    for i in range(n_products):
        for j in range(x_widths[i]):
            var_map[('x', i, j)] = current_idx
            current_idx += 1
    
    # Map s variables  
    for i in range(n_raw_materials):
        for j in range(s_widths[i]):
            var_map[('s', i, j)] = current_idx
            current_idx += 1
    
    # Step 4: Create BQM
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, 'BINARY')
    
    # Add linear terms for objective function: P @ x
    for i in range(n_products):
        for j in range(x_widths[i]):
            var_idx = var_map[('x', i, j)]
            coefficient = P[i] * x_coefficients[i][j]
            bqm.add_variable(var_idx, coefficient)
    
    # Step 5: Add constraint penalties: A @ x + s = R
    # For each raw material constraint: sum(A[i,j] * x[j]) + s[i] = R[i]
    
    for i in range(n_raw_materials):  # For each raw material constraint
        
        # Linear terms: A[i,j] * x[j] terms
        linear_terms = {}
        
        # Add A[i,j] * x[j] terms (positive contribution)
        for j in range(n_products):
            for k in range(x_widths[j]):
                var_idx = var_map[('x', j, k)]
                coeff = A[i, j] * x_coefficients[j][k]
                if var_idx in linear_terms:
                    linear_terms[var_idx] += coeff
                else:
                    linear_terms[var_idx] = coeff
        
        # Add s[i] terms (positive contribution)
        for k in range(s_widths[i]):
            var_idx = var_map[('s', i, k)]
            coeff = s_coefficients[i][k]
            if var_idx in linear_terms:
                linear_terms[var_idx] += coeff
            else:
                linear_terms[var_idx] = coeff
        
        # Add quadratic penalty terms for constraint: (A @ x + s - R)^2
        # Expand: (sum_terms - R[i])^2 = sum_terms^2 - 2*R[i]*sum_terms + R[i]^2
        
        # Quadratic terms: interactions between different variables
        var_indices = list(linear_terms.keys())
        for idx1 in range(len(var_indices)):
            for idx2 in range(idx1, len(var_indices)):
                v1, v2 = var_indices[idx1], var_indices[idx2]
                coeff1, coeff2 = linear_terms[v1], linear_terms[v2]
                
                if v1 == v2:
                    # Diagonal term: coeff^2
                    penalty = penalty_strength * coeff1 * coeff2
                    bqm.add_variable(v1, penalty)
                else:
                    # Off-diagonal term: 2 * coeff1 * coeff2
                    penalty = 2 * penalty_strength * coeff1 * coeff2
                    bqm.add_interaction(v1, v2, penalty)
        
        # Linear penalty terms: -2 * R[i] * sum_terms
        for var_idx, coeff in linear_terms.items():
            penalty = -2 * penalty_strength * R[i] * coeff
            bqm.add_variable(var_idx, penalty)
        
        # Constant term: R[i]^2 (added to offset)
        bqm.offset += penalty_strength * R[i] * R[i]
    
    # Encoding information for solution decoding
    encoding_info = {
        'n_products': n_products,
        'n_raw_materials': n_raw_materials,
        'x_coefficients': x_coefficients,
        's_coefficients': s_coefficients,
        'x_widths': x_widths,
        's_widths': s_widths,
        'var_map': var_map,
        'total_vars': total_vars,
        'A': A,
        'R': R,
        'P': P,
        'D': D
    }
    
    return bqm, encoding_info

def decode_solution(sample, encoding_info):
    """Decode binary solution back to original variables"""
    
    x_coeffs = encoding_info['x_coefficients']
    s_coeffs = encoding_info['s_coefficients']
    x_widths = encoding_info['x_widths']
    s_widths = encoding_info['s_widths']
    var_map = encoding_info['var_map']
    n_products = encoding_info['n_products']
    n_raw_materials = encoding_info['n_raw_materials']
    
    # Decode x variables
    x_values = np.zeros(n_products)
    for i in range(n_products):
        for j in range(x_widths[i]):
            var_idx = var_map[('x', i, j)]
            if sample[var_idx] == 1:
                x_values[i] += x_coeffs[i][j]
    
    # Decode s variables
    s_values = np.zeros(n_raw_materials)
    for i in range(n_raw_materials):
        for j in range(s_widths[i]):
            var_idx = var_map[('s', i, j)]
            if sample[var_idx] == 1:
                s_values[i] += s_coeffs[i][j]
    
    return x_values, s_values

def validate_solution(x_values, s_values, encoding_info):
    """Validate that solution satisfies constraints"""
    
    A = encoding_info['A']
    R = encoding_info['R']
    D = encoding_info['D']
    P = encoding_info['P']
    
    print("Solution Validation:")
    print(f"Production quantities: {x_values}")
    print(f"Slack variables: {s_values}")
    
    # Check demand constraints
    demand_satisfied = all(x_values[i] <= D[i] for i in range(len(D)))
    print(f"Demand constraints satisfied: {demand_satisfied}")
    
    # Check raw material constraints
    raw_material_usage = A @ x_values
    constraint_satisfaction = raw_material_usage + s_values
    constraints_satisfied = np.allclose(constraint_satisfaction, R, atol=1e-6)
    
    print(f"Raw material usage: {raw_material_usage}")
    print(f"Raw material limits: {R}")
    print(f"Constraint A@x + s: {constraint_satisfaction}")
    print(f"Should equal R: {R}")
    print(f"Raw material constraints satisfied: {constraints_satisfied}")
    
    # Calculate objective value
    objective_value = P @ x_values
    print(f"Objective value: {objective_value}")
    
    return demand_satisfied and constraints_satisfied

def calculate_profit_metrics(x_values, encoding_info):
    """Calculate detailed profit and cost metrics"""
    
    S = encoding_info.get('S', None)
    A = encoding_info['A']
    C = encoding_info.get('C', None)
    
    # Need to extract S and C from problem data
    # We'll calculate them from P if not available
    P = encoding_info['P']
    
    metrics = {}
    
    # Raw material usage
    raw_material_usage = A @ x_values
    metrics['raw_material_usage'] = raw_material_usage
    
    # If we have the original cost and selling price data
    if S is not None and C is not None:
        # Revenue calculation
        total_revenue = S @ x_values
        metrics['total_revenue'] = total_revenue
        metrics['revenue_by_product'] = S * x_values
        
        # Cost calculation  
        total_raw_material_cost = C @ raw_material_usage
        metrics['total_raw_material_cost'] = total_raw_material_cost
        metrics['cost_by_material'] = C * raw_material_usage
        
        # Profit calculation
        total_profit = total_revenue - total_raw_material_cost
        metrics['total_profit'] = total_profit
        metrics['profit_margin'] = total_profit / total_revenue if total_revenue > 0 else 0
    
    # Objective function value (cost-based objective from BQM)
    objective_value = P @ x_values
    metrics['objective_value'] = objective_value
    
    return metrics

def analyze_resource_utilization(x_values, s_values, encoding_info):
    """Analyze how efficiently resources are being used"""
    
    A = encoding_info['A']
    R = encoding_info['R']
    D = encoding_info['D']
    n_products = encoding_info['n_products']
    n_raw_materials = encoding_info['n_raw_materials']
    
    analysis = {}
    
    # Raw material utilization
    raw_usage = A @ x_values
    utilization_rates = raw_usage / R
    analysis['raw_material_utilization'] = {
        'usage': raw_usage,
        'capacity': R,
        'utilization_rate': utilization_rates,
        'slack': s_values,
        'unused_capacity': R - raw_usage
    }
    
    # Product demand fulfillment
    demand_fulfillment = x_values / D
    analysis['demand_fulfillment'] = {
        'production': x_values,
        'demand': D,
        'fulfillment_rate': demand_fulfillment,
        'unmet_demand': D - x_values
    }
    
    # Bottleneck analysis
    bottleneck_material = np.argmax(utilization_rates)
    analysis['bottlenecks'] = {
        'most_constrained_material': bottleneck_material,
        'highest_utilization_rate': utilization_rates[bottleneck_material],
        'utilization_ranking': np.argsort(utilization_rates)[::-1]
    }
    
    return analysis

def comprehensive_solution_report(x_values, s_values, encoding_info, include_sensitivity=True):
    """Generate a comprehensive report of the solution"""
    
    print("\n" + "="*60)
    print("           COMPREHENSIVE SOLUTION REPORT")
    print("="*60)
    
    # Basic solution
    print("\n📊 OPTIMAL SOLUTION:")
    print("-" * 30)
    for i, val in enumerate(x_values):
        print(f"Product {i+1} production: {val:.2f} units")
    
    print(f"\nSlack variables: {s_values}")
    
    # Profit metrics
    print("\n💰 FINANCIAL ANALYSIS:")
    print("-" * 30)
    metrics = calculate_profit_metrics(x_values, encoding_info)
    
    if 'total_profit' in metrics:
        print(f"Total Revenue: ${metrics['total_revenue']:.2f}")
        print(f"Total Raw Material Cost: ${metrics['total_raw_material_cost']:.2f}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Profit Margin: {metrics['profit_margin']*100:.1f}%")
        
        print(f"\nRevenue by Product:")
        for i, rev in enumerate(metrics['revenue_by_product']):
            print(f"  Product {i+1}: ${rev:.2f}")
            
        print(f"\nCost by Raw Material:")
        for i, cost in enumerate(metrics['cost_by_material']):
            print(f"  Material {i+1}: ${cost:.2f}")
    
    print(f"\nObjective Function Value: {metrics['objective_value']:.2f}")
    
    # Resource utilization
    print("\n🏭 RESOURCE UTILIZATION ANALYSIS:")
    print("-" * 30)
    resource_analysis = analyze_resource_utilization(x_values, s_values, encoding_info)
    
    rm_util = resource_analysis['raw_material_utilization']
    print("Raw Material Utilization:")
    for i in range(len(rm_util['usage'])):
        util_pct = rm_util['utilization_rate'][i] * 100
        print(f"  Material {i+1}: {rm_util['usage'][i]:.1f}/{rm_util['capacity'][i]:.1f} "
              f"({util_pct:.1f}% utilized, {rm_util['unused_capacity'][i]:.1f} unused)")
    
    demand_analysis = resource_analysis['demand_fulfillment']
    print(f"\nDemand Fulfillment:")
    for i in range(len(demand_analysis['production'])):
        fulfill_pct = demand_analysis['fulfillment_rate'][i] * 100
        print(f"  Product {i+1}: {demand_analysis['production'][i]:.1f}/{demand_analysis['demand'][i]:.1f} "
              f"({fulfill_pct:.1f}% fulfilled, {demand_analysis['unmet_demand'][i]:.1f} unmet)")
    
    # Bottleneck analysis
    bottlenecks = resource_analysis['bottlenecks']
    print(f"\n🚦 BOTTLENECK ANALYSIS:")
    print("-" * 30)
    print(f"Most constrained material: Material {bottlenecks['most_constrained_material']+1} "
          f"({bottlenecks['highest_utilization_rate']*100:.1f}% utilized)")
    
    print("Materials ranked by utilization:")
    for rank, material_idx in enumerate(bottlenecks['utilization_ranking']):
        util_rate = rm_util['utilization_rate'][material_idx] * 100
        print(f"  {rank+1}. Material {material_idx+1}: {util_rate:.1f}%")
    
    return metrics, resource_analysis

def compare_multiple_solutions(sampleset, encoding_info, top_n=5):
    """Compare top N solutions from the sampleset"""
    
    print(f"\n🔍 COMPARING TOP {top_n} SOLUTIONS:")
    print("="*50)
    
    solutions_data = []
    
    for i, sample_data in enumerate(sampleset.data()):
        if i >= top_n:
            break
            
        sample = sample_data.sample
        energy = sample_data.energy
        
        x_vals, s_vals = decode_solution(sample, encoding_info)
        is_valid = validate_solution(x_vals, s_vals, encoding_info)
        metrics = calculate_profit_metrics(x_vals, encoding_info)
        
        solution_info = {
            'rank': i+1,
            'energy': energy,
            'x_values': x_vals,
            's_values': s_vals,
            'is_valid': is_valid,
            'metrics': metrics
        }
        solutions_data.append(solution_info)
        
        print(f"\nSolution {i+1}:")
        print(f"  Energy: {energy:.2f}")
        print(f"  Valid: {is_valid}")
        print(f"  Production: {x_vals}")
        if 'total_profit' in metrics:
            print(f"  Profit: ${metrics['total_profit']:.2f}")
        print(f"  Objective: {metrics['objective_value']:.2f}")
    
    return solutions_data

def sensitivity_analysis(encoding_info, base_solution, parameter_variations=0.1):
    """Perform basic sensitivity analysis on key parameters"""
    
    print(f"\n🎯 SENSITIVITY ANALYSIS:")
    print("="*40)
    print("(Showing impact of ±10% parameter changes)")
    
    # This is a simplified sensitivity analysis
    # In practice, you'd re-solve the BQM for each parameter change
    
    base_x, base_s = base_solution
    base_metrics = calculate_profit_metrics(base_x, encoding_info)
    
    if 'total_profit' in base_metrics:
        base_profit = base_metrics['total_profit']
        print(f"\nBase case profit: ${base_profit:.2f}")
        
        # Simulate impact of demand changes
        D = encoding_info['D']
        print(f"\nImpact of demand changes:")
        for i, demand in enumerate(D):
            # Simplified: assume proportional production change
            new_production = base_x.copy()
            new_production[i] *= 1.1  # 10% increase
            new_production[i] = min(new_production[i], demand * 1.1)
            
            new_metrics = calculate_profit_metrics(new_production, encoding_info)
            if 'total_profit' in new_metrics:
                profit_change = new_metrics['total_profit'] - base_profit
                print(f"  +10% demand for Product {i+1}: ${profit_change:+.2f} profit change")