import numpy as np
from dimod import BinaryQuadraticModel, BINARY
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import matplotlib.pyplot as plt

class TSPDimod:
    """
    Traveling Salesman Problem implementation using DIMOD for D-Wave quantum computers.
    
    This class formulates TSP as a Binary Quadratic Model (BQM) where:
    - Each binary variable x[i,j] = 1 if city i is visited at position j
    - Constraints ensure valid tours (each city visited once, each position filled once)
    - Objective minimizes total travel distance
    """
    
    def __init__(self, distance_matrix, city_names=None):
        """
        Initialize TSP instance.
        
        Args:
            distance_matrix: 2D array/list of distances between cities
            city_names: Optional list of city names
        """
        self.distance_matrix = np.array(distance_matrix)
        self.n_cities = len(distance_matrix)
        self.city_names = city_names or [f"City_{i}" for i in range(self.n_cities)]
        self.bqm = None
        
    def create_bqm(self, constraint_strength=None):
        """
        Create the Binary Quadratic Model for TSP.
        
        Uses binary variables x[i,j] where x[i,j] = 1 if city i is at position j.
        
        Args:
            constraint_strength: Weight for constraint violations (auto-calculated if None)
            
        Returns:
            BinaryQuadraticModel: The formulated BQM
        """
        n = self.n_cities
        
        # Auto-calculate constraint strength if not provided
        if constraint_strength is None:
            max_distance = np.max(self.distance_matrix)
            constraint_strength = max_distance * n * 2
        
        # Create BQM
        bqm = BinaryQuadraticModel(vartype=BINARY)
        
        # Create binary variables x[i,j] for city i at position j
        variables = {}
        for i in range(n):
            for j in range(n):
                var_name = f'x_{i}_{j}'
                variables[(i, j)] = var_name
                bqm.add_variable(var_name)
        
        # Objective: minimize total travel distance
        # Add quadratic terms for consecutive cities in the tour
        for j in range(n):
            next_j = (j + 1) % n
            for i1 in range(n):
                for i2 in range(n):
                    if i1 != i2:
                        # Cost of going from city i1 at position j to city i2 at position next_j
                        weight = self.distance_matrix[i1][i2]
                        bqm.add_quadratic(variables[(i1, j)], 
                                        variables[(i2, next_j)], weight)
        
        # Constraint 1: Each city must be visited exactly once
        # Sum over positions for each city = 1
        for i in range(n):
            city_vars = [variables[(i, j)] for j in range(n)]
            # Add penalty for not visiting city exactly once
            self._add_constraint_exactly_one(bqm, city_vars, constraint_strength)
        
        # Constraint 2: Each position must have exactly one city
        # Sum over cities for each position = 1
        for j in range(n):
            position_vars = [variables[(i, j)] for i in range(n)]
            # Add penalty for not having exactly one city at each position
            self._add_constraint_exactly_one(bqm, position_vars, constraint_strength)
        
        self.bqm = bqm
        self.variables = variables
        return bqm
    
    def _add_constraint_exactly_one(self, bqm, variables, strength):
        """
        Add constraint that exactly one variable in the list should be 1.
        Implements: (sum(x_i) - 1)^2 = sum(x_i) + 2*sum(x_i * x_j) - 2*sum(x_i)
        Which simplifies to: 2*sum(x_i * x_j) - sum(x_i)
        """
        # Linear terms: -1 * sum(x_i)
        for var in variables:
            bqm.add_linear(var, -strength)
        
        # Quadratic terms: 2 * sum(x_i * x_j) for i != j
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Avoid duplicate pairs
                    bqm.add_quadratic(var1, var2, 2 * strength)
    
    def solve_classical(self, num_reads=100):
        """
        Solve using classical simulated annealing (for testing/comparison).
        
        Args:
            num_reads: Number of samples to generate
            
        Returns:
            dict: Solution results
        """
        from dimod import SimulatedAnnealingSampler
        
        if self.bqm is None:
            self.create_bqm()
        
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(self.bqm, num_reads=num_reads)
        
        return self._process_results(sampleset)
    
    def solve_hybrid(self):
        """
        Solve using D-Wave Leap Hybrid solver.
        
        Returns:
            dict: Solution results
        """
        if self.bqm is None:
            self.create_bqm()
        
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(self.bqm, label="TSP Problem")
        
        return self._process_results(sampleset)
    
    def solve_quantum(self, num_reads=100, annealing_time=20):
        """
        Solve using D-Wave quantum annealer.
        
        Args:
            num_reads: Number of quantum annealing runs
            annealing_time: Annealing time in microseconds
            
        Returns:
            dict: Solution results
        """
        if self.bqm is None:
            self.create_bqm()
        
        # Use D-Wave quantum annealer with embedding
        sampler = EmbeddingComposite(DWaveSampler())
        
        sampleset = sampler.sample(self.bqm, 
                                 num_reads=num_reads,
                                 annealing_time=annealing_time,
                                 label="TSP Problem")
        
        return self._process_results(sampleset)
    
    def _process_results(self, sampleset):
        """Process and format the solution results."""
        best_solution = sampleset.first
        
        # Extract tour from solution
        tour = self._extract_tour(best_solution.sample)
        
        # Calculate tour distance
        tour_distance = self._calculate_tour_distance(tour)
        
        # Check constraint violations
        violations = self._check_constraints(best_solution.sample)
        
        return {
            'tour': tour,
            'distance': tour_distance,
            'energy': best_solution.energy,
            'city_names': [self.city_names[i] for i in tour] if tour else [],
            'is_valid': self._is_valid_tour(tour),
            'constraint_violations': violations,
            'sample': best_solution.sample
        }
    
    def _extract_tour(self, sample):
        """Extract tour sequence from BQM solution."""
        n = self.n_cities
        tour = []
        
        # For each position, find which city is assigned (x[i,j] = 1)
        for j in range(n):
            city_at_position = None
            for i in range(n):
                var_name = f'x_{i}_{j}'
                if sample.get(var_name, 0) == 1:
                    city_at_position = i
                    break
            
            if city_at_position is not None:
                tour.append(city_at_position)
            else:
                # Handle case where no city is assigned to position j
                # Find the most likely city (could be due to constraint violations)
                tour.append(0)  # Default assignment
        
        return tour
    
    def _calculate_tour_distance(self, tour):
        """Calculate total distance of a tour."""
        if not tour or len(tour) != self.n_cities:
            return float('inf')
        
        total_distance = 0
        n = len(tour)
        for i in range(n):
            total_distance += self.distance_matrix[tour[i]][tour[(i + 1) % n]]
        return total_distance
    
    def _is_valid_tour(self, tour):
        """Check if tour visits each city exactly once."""
        if not tour or len(tour) != self.n_cities:
            return False
        return (len(set(tour)) == self.n_cities and 
                all(city in tour for city in range(self.n_cities)))
    
    def _check_constraints(self, sample):
        """Check constraint violations in the solution."""
        n = self.n_cities
        violations = {'city_constraints': 0, 'position_constraints': 0}
        
        # Check city constraints (each city visited exactly once)
        for i in range(n):
            city_sum = sum(sample.get(f'x_{i}_{j}', 0) for j in range(n))
            if city_sum != 1:
                violations['city_constraints'] += abs(city_sum - 1)
        
        # Check position constraints (each position has exactly one city)
        for j in range(n):
            position_sum = sum(sample.get(f'x_{i}_{j}', 0) for i in range(n))
            if position_sum != 1:
                violations['position_constraints'] += abs(position_sum - 1)
        
        return violations
    
    def visualize_tour(self, tour, positions=None):
        """
        Visualize the TSP tour.
        
        Args:
            tour: List of city indices in order
            positions: Optional dict of city positions for plotting
        """
        if not tour:
            print("No valid tour to visualize")
            return
            
        if positions is None:
            # Generate random positions for visualization
            np.random.seed(42)
            positions = {i: (np.random.rand(), np.random.rand()) 
                        for i in range(self.n_cities)}
        
        plt.figure(figsize=(10, 8))
        
        # Plot cities
        for i, (x, y) in positions.items():
            plt.scatter(x, y, s=200, c='red', zorder=5)
            plt.annotate(self.city_names[i], (x, y), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot tour edges
        for i in range(len(tour)):
            start_city = tour[i]
            end_city = tour[(i + 1) % len(tour)]
            
            x1, y1 = positions[start_city]
            x2, y2 = positions[end_city]
            
            plt.arrow(x1, y1, x2-x1, y2-y1, 
                     head_width=0.02, head_length=0.02, 
                     fc='blue', ec='blue', alpha=0.7)
        
        plt.title(f'TSP Tour (Distance: {self._calculate_tour_distance(tour):.1f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_solution_details(self, result):
        """Print detailed solution information."""
        print("=" * 50)
        print("TSP SOLUTION DETAILS")
        print("=" * 50)
        print(f"Tour: {result['city_names']}")
        print(f"Distance: {result['distance']:.2f}")
        print(f"Energy: {result['energy']:.2f}")
        print(f"Valid Tour: {result['is_valid']}")
        print(f"Constraint Violations:")
        print(f"  - City constraints: {result['constraint_violations']['city_constraints']}")
        print(f"  - Position constraints: {result['constraint_violations']['position_constraints']}")
        print("=" * 50)

def create_example_tsp():
    """Create an example TSP instance."""
    # Example: 4-city TSP with symmetric distances
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    city_names = ['New York', 'Boston', 'Chicago', 'Miami']
    return TSPDimod(distance_matrix, city_names)

def create_larger_tsp():
    """Create a larger TSP instance for testing."""
    n = 6
    np.random.seed(42)
    
    # Generate random symmetric distance matrix
    distance_matrix = np.random.randint(10, 50, size=(n, n))
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(distance_matrix, 0)  # Zero diagonal
    
    city_names = [f'City_{i}' for i in range(n)]
    return TSPDimod(distance_matrix.astype(int), city_names)

def run_tsp_example():
    """Run a complete TSP example."""
    print("Creating TSP instance...")
    tsp = create_example_tsp()
    
    print("Formulating as BQM...")
    bqm = tsp.create_bqm()
    print(f"BQM created with {len(bqm.variables)} variables and {len(bqm.quadratic)} quadratic terms")
    
    print("\nSolving with classical simulated annealing...")
    classical_result = tsp.solve_classical(num_reads=100)
    tsp.print_solution_details(classical_result)
    
    # Uncomment to solve with D-Wave hybrid solver (requires D-Wave access)
    # print("\nSolving with D-Wave Leap Hybrid solver...")
    # hybrid_result = tsp.solve_hybrid()
    # tsp.print_solution_details(hybrid_result)
    
    # Uncomment to solve with quantum annealer (requires D-Wave access)
    # print("\nSolving with D-Wave quantum annealer...")
    # quantum_result = tsp.solve_quantum(num_reads=100)
    # tsp.print_solution_details(quantum_result)
    
    return tsp, classical_result