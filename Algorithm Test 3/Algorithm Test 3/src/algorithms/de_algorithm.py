import numpy as np

class DifferentialEvolution:
    def __init__(self, func, bounds, pop_size=50, mutation_factor=0.5, crossover_prob=0.7, max_generations=1000):
        """
        Differential Evolution (DE) Algorithm implementation.
        
        Args:
            func (callable): The objective function to minimize.
            bounds (list of tuple): List of (min, max) for each dimension.
            pop_size (int): Population size.
            mutation_factor (float): Differential weight (F), typically [0, 2].
            crossover_prob (float): Crossover probability (CR), [0, 1].
            max_generations (int): Maximum number of generations/iterations.
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.F = mutation_factor
        self.CR = crossover_prob
        self.max_generations = max_generations
        self.dim = len(bounds)
        
    def optimize(self):
        """
        Runs the DE algorithm.
        
        Returns:
            tuple: (best_solution, best_fitness, fitness_history)
        """
        # 1. Initialization
        # Generate population within bounds
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]
        population = min_b + (max_b - min_b) * np.random.rand(self.pop_size, self.dim)
        
        # Evaluate initial population
        fitness = np.array([self.func(ind) for ind in population])
        
        # Track best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        
        # 2. Main Loop
        for gen in range(self.max_generations):
            for i in range(self.pop_size):
                # 2.1 Mutation (DE/rand/1)
                # Select 3 distinct random indices other than i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
                
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                
                # Check bounds (clamping)
                mutant = np.clip(mutant, min_b, max_b)
                
                # 2.2 Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # 2.3 Selection
                trial_fitness = self.func(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update global best
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
            
            fitness_history.append(best_fitness)
            
            # Optional: Print progress
            if (gen + 1) % 100 == 0:
                # print(f"Generation {gen+1}/{self.max_generations}, Best Fitness: {best_fitness:.6f}")
                pass
                
        return best_solution, best_fitness, fitness_history

