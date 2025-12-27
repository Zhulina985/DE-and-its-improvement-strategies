"""
mLSHADE-SPACMA Strategy Adapter
Implements the three main strategies from mLSHADE-SPACMA:
1. Precise elimination and generation mechanism
2. Modified mutation strategy with semi-parameter adaptation and RSP
3. Elite archiving mechanism
"""

import numpy as np
from scipy.stats import cauchy


class MLSHADESPACMAAdapter:
    """
    Adapter for mLSHADE-SPACMA strategies that can be integrated into
    CPO, FGO, FLOOD, and WOO algorithms.
    """
    
    def __init__(self, func, dim, pop_size, max_generations, bounds, rho=0.1):
        """
        Initialize the adapter.
        
        Parameters:
        -----------
        func : callable
            Objective function
        dim : int
            Problem dimension
        pop_size : int
            Population size
        max_generations : int
            Maximum number of generations
        bounds : numpy array
            Bounds for each dimension [dim, 2]
        rho : float
            Percentage of population to eliminate (default: 0.1)
        """
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.bounds = bounds
        self.rho = rho  # Elimination percentage
        
        # Historical memory for F parameter (Cauchy distribution)
        self.H = 6  # Memory size
        self.M_F = np.ones(self.H) * 0.5  # Initialize to 0.5
        self.S_F = []  # Success values for F
        
        # External archive
        self.archive = []
        self.archive_size = int(2.6 * pop_size)  # Typical archive size
        
        # Function evaluation counter
        self.nfes = 0
        self.max_fes = max_generations * pop_size  # Approximate
        
    def increment_nfes(self, count):
        """Increment function evaluation counter."""
        self.nfes += count
    
    def get_fes_ratio(self):
        """Get current FES ratio."""
        return self.nfes / self.max_fes if self.max_fes > 0 else 0.0
    
    def precise_elimination_and_generation(self, population, fitness):
        """
        Precise elimination and generation mechanism (Equation 35, 36).
        Active only in the first half of evaluations.
        
        Parameters:
        -----------
        population : numpy array
            Current population [pop_size, dim]
        fitness : numpy array
            Current fitness values [pop_size]
            
        Returns:
        --------
        new_population : numpy array
            Updated population after elimination and generation
        new_fitness : numpy array
            Updated fitness values
        """
        fes_ratio = self.get_fes_ratio()
        
        # Only active in first half
        if fes_ratio >= 0.5:
            return population, fitness
        
        # Sort by fitness (ascending for minimization)
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        sorted_fit = fitness[sorted_indices]
        
        # Calculate number of individuals to eliminate (Equation 35)
        PE_m = int(np.ceil(self.rho * len(population)))
        
        if PE_m <= 0 or PE_m >= len(population):
            return population, fitness
        
        # Eliminate worst individuals
        remaining_pop = sorted_pop[:-PE_m]
        remaining_fit = sorted_fit[:-PE_m]
        
        # Generate new individuals using Equation (36)
        # xi = xbest1 + rand * (xbest1 - xbest2)
        new_individuals = []
        new_fitness = []
        
        xbest1 = sorted_pop[0]  # Best individual
        xbest2 = sorted_pop[1] if len(sorted_pop) > 1 else sorted_pop[0]  # Second best
        
        for _ in range(PE_m):
            rand = np.random.rand(self.dim)
            xi = xbest1 + rand * (xbest1 - xbest2)
            
            # Apply bounds
            lb, ub = self.bounds[:, 0], self.bounds[:, 1]
            xi = np.clip(xi, lb, ub)
            
            new_individuals.append(xi)
        
        # Combine remaining and new individuals
        if new_individuals:
            new_pop = np.vstack([remaining_pop, np.array(new_individuals)])
            # Evaluate new individuals
            new_fit = np.array([self.func(ind) for ind in new_individuals])
            combined_fit = np.concatenate([remaining_fit, new_fit])
            self.increment_nfes(len(new_individuals))
            
            return new_pop, combined_fit
        
        return remaining_pop, remaining_fit
    
    def get_adaptive_F(self, r_i):
        """
        Get adaptive F parameter (Equation 37).
        First half: F_i = 0.5 + 0.1 * rand
        Second half: F_i = randc(M_F,r_i, 0.1)
        
        Parameters:
        -----------
        r_i : int
            Random index from [1, H]
            
        Returns:
        --------
        F_i : float
            Scaling factor
        """
        fes_ratio = self.get_fes_ratio()
        
        if fes_ratio < 0.5:
            # First half: fixed range
            F_i = 0.5 + 0.1 * np.random.rand()
        else:
            # Second half: Cauchy distribution
            M_F_r = self.M_F[r_i - 1]  # r_i is 1-indexed
            F_i = cauchy.rvs(loc=M_F_r, scale=0.1)
            
            # Regenerate if out of bounds [0, 1]
            while F_i <= 0 or F_i >= 1:
                F_i = cauchy.rvs(loc=M_F_r, scale=0.1)
        
        return F_i
    
    def update_archive(self, parent, offspring, parent_fit, offspring_fit):
        """
        Update external archive with elite archiving mechanism.
        If parent is better, keep it; otherwise, add parent to archive.
        When archive is full, remove worst individuals (elite archiving).
        
        Parameters:
        -----------
        parent : numpy array
            Parent individual
        offspring : numpy array
            Offspring individual
        parent_fit : float
            Parent fitness
        offspring_fit : float
            Offspring fitness
        """
        # If parent is better, it stays in population
        # Otherwise, add parent to archive
        if offspring_fit < parent_fit:
            # Offspring is better, add parent to archive
            self.archive.append((parent.copy(), parent_fit))
        else:
            # Parent is better, add offspring to archive (if we want to keep it)
            # Actually, in LSHADE, we add the parent to archive if offspring is better
            pass
        
        # Elite archiving: remove worst if archive is full
        if len(self.archive) > self.archive_size:
            # Sort by fitness (ascending for minimization)
            self.archive.sort(key=lambda x: x[1])
            # Remove worst individuals
            self.archive = self.archive[:self.archive_size]
    
    def get_archive_individual(self):
        """
        Get a random individual from archive.
        
        Returns:
        --------
        individual : numpy array or None
            Random individual from archive, or None if archive is empty
        """
        if len(self.archive) == 0:
            return None
        
        idx = np.random.randint(0, len(self.archive))
        return self.archive[idx][0]
    
    def update_memory_F(self, successful_F_values):
        """
        Update historical memory M_F using successful F values.
        Only update in second half of evaluations.
        
        Parameters:
        -----------
        successful_F_values : list
            List of successful F values from current generation
        """
        fes_ratio = self.get_fes_ratio()
        
        # Only update in second half
        if fes_ratio < 0.5 or len(successful_F_values) == 0:
            return
        
        # Store successful values
        self.S_F.extend(successful_F_values)
        
        # Update memory using weighted Lehmer mean (similar to LSHADE)
        if len(self.S_F) > 0:
            # Use simple mean for simplicity (can be enhanced with weighted mean)
            mean_F = np.mean(self.S_F)
            
            # Update current memory cell (cycling through H cells)
            k = (self.nfes // (self.max_fes // self.H)) % self.H
            self.M_F[k] = mean_F
            
            # Clear success values for next generation
            self.S_F = []
    
    def rank_based_selection(self, population, fitness, num_select=1):
        """
        Rank-based selective pressure (RSP) selection.
        Higher rank (better fitness) has higher selection probability.
        
        Parameters:
        -----------
        population : numpy array
            Population [pop_size, dim]
        fitness : numpy array
            Fitness values [pop_size]
        num_select : int
            Number of individuals to select
            
        Returns:
        --------
        selected : numpy array
            Selected individuals [num_select, dim]
        """
        # Sort by fitness (ascending)
        sorted_indices = np.argsort(fitness)
        
        # Calculate rank-based probabilities
        pop_size = len(population)
        ranks = np.arange(1, pop_size + 1)  # Rank 1 is best
        rank_sum = np.sum(ranks)
        
        # Higher rank (lower number) should have higher probability
        # So we invert: probability proportional to (pop_size - rank + 1)
        probabilities = (pop_size - ranks + 1) / rank_sum
        
        # Select based on probabilities
        selected_indices = np.random.choice(
            pop_size, 
            size=num_select, 
            replace=True, 
            p=probabilities
        )
        
        return population[sorted_indices[selected_indices]]
    
    def apply_mutation_strategy(self, x_i, x_pbest, population, fitness, F_i):
        """
        Apply mutation strategy: v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_pr1 - x_r2)
        where x_pr1 is selected using RSP, x_r2 is from archive or population.
        
        Parameters:
        -----------
        x_i : numpy array
            Current individual
        x_pbest : numpy array
            pbest individual (from top p% of population)
        population : numpy array
            Current population
        fitness : numpy array
            Current fitness values
        F_i : float
            Scaling factor
            
        Returns:
        --------
        v_i : numpy array
            Mutant vector
        """
        # Select x_pr1 using RSP
        x_pr1 = self.rank_based_selection(population, fitness, num_select=1)[0]
        
        # Select x_r2 from archive or population
        x_r2_archive = self.get_archive_individual()
        if x_r2_archive is not None and np.random.rand() < 0.5:
            x_r2 = x_r2_archive
        else:
            # Random from population
            idx = np.random.randint(0, len(population))
            x_r2 = population[idx]
        
        # Apply mutation (Equation 38)
        v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_pr1 - x_r2)
        
        # Apply bounds
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        v_i = np.clip(v_i, lb, ub)
        
        return v_i

