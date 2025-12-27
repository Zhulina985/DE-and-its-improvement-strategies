import numpy as np

class TSMS_Utils:
    @staticmethod
    def obl_initialization(func, bounds, pop_size, dim):
        """
        Opposite-Based Learning Initialization.
        Generates N random, N opposite, selects best N.
        """
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        range_width = ub - lb
        
        # 1. Random Population
        pop_rand = lb + range_width * np.random.rand(pop_size, dim)
        fit_rand = np.array([func(ind) for ind in pop_rand])
        
        # 2. Opposite Population
        # X* = a + b - X
        pop_opp = lb + ub - pop_rand
        # Check bounds
        pop_opp = np.clip(pop_opp, lb, ub)
        fit_opp = np.array([func(ind) for ind in pop_opp])
        
        # 3. Combine and Select
        pop_all = np.vstack((pop_rand, pop_opp))
        fit_all = np.concatenate((fit_rand, fit_opp))
        
        sorted_indices = np.argsort(fit_all)
        best_indices = sorted_indices[:pop_size]
        
        return pop_all[best_indices], fit_all[best_indices]

class Archive:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        
    def add(self, individuals):
        """Add individuals to archive. Randomly replace if full."""
        for ind in individuals:
            if len(self.data) < self.capacity:
                self.data.append(ind.copy())
            else:
                idx = np.random.randint(0, self.capacity)
                self.data[idx] = ind.copy()
                
    def get_candidates(self, population):
        """Return Union of Population and Archive."""
        if not self.data:
            return population
        return np.vstack((population, np.array(self.data)))

    def sample(self, count):
        """Sample individuals from archive."""
        if not self.data:
            return None
        indices = np.random.choice(len(self.data), count, replace=True)
        return np.array(self.data)[indices]

