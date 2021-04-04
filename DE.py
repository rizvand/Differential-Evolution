import numpy as np
from numpy.random import default_rng
from testfunction import sphere, ackley
import copy

class Chromosome:
    def __init__(self, values, boundary):
        self.values = np.array(values)
        self.boundary = boundary
        self.dim = len(values)
    
    def fit(self, fitness_function):
        self.fitness = fitness_function(self.values)

    def mutation(self, x_p, x_q, x_r, F):
        v = x_p + F*(x_q - x_r)
        for i in range(self.dim):
            if v[i] > self.boundary[1]:
                v[i] = self.boundary[1]
            elif v[i] < self.boundary[0]:
                v[i] = self.boundary[0]
        return v

class DifferentialEvolution():
    def __init__(self, size, n_iter, fitness_function, boundary, crossover_prob, mutation_factor, save_history,  random_state):
        self.size = size
        self.n_iter = n_iter
        self.fitness_function = fitness_function
        self.boundary = boundary
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor
        self.save_history = save_history
        self.rng = default_rng(seed=random_state)
        np.random.seed(random_state)

    def simulate(self):
        # Initialize
        population = []
        for i in range(self.size):
            position = []
            for pos_bound in self.boundary:
                pos_ = np.random.uniform(pos_bound[0], pos_bound[1])
                position.append(pos_)
            initial_chromosome = Chromosome(position, self.boundary)
            initial_chromosome.fit(self.fitness_function)
            population.append(initial_chromosome)

        iteration_fitness = np.array([x.fitness for x in population])
        best_idx = np.argmin(iteration_fitness)
        iteration_best = population[best_idx].values, population[best_idx].fitness
        
        iteration = 0
#         print(f'iteration: {iteration} | best_position : {iteration_best[0]} | best_fit : {iteration_best[1]}')

        if self.save_history == True:
            self.population_history = []
            self.fitness_history = []
            self.best_history = []
            self.population_history.append([x.values for x in population.copy()])
            self.fitness_history.append(iteration_fitness.copy())
            self.best_history.append(iteration_best)
        
        # Optimization
        while iteration < self.n_iter:
            iteration += 1
            offspring_population = []
            for j in range(self.size):
                # Mutation
                idx1, idx2, idx3 = self.rng.choice(self.size, size=3, replace=False)
                v = population[idx1].values + self.mutation_factor*(population[idx2].values - population[idx3].values)
                
                # Crossover
                u = copy.deepcopy(population[j])
                for i in range(u.dim):
                    r = np.random.uniform()
                    
                # Apply Penalty
                    if r <= self.crossover_prob:
                        u.values[i] = v[i]
                        if u.values[i] > self.boundary[i][1]:
                            u.values[i] = self.boundary[i][1]
                        elif u.values[i] < self.boundary[i][0]:
                            u.values[i] = self.boundary[i][0]
                # Replacement
                u.fit(self.fitness_function)
                if u.fitness <= population[j].fitness:
                    offspring_population.append(u)
                else:
                    offspring_population.append(copy.deepcopy(population[j]))
            
            iteration_fitness = np.array([x.fitness for x in offspring_population])
            best_idx = np.argmin(iteration_fitness)
            iteration_best = offspring_population[best_idx].values, offspring_population[best_idx].fitness
            population = offspring_population.copy()
#             print(f'iteration: {iteration} | best_position : {iteration_best[0]} | best_fit : {iteration_best[1]}')
            
            if self.save_history == True:
                self.population_history.append([x.values for x in population])
                self.fitness_history.append(iteration_fitness)
                self.best_history.append(iteration_best)
                
            