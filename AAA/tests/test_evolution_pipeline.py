import unittest
from ikaggle.AAA.utils.data_loading import load_dataset  # updated import
from ikaggle.AAA.experiments.genetic_ops import custom_crossover, custom_mutation, setup_toolbox  # updated import    
from ikaggle.AAA.experiments.neurogenetic import main
import unittest
import random
from deap import base, creator, tools

class TestEvolutionPipeline(unittest.TestCase):
    def test_main_pipeline(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        
        # Check outputs are of expected types
        self.assertIsInstance(pop, list, "Population should be a list.")
        self.assertIsInstance(stats, dict, "Stats should be a dictionary.")
        self.assertIsInstance(hof, list, "Hall of Fame should be a list.")
    
    def test_population_size(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        
        # Check population size
        self.assertGreater(len(pop), 0, "Population should not be empty.")
    
    def test_stats_keys(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        
        # Check stats keys
        expected_keys = {'gen', 'nevals', 'avg', 'std', 'min', 'max'}
        self.assertTrue(expected_keys.issubset(stats.keys()), "Stats should contain the expected keys.")
    
    def test_hof_size(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        
        # Check Hall of Fame size
        self.assertGreater(len(hof), 0, "Hall of Fame should not be empty.")
    
    def test_individual_structure(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        
        # Check structure of individuals in population
        for individual in pop:
            self.assertIsInstance(individual, list, "Each individual should be a list.")
            self.assertGreater(len(individual), 0, "Each individual should not be empty.")
    
    def test_fitness_values(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        
        # Check fitness values of individuals in population
        for individual in pop:
            self.assertIsNotNone(individual.fitness.values, "Each individual should have a fitness value.")
            self.assertIsInstance(individual.fitness.values, tuple, "Fitness value should be a tuple.")
    
    def test_hof_individuals(self):
        # Load test data using data_loading.py
        X_full, y_full = load_dataset('testing')
        pop, stats, hof = main(X_full, y_full, verbose=0, test_mode=True)
        # Check individuals in Hall of Fame
        for individual in hof:
            self.assertIsInstance(individual, list, "Each individual in Hall of Fame should be a list.")
            self.assertGreater(len(individual), 0, "Each individual in Hall of Fame should not be empty.")
            self.assertIsNotNone(individual.fitness.values, "Each individual in Hall of Fame should have a fitness value.")
            self.assertIsInstance(individual.fitness.values, tuple, "Fitness value should be a tuple.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
