from toolz import pipe

from leap_ec.individual import Individual
from leap_ec.decoder import Decoder
from leap_ec.problem import ScalarProblem
from leap_ec.context import context

import leap_ec.ops as ops
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec import util

from toolz import curry
from leap_ec.ops import iteriter_op
from typing import Iterator
from itertools import permutations  

from simulation import X
import numpy as np
import variables as vars
import random

class FBGDecoder(Decoder):

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        genome = np.array(genome)
        phenome = X(genome)
        return phenome

    def __repr__(self):
        return type(self).__name__ + "()"


class FBGProblem(ScalarProblem):

    def __init__(self, x):
        super().__init__(maximize=False)
        self.x = x

    def evaluate(self, phenome, *args, **kwargs):
        return np.sum((self.x - phenome)**2)


@curry
@iteriter_op
# yield all permutations of each genome
def swap(individual_itr: Iterator) -> Iterator:

    for individual in individual_itr:
        genome_perm = permutations(individual.genome)
        for genome in genome_perm:
            individual = individual.clone()
            individual.genome = list(genome)
            yield individual


@curry
@iteriter_op
# yield permutations of each genome stochastically
def stock_swap(individual_itr: Iterator, p_swap: int = 0.5) -> Iterator:

    for individual in individual_itr:
        genome_perm = permutations(individual.genome)
        # yield original deterministically
        next(genome_perm)
        yield  individual 
        for genome in genome_perm:
            # TODO replace p_swap with expected permutations to try
            if random.random() < p_swap: 
                individual = individual.clone()
                individual.genome = list(genome)
                yield individual

@curry
@iteriter_op
# differential step of differential evolution algorithm
def diff(individual_itr: Iterator, F: float = 0.8, p_diff:float = 0.9, n: int = 2) -> Iterator:
    # F: Differential weight $F /in [0,2]$
    # p_diff: crossover probability $p_diff /in [0,1]
    # n: number of genes 

    pop = list(individual_itr)
    N = len(pop)
    for i in range(N):        
        original = pop[i]
        individual = original.clone()

        id_diff = _select_samples(i, N)
        A,B,C = pop[id_diff]
        V = A + F*(B-C)

        id_swap = random.random(n) < p_diff
        # TODO add random index?
        individual.genome[id_swap] = V.genome[id_swap] 
        individual.evaluate()
        if original > individual:
            yield original
        else:
            yield individual

@curry
@iteriter_op
# differential step and swap step of swap differential evolution algorithm
def diff_swap(individual_itr: Iterator, F: float = 0.8, p_diff:float = 0.9, n: int = 2) -> Iterator:
    # F: Differential weight $F /in [0,2]$
    # p_diff: crossover probability $p_diff /in [0,1]
    # n: number of genes 

    pop = list(individual_itr)
    N = len(pop)
    for i in range(N):        
        original = pop[i]
        individual = original.clone()

        id_diff = _select_samples(i, N)
        A,B,C = [pop[i] for i in id_diff]
        V = A + F*(B-C)

        id_swap = np.random.rand(n) < p_diff
        # TODO add random index?
        individual.genome[id_swap] = V.genome[id_swap] 
        individual.evaluate()

        individual_swap = individual.clone()
        id_swap = random.sample(list(range(n)), k = 2)
        #aux = individual.genome[id_swap[1]]
        #individual_swap.genome[id_swap[1]] = individual_swap.genome[id_swap[0]] 
        #individual_swap.genome[id_swap[0]] = aux

        individual_swap.genome[id_swap] = individual_swap.genome[id_swap[::-1]]        
        individual_swap.evaluate()

        if individual_swap > individual:
            individual = individual_swap

        if original > individual: # This is adjusted with maximize to mean 'better than'
            yield original
        else:
            yield individual

def _select_samples(candidate, N):
    """
    obtain 3 random integers from range(N),
    without replacement. You can't have the original candidate.
    """
    idx = list(range(N))
    idx.remove(candidate)
    id_list = random.sample(idx, k=3)
    return id_list
 

class swap_differential_evolution():
    def __init__(self, pop_size=30, max_generation=100, bounds = vars.bounds, threshold = 1*vars.n, F=0.8, p_diff=0.9):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.bounds = bounds
        self.threshold = threshold
        self.F = F
        self.p_diff = p_diff


    def predict(self, x, verbose=False):
        if len(x.shape)==1:
            return self._predict(x, verbose)
        else:
            return np.stack([self._predict(k, verbose) for k in x])


    def _predict(self, x, verbose):
        parents = Individual.create_population(self.pop_size,
                                                initialize=create_real_vector(((self.bounds, ) * 2) ),
                                                decoder=FBGDecoder(),
                                                problem=FBGProblem(x))

        # Evaluate initial population
        parents = Individual.evaluate_population(parents)
        best_individual = ops.truncation_selection(parents, 1)[0]

        # print initial, random population
        if verbose:
            util.print_population(parents, generation=0)

        generation_counter = util.inc_generation(context=context)

        while generation_counter.generation() < self.max_generation:

        #util.print_population(parents, context['leap']['generation'])

            offspring = pipe(
                                iter(parents),
                                diff_swap(F=self.F, p_diff=self.p_diff),
                                ops.pool(size = -1)
                            )

            parents = offspring

            best_individual = ops.truncation_selection(parents, 1)[0]

            #if best_individual.fitness < self.threshold:
            #    return best_individual.genome

            generation_counter()  # increment to the next generation

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return np.array(best_individual.genome)
 

class genetic_algorithm():
    def __init__(self, pop_size=30, max_generation=100, bounds = vars.bounds, threshold = 1*vars.n, p_swap=0.3, std=0.01*vars.n):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.bounds = bounds
        self.threshold = threshold
        self.p_swap = p_swap
        self.std = std

    def predict(self, x, verbose=False):
        if len(x.shape)==1:
            return self._predict(x, verbose)
        else:
            return np.stack([self._predict(k, verbose) for k in x])


    def _predict(self, x, verbose):
        parents = Individual.create_population(self.pop_size,
                                                initialize=create_real_vector(((self.bounds, ) * 2) ),
                                                decoder=FBGDecoder(),
                                                problem=FBGProblem(x))

        # Evaluate initial population
        parents = Individual.evaluate_population(parents)
        best_individual = ops.truncation_selection(parents, 1)[0]

        # print initial, random population
        if verbose:
            util.print_population(parents, generation=0)

        generation_counter = util.inc_generation(context=context)

        while generation_counter.generation() < self.max_generation:

        #util.print_population(parents, context['leap']['generation'])

            offspring = pipe(
                                iter(parents),
                                #ops.cyclic_selection,
                                ops.clone,
                                ops.uniform_crossover(p_swap = self.p_swap),
                                mutate_gaussian(std = self.std, expected_num_mutations = 1, hard_bounds = self.bounds),
                                swap,
                                ops.evaluate,
                                ops.pool(size = -1),
                                ops.truncation_selection(size = self.pop_size, parents = parents)
                            )

            parents = offspring

            best_individual = ops.truncation_selection(parents, 1)[0]

            #if best_individual.fitness < self.threshold:
            #    return best_individual.genome

            generation_counter()  # increment to the next generation

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return np.array(best_individual.genome)