from math import factorial, nan, isclose
from toolz import pipe
from itertools import chain, tee

from leap_ec.individual import Individual
from leap_ec.decoder import Decoder
from leap_ec.problem import ScalarProblem
from leap_ec.global_vars import context

import leap_ec.ops as ops
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec import util

from toolz import curry
from typing import Iterator, List, Union
from itertools import permutations

from fbg_swdm.simulation import X, normalize, denormalize
import numpy as np
import fbg_swdm.variables as vars
import random

from sklearn.mixture import GaussianMixture

from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------- #
#                               Module Management                              #
# ---------------------------------------------------------------------------- #

def clone_module(module):
    """ Clones vars module to keep an static local copy """
    global vars
    vars = module.clone()

# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #

eps = np.finfo(float).eps # epsilon

# ---------------------------------------------------------------------------- #
#                                   Defaults                                   #
# ---------------------------------------------------------------------------- #

pop_size = [3, 5, 5]
swarms = [5, 15, 30]
dms_pso_defaults = [pop_size, swarms]

# -------------------------------------------------------------------------- #
#                                 Decorators                                 #
# -------------------------------------------------------------------------- #


def stack(func):
    """ Decorator for allowing predict method to accept multiple requests """
    def stacked(self, x, verbose=False):
        shape = x.shape
        if len(shape) == 1:
            return func(self, x, verbose)
        else:
            Y_hat = np.empty((shape[0], vars.Q))
            for i, k in enumerate(x):
                Y_hat[i] = func(self, k, verbose)
            return Y_hat
    return stacked

# -------------------------------------------------------------------------- #
#                               Auxiliary Classes                            #
# -------------------------------------------------------------------------- #

# -------------------------------- Decoders -------------------------------- #

class FBGDecoder(Decoder):

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        A_b = genome
        phenome = X(A_b, vars.λ, vars.A, vars.Δλ, vars.I, vars.Δn_dc, simulation=vars.simulation)
        return phenome

    def __repr__(self):
        return type(self).__name__ + "()"


class FBGDecoder_binary(Decoder):

    def __init__(self, B):
        super().__init__()
        self.B = B

    def decode(self, genome, *args, **kwargs):
        I, A_b = partial_decode(genome, self.B)
        phenome = X(A_b, vars.λ, vars.A, vars.Δλ, I, vars.Δn_dc, simulation=vars.simulation)
        return phenome

    def __repr__(self):
        return type(self).__name__ + "()"

# --------------------------------- Problems-------------------------------- #

class FBGProblem(ScalarProblem):

    def __init__(self, x=None):
        super().__init__(maximize=False)
        self.x = x
    
    def set(self, x):
        self.x = x

    def evaluate(self, phenome, *args, **kwargs):
        return np.sum(np.abs(self.x - phenome))

    def evaluate_multiple(self, phenomes, *args, **kwargs):
        return np.sum(np.abs(self.x[None, :] - phenomes), axis=-1)

# ---------------------------------------------------------------------------- #
#                                   Operators                                  #
# ---------------------------------------------------------------------------- #

# ------------------------------- Genome Swap ------------------------------ #

@curry
def swap(population: Iterator) -> Iterator:
    """ Yield all permutations of each genome"""
    for ind in population:
        genome_perm = permutations(ind.genome)
        for genome in genome_perm:
            ind = ind.clone()
            ind.genome = np.array(genome)
            yield ind


@curry
def stoc_swap(population: Iterator, p_swap: int = 0.5) -> Iterator:
    """ Yield permutations of each genome stochastically"""
    for ind in population:
        genome_perm = permutations(ind.genome)
        # always yield original
        yield next(genome_perm)
        for genome in genome_perm:
            # TODO: replace p_swap with expected permutations to try
            if random.random() < p_swap:
                ind = ind.clone()
                ind.genome = np.array(genome)
                yield ind


@curry
def single_swap(population: Iterator) -> Iterator:
    """ Yield  individual with one random permutation"""
    for ind in population:
        id_swap = random.sample(list(range(vars.Q)), k=2)
        genome_aux = ind.genome
        ind.genome[id_swap] = genome_aux[id_swap[::-1]]
        yield ind

# ------------------------- Distributed Estimation ------------------------- #


@curry
def update_sample(population, dist, sample_size, bounds):
    """ Update and then sample Gaussian distribution """
    dist.fit(normalize(y=get_genome(population)))

    new_pop_genome = denormalize(y=dist.sample(sample_size)[0])
    for genome in new_pop_genome:
        ind = population[0].clone()
        ind.genome = genome.clip(*bounds)
        yield ind

# ------------------------ Genetic Algorithm ------------------------ #


@curry
def sort_genome(population: Iterator, B: int) -> Iterator:
    """ Sort binary genome to place larger I first """
    for ind in population:
        I_float = next(partial_decode(ind.genome, B))  # decode I from genome
        indx = np.argsort(I_float)  # index of sorted I
        if vars.I[0]>vars.I[1]: # if larger first
            indx = indx[::-1]  # invert sort

        # split I from A_b chromosomes
        genome = np.split(ind.genome, vars.Q)
        I, A_b = np.split(np.array(genome), 2, axis=-1)
        # sort chromosomes according to indx
        I = np.take(I, indx, axis=0)
        A_b = np.take(A_b, indx, axis=0)
        # combine sorted chromosomes
        ind.genome = np.hstack((I, A_b)).flatten()

        yield ind


@curry 
def stoc_pair_selection(population: Iterator, problem) -> Iterator:
    """ Stocastic Pair selection. Samples pairs with probabilities proportional
        to the fitness. If the problem is minimization fitness is inverted.
    """

    fitness = np.array([ind.fitness for ind in population])
    if not problem.maximize:
        fitness = 1/fitness
    p = fitness/np.sum(fitness)
    # handle case of all zero fitness
    p = None if np.isnan(p).any() else p

    while True:
        pair = np.random.choice(population, size=2, replace=False, p=p)
        for ind in pair:
            yield ind


@curry
def one_point_crossover(population: Iterator, p=1.0) -> Iterator:
    """ Do crossover with probability p between pairs of individuals with respect to
        one randomly selected crossover point.

    :param population: Iterator of individuals
    :param p: Population of individuals
    :return: two recombined
    """
    
    def _one_point_crossover(child1, child2):

        x = np.random.randint(1, len(child1.genome))
        tmp = child1.genome[:x].copy()
        child2.genome[:x], child1.genome[:x] = child1.genome[:x], tmp

        return child1, child2

    for parent1, parent2 in zip(population, population):

        # Return the parents unmodified if we're not performing crossover
        if np.random.uniform() > p:
            yield parent1
            yield parent2
        else:  # Else cross them over
            child1, child2 = _one_point_crossover(parent1, parent2)
            yield child1
            yield child2

# ------------------------- Differential Evolution ------------------------- #


@curry
def diff(population: List, F: float = 0.8, p_diff: float = 0.9, Q: int = 2,
         bounds = None) -> Iterator:
    """ Differential step of differential evolution algorithm.
    :param population: List of individuals
    :param F: Differential weight $F /in [0,2]$
    :param p_diff: Crossover probability $p_diff /in [0,1]
    :param Q: Number of genes
    :param bounds: Bounds to clip the differential vector
    :return: yields individuals
    """

    N = len(population)
    for i in range(N):
        individual = population[i].clone()
        #diferential vector
        id_diff = select_samples(i, N)
        A, B, C = [population[i] for i in id_diff]
        V = individual.clone()
        V.genome = A.genome + F*(B.genome-C.genome)
        #trial vector
        id_swap = np.random.rand(Q) < p_diff
        # TODO add random index?
        individual.genome[id_swap] = V.genome[id_swap].clip(*bounds)
        yield individual


@curry
def pair_compare(population, parents):
    """ Yield max between each parent and its child """
    for ind1, ind2 in zip(population, parents):
        yield max(ind1, ind2)


# --------------------------------------------------------------------------- #
#                              Auxiliary Functions                            #
# --------------------------------------------------------------------------- #

# -------------------------- Binary Representation -------------------------- #

def bool2float(a, B):
    """ Transforms size 10 bool to float between 0 and 1 """
    to_float = np.exp2(np.arange(B))
    return np.dot(a, to_float)/np.sum(to_float)

def partial_decode(genome, B):
    """ return I and then A_b from binary genome """
    genome = np.split(genome, vars.Q)
    I_bin, A_b_bin = np.split(np.array(genome), 2, axis=-1)
    I_float = vars.I_min + bool2float(I_bin, B)*(vars.I_max-vars.I_min)
    yield I_float
    A_b_float = vars.λ0 + vars.portion*(2*vars.Δ*bool2float(A_b_bin, B) - vars.Δ)
    yield A_b_float

# -------------------------- Differential Evolution ------------------------- #

def select_samples(candidate, N):
    """ Obtain 3 random integers from [0 ,..., N-1]
    without replacement, excluding the candidate.
    """
    idx = list(range(N))
    idx.pop(candidate)  # remove candidate
    id_list = random.sample(idx, k=3)
    return id_list

# --------------------------------- General --------------------------------- #

def get_genome(population):
    """ Get genome from population """
    return np.array([individual.genome for individual in population])

def get_genome_pop_best(population):
    """ Get best genome of population """
    best_individual = max(population)
    return best_individual.genome

def divide_chunks(L: List, size: int)->List[List]:
    """Separate L into size-chunks"""
    return [L[i:i + size] for i in range(0, len(L), size)]

def flatten(L: List[List])->List:
    """ Flatten a list of lists to a list"""
    return list(chain(*L))

def clone(ind):
    """Clone individual but retain fitness"""
    cloned = ind.clone()
    cloned.fitness = ind.fitness
    return cloned

def clone_multiple(population):
    """Clone a population while rataining fitness"""
    return [clone(ind) for ind in population]


# --------------------------------------------------------------------------- #
#                         Genetic Algorithm Estimators                        #
# --------------------------------------------------------------------------- #

class GeneticAlgo():
    def __init__(self, pop_size=20, max_generation=500, Q=None,
                 threshold=0.1, patience=200, bounds=None):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.Q = vars.Q if Q is None else Q
        self.threshold = threshold
        self.patience = patience
        self.bounds = vars.bounds if bounds is None else bounds

        self.problem = FBGProblem()

    def predict(self, x, verbose=False):
            shape = x.shape
            if len(shape) == 1:
                return self.predict_func(x, verbose)
            elif vars.multiprocessing:
                max_workers = None if vars.multiprocessing is True else vars.multiprocessing
                with ProcessPoolExecutor(max_workers) as executor:
                    results = executor.map(self.predict_func, x)
                    Y_hat = np.empty((shape[0], vars.Q))
                    for i, y_hat in zip(range(shape[0]), results):
                        Y_hat[i] = y_hat
                return Y_hat
            else:
                Y_hat = np.empty((shape[0], vars.Q))
                for i, k in enumerate(x):
                    Y_hat[i] = self.predict_func(k, verbose)
                return Y_hat

    def predict_func(self, x, verbose=False):
        parents = self.init_population(x)
        best_individual = self.loop(parents, verbose)
        return best_individual.genome

    def init_population(self, x):
        bounds = ((self.bounds, ) * vars.Q)
        self.problem.set(x)
        parents = Individual.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=self.problem
                                   )
        parents = ops.grouped_evaluate(parents, problem=self.problem)
        return parents

    def loop(self, parents, verbose):

        # print initial, random population
        if verbose:
            util.print_population(parents, generation=0)

        # start generation counter
        generation_counter = util.inc_generation(context=context)

        static_counter = 0
        prev_best = max(parents)

        # main loop
        while generation_counter.generation() < self.max_generation:

            # print current generation
            if verbose == 2:
                util.print_population(parents, generation_counter.generation())

            offspring = self.pipeline(parents)

            best_individual = max(offspring)

            parents = offspring

            generation_counter()  # increment to the next generation

            # Check fitness plateau
            if best_individual > prev_best:
                prev_best = best_individual
                static_counter = 0
            elif prev_best.fitness < self.threshold:
                static_counter +=1

                if static_counter > self.patience:
                    if verbose:
                        util.print_population(parents,
                                            generation_counter.generation())

                    return best_individual

        if verbose:
            util.print_population(parents, generation_counter.generation())

        return best_individual


class genetic_algorithm_binary(GeneticAlgo):
    def __init__(self, pop_size=20, max_generation=500, Q=None,
                 threshold=0.1, p_crossover=1, p_mut=0.1, B=10, bounds=None, patience=200):
        super().__init__(pop_size, max_generation, Q, threshold, patience, bounds)
        self.p_swap = p_crossover
        self.B = B
        self.p_mut = p_mut

    def pipeline(self, parents):
        N = self.Q*2*self.B # number of chromosomes
        return pipe(
                    parents,
                    # stoc_pair_selection(problem=self.problem),
                    # ops.proportional_selection(offset='pop-min',
                    #                            key=lambda x: -x.fitness),
                    ops.sus_selection(offset='pop-min', key=lambda x: -x.fitness),
                    ops.clone,
                    one_point_crossover(p=self.p_swap),
                    mutate_bitflip(expected_num_mutations=self.p_mut*N),
                    sort_genome(B=self.B),
                    ops.pool(size=self.pop_size),
                    ops.grouped_evaluate(problem=self.problem),
                    ops.truncation_selection(size=self.pop_size,
                                             parents=parents)
                    )

    def predict_func(self, x, verbose=False):
        parents = self.init_population(x)
        best_individual = self.loop(parents, verbose)
        I, y_hat = partial_decode(best_individual.genome, self.B)
        if verbose:
            print('Best Individual:')
            print("I = ",I)
            print("y_hat = ",y_hat)
        return y_hat

    def init_population(self, x):
        self.problem.set(x)
        N = self.Q*2*self.B # number of chromosomes
        parents = Individual.create_population(
                                   self.pop_size,
                                   initialize=create_binary_sequence(N),
                                   decoder=FBGDecoder_binary(self.B),
                                   problem=self.problem
                                   )

        # Evaluate initial population
        parents = ops.grouped_evaluate(parents, problem=self.problem)
        return parents


class genetic_algorithm_real(GeneticAlgo):
    def __init__(self, pop_size=20, max_generation=500, Q=None,
                 threshold=0.1, patience=200, bounds=None, p_crossover=0.2,
                 std=0.01*vars.n, swap=False):
        super().__init__(pop_size, max_generation, Q, threshold, patience, bounds)
        self.p_crossover = p_crossover
        self.std = std
        self.swap = swap

    def pipeline(self, parents):
        return pipe(
                    parents,
                    # stoc_pair_selection(problem=self.problem),
                    # ops.proportional_selection(offset='pop-min',
                    #         key=lambda x: -x.fitness),
                    ops.sus_selection(offset='pop-min', key=lambda x: -x.fitness),
                    ops.clone,
                    ops.uniform_crossover(p_swap=self.p_crossover),
                    mutate_gaussian(std=self.std,
                                    expected_num_mutations=1,
                                    hard_bounds=self.bounds),
                    ops.pool(size=self.pop_size),
                    ops.grouped_evaluate(problem=self.problem),
                    ops.truncation_selection(size=self.pop_size,
                                                parents=parents)
                    )
    
    def loop(self, parents, verbose=False):
        best_individual = super().loop(parents, verbose)
        if self.swap:
            best_individual_swaps = list(swap([best_individual]))
            best_individual_swaps = ops.grouped_evaluate(best_individual_swaps, problem=self.problem)
            best_individual = max(best_individual_swaps)
        return best_individual

class DistributedEstimation(GeneticAlgo):
    def __init__(self, pop_size=100, max_generation=500, Q=None,
                 threshold=0.1, bounds=None, m=None, top_size=10, patience=200):
        super().__init__(pop_size, max_generation, Q, threshold, patience, bounds)
        self.m = factorial(vars.Q) if m is None else m
        self.top_size = top_size

    def pipeline(self, parents):
        return pipe(
                    parents,
                    update_sample(dist=self.dist,
                                    sample_size=self.pop_size,
                                    bounds=self.bounds),
                    ops.pool(size=self.pop_size),
                    ops.grouped_evaluate(problem=self.problem),
                    ops.truncation_selection(size=self.top_size,
                                                parents=parents)
                    )

    def predict_func(self, x, verbose=False):
        bounds = ((self.bounds, ) * self.Q)  # repeat bounds for each FBG
        self.problem.set(x)
        parents = Individual.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=self.problem
                                   )

        # Evaluate initial population
        parents = ops.grouped_evaluate(parents, problem=self.problem)
        parents = ops.truncation_selection(parents, size=self.top_size)

        self.dist = GaussianMixture(warm_start=False, n_components=self.m)

        best_individual = self.loop(parents, verbose)

        return best_individual.genome


class swap_differential_evolution(GeneticAlgo):
    def __init__(self, pop_size=30, max_generation=100, Q=None,
                 threshold=0.1, bounds=None, F=0.8, p_diff=0.9, patience=200):
        super().__init__(pop_size, max_generation, Q, threshold, patience, bounds)
        self.F = F
        self.p_diff = p_diff

    def pipeline(self, parents):
        diff_offspring = pipe(parents,
                              diff(F=self.F, p_diff=self.p_diff, Q=self.Q,
                                   bounds=self.bounds),
                              ops.pool(size=self.pop_size),
                              ops.grouped_evaluate(problem=self.problem)
                              )
        swap_offspring = pipe(iter(diff_offspring),
                              ops.clone,
                              single_swap,
                              ops.pool(size=self.pop_size),
                              ops.grouped_evaluate(problem=self.problem),
                              pair_compare(parents=diff_offspring)
                              )
        offspring = pipe(swap_offspring,
                         pair_compare(parents=parents),
                         ops.pool(size=self.pop_size)
                         )
        return offspring

class particle_swarm_optimization(GeneticAlgo):
    def __init__(self, pop_size=50, max_generation=1000, Q=None,
                 threshold=0.1, bounds=None, w=0.6, pa=2, ga=2,
                 lr=0.1, vel_init='gaussian', patience=200):
        super().__init__(pop_size, max_generation, Q, threshold, patience, bounds)
        self.w = w  # previous velocity weight constant
        self.pa = pa  # population acceleration
        self.ga = ga  # group acceleration
        self.lr = lr  # learning rate
        self.velocities = None
        self.vel_init = vel_init

    def pipeline(self, parents):
        offspring = pipe(parents,
                    self.update_position(velocities=self.velocities,
                                         lr=self.lr, w=self.w, pa=self.pa,
                                         ga=self.ga, bounds=self.bounds),
                    ops.pool(size=self.pop_size),
                    ops.grouped_evaluate(problem=self.problem),
                    self.update_best
                    )
        return offspring

    def get_size(self):
        """ get size for velocities initialization"""
        return 1, self.pop_size, vars.Q
    
    def init_velocities(self):
        if self.vel_init == 'gaussian': # X ~ N(0, diff(bounds))
            velocities = np.random.randn(*self.get_size())
            velocities = velocities*np.diff(self.bounds)
        elif self.vel_init == 'uniform': # X ~ Uniform(-diff(bounds), diff(bounds))
            velocities = np.random.rand(*self.get_size())
            velocities = (2*velocities-1)*np.diff(self.bounds)
        elif self.vel_init == 'zeros': # X ~ 0
            velocities = np.zeros(self.get_size())
        else:
            raise ValueError('vel_init must be one of {"gaussian", "uniform", "zeros"}')
        return velocities

    def predict_func(self, x, verbose=False):
        parents = self.init_population(x)

        self.current_subpopulation = 0
        self.subpopulation_ind_bsf = [clone_multiple(parents)]
        self.subpopulation_bsf = [max(self.subpopulation_ind_bsf[0])]

        self.velocities = self.init_velocities()

        best_individual = self.loop(parents, verbose)

        return best_individual.genome
    
    def update_velocities(self, parents, velocities, w, pa, ga):

        N = len(parents)
        r = np.random.rand(N, vars.Q)
        q = np.random.rand(N, vars.Q)

        parents_genome = get_genome(parents)
        i = self.current_subpopulation
        subpop_ind_bsf = self.subpopulation_ind_bsf[i]
        individual_bsf_genome = get_genome(subpop_ind_bsf)
        subpop_bsf = self.subpopulation_bsf[i]

        # previous velocity component
        new_velocities = w*velocities
        # attraction to previous particle best
        new_velocities += pa*r*(individual_bsf_genome - parents_genome)
        # attraction to previous subpopulation best
        new_velocities += ga*q*(subpop_bsf.genome - parents_genome)

        return new_velocities
        
    @curry
    def update_position(self, subpopulation, velocities, lr, w, pa, ga, bounds):
        i = self.current_subpopulation
        velocities[i] = self.update_velocities(subpopulation, velocities[i], w, pa, ga)

        for ind, v in zip(subpopulation, velocities[i]):
            ind.genome += v * lr
            ind.genome = ind.genome.clip(*bounds)
            yield ind

    def update_best(self, subpopulation):
        i = self.current_subpopulation
        # update individual best instance so far
        ind_bsf = self.subpopulation_ind_bsf[i]
        for j, ind in enumerate(subpopulation):
            if ind > ind_bsf[j]:
                ind_bsf[j] = clone(ind)
        # update best of subpopulation
        self.subpopulation_bsf[i]= max(ind_bsf)
        return subpopulation

    def random_migrate(self, subpopulations, pop_size, swarms):
        indeces = list(range(pop_size*swarms))
        random.shuffle(indeces)
        indeces = divide_chunks(indeces, size=pop_size)

        population = flatten(subpopulations)
        subpopulations = [[population[idx] for idx in subpop_idx] for subpop_idx in indeces]

        population_ind_bsf = flatten(self.subpopulation_ind_bsf)
        for i, subpop_idx in enumerate(indeces):
            self.subpopulation_ind_bsf[i] = [population_ind_bsf[idx] for idx in subpop_idx]

        return subpopulations


class dynamic_multi_swarm_particle_swarm_optimization(particle_swarm_optimization):
    def __init__(self, pop_size=None, max_generation=1000, Q=None,
                 threshold=0.1, bounds=None, w=0.6, pa=2, ga=2, lr=0.1,
                 swarms=None, migration_gap=5, vel_init='gaussian', patience=200):
        pop_size = dms_pso_defaults[0][vars.Q-1] if pop_size is None else pop_size
        swarms = dms_pso_defaults[1][vars.Q-1] if swarms is None else swarms
        super().__init__(pop_size, max_generation, Q,
                 threshold, bounds, w, pa, ga,
                 lr, vel_init, patience)
        self.swarms = swarms
        self.migration_gap = migration_gap

    def get_size(self):
        """ get size for velocities initialization"""
        return self.swarms, self.pop_size, vars.Q

    def predict_func(self, x, verbose=False):
        subpopulations = self.init_population(x)
        
        # best historical instance for each individual in each subpopulation
        self.subpopulation_ind_bsf = [clone_multiple(subpop)
                                                    for subpop in subpopulations]
        # best historical instance for each subpopulation
        self.subpopulation_bsf = list(map(max, self.subpopulation_ind_bsf))

        self.velocities = self.init_velocities()

        best_individual = self.loop(subpopulations, verbose)

        return best_individual.genome

    def init_population(self, x):
        bounds = ((self.bounds, ) * vars.Q)
        self.problem.set(x)
        subpopulations = [Individual.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=self.problem,
                                   ) for _ in range(self.swarms)]

        # Evaluate initial subpopulations
        ops.grouped_evaluate(flatten(subpopulations), problem=self.problem)
        return subpopulations

    def loop(self, subpopulations, verbose):
        # print initial, random population
        if verbose:
            for population in subpopulations:
                util.print_population(population, generation=0)

        generation_counter = util.inc_generation(context=context)

        static_counter = 0
        bsf_prev = clone(max(self.subpopulation_bsf))

        while generation_counter.generation() < self.max_generation:

            for i, subpop in enumerate(subpopulations):

                self.current_subpopulation = i
                subpopulations[i] = self.pipeline(subpop)

                if verbose==2:
                    util.print_population(subpop, generation_counter.generation())

            generation_counter()  # increment to the next generation

            population_bsf = clone(max(self.subpopulation_bsf))

            if verbose:
                print('Generation: ', generation_counter.generation())
                print('Best so far: ', population_bsf)
                print('Previous best so far: ', bsf_prev)

            # Check fitness plateau
            if population_bsf > bsf_prev:
                bsf_prev = population_bsf
                static_counter = 0
            else:
                static_counter += 1
            if static_counter > self.patience:
                break

            # Migration
            if generation_counter.generation() % self.migration_gap == 0:
                subpopulations = self.random_migrate(subpopulations, self.pop_size,
                                                self.swarms)

        # swap step
        population = list(swap([population_bsf]))
        population = ops.grouped_evaluate(population, problem=self.problem)

        if verbose:
            util.print_population(population, generation_counter.generation())

        return max(population)
