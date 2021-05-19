from math import nan, isclose
from copy import deepcopy
from toolz import pipe, juxt, compose, identity
from itertools import chain

from leap_ec.individual import Individual
from leap_ec.decoder import Decoder
from leap_ec.problem import ScalarProblem
from leap_ec.context import context

import leap_ec.ops as ops
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec import util

from toolz import curry
from leap_ec.ops import listiter_op
from typing import Iterator, List
from itertools import permutations

from fbg_swdm.simulation import X
import numpy as np
import fbg_swdm.variables as vars
import random

from sklearn.mixture import GaussianMixture

# -------------------------------------------------------------------------- #
#                                 Decorators                                 #
# -------------------------------------------------------------------------- #


def stack(func):
    '''Decorator for allowing predict to accept multiple requests'''
    def stacked(self, x, verbose=False):
        shape = x.shape
        if len(shape) == 1:
            return func(self, x, verbose)
        else:
            Y_hat = np.empty((shape[0], vars.FBGN))
            for i, k in enumerate(x):
                Y_hat[i] = func(self, k, verbose)
            return Y_hat
    return stacked

# -------------------------------------------------------------------------- #
#                               Auxiliary Classes                            #
# -------------------------------------------------------------------------- #

# ------------------------------- Individuals ------------------------------ #


class Individual_numpy(Individual):
    '''Individual with numpy genome'''
    def __init__(self, genome, decoder=None, problem=None):
        genome = np.array(genome)
        super().__init__(genome, decoder, problem)


class Individual_simple(Individual):
    '''Simplified Individual that serves as a container for genome and fitness
    pairs
    '''
    def __init__(self, genome, fitness):
        self.genome = genome
        if fitness:
            self.fitness = fitness
        else:
            self.fitness = nan

    def clone(self):
        new_genome = deepcopy(self.genome)
        cloned = Individual_simple(new_genome, self.fitness)
        return cloned


class Individual_hist(Individual_numpy):
    ''' Individual with memory of best state genome and fitness
    '''

    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)
        self.best = Individual_simple(genome, nan)

    def evaluate(self):
        self.fitness = super().evaluate_imp()
        # update individual best
        if self > self.best:
            self.best = Individual_simple.clone(self)

        # update subpopulation best
        i = context['leap']['current_subpopulation']
        population_best = context['leap']['population_best']
        if self > population_best[i]:
            population_best[i] = Individual_simple.clone(self)

        return self.fitness

# -------------------------------- Decoders -------------------------------- #


class FBGDecoder(Decoder):

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        phenome = X(genome)
        return phenome

    def __repr__(self):
        return type(self).__name__ + "()"


class FBGDecoder_binary(Decoder):

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        I, A_b = partial_decode(genome)
        phenome = X(A_b, I)
        return phenome

    def __repr__(self):
        return type(self).__name__ + "()"

# --------------------------------- Problems-------------------------------- #


class FBGProblem(ScalarProblem):

    def __init__(self, x):
        super().__init__(maximize=False)
        self.x = x

    def evaluate(self, phenome, *args, **kwargs):
        return np.sum(np.abs(self.x - phenome))

# -------------------------------------------------------------------------- #
#                                  Operators                                 #
# -------------------------------------------------------------------------- #


@curry
def to_np_array(parents):
    '''Convert genomes to numpy representation'''
    for ind in parents:
        ind.genome = np.array(ind.genome)
        yield ind

# ------------------------------- Genome Swap ------------------------------ #


@curry
# yield all permutations of each genome
def swap(population: Iterator) -> Iterator:

    for individual in population:
        genome_perm = permutations(individual.genome)
        for genome in genome_perm:
            individual = individual.clone()
            individual.genome = np.array(genome)
            yield individual


@curry
# yield permutations of each genome stochastically
def stoc_swap(individual_itr: Iterator, p_swap: int = 0.5) -> Iterator:

    for individual in individual_itr:
        genome_perm = permutations(individual.genome)
        # always yield original
        next(genome_perm)
        yield individual
        for genome in genome_perm:
            # TODO replace p_swap with expected permutations to try
            if random.random() < p_swap:
                individual = individual.clone()
                individual.genome = list(genome)
                yield individual


@curry
# yield one permutation of each genome stochastically
def single_swap(individual_itr: Iterator, p_swap: int = 1) -> Iterator:

    for individual in individual_itr:
        yield individual
        if random.random() < p_swap:
            individual = individual.clone()
            id_swap = random.sample(list(range(vars.FBGN)), k=2)
            individual.genome[id_swap] = individual.genome[id_swap[::-1]]
            yield individual

# ------------------------- Distributed Estimation ------------------------- #


@curry
def update_sample(population, dist, sample_size):
    '''Update and then sample Gaussian distribution '''
    dist.fit(get_genome(population))

    # if np.any(dist.weights_ < 1e-15):
    #     dist.weights_[0] = 1

    new_pop_genome = dist.sample(sample_size)[0]
    for genome in new_pop_genome:
        ind = population[0].clone()
        ind.genome = genome
        yield ind

# ------------------------ Genetic Algorithm Binary ------------------------ #


@curry
def sort_genome(population: Iterator) -> Iterator:
    """ Sort genome to place larger I first """
    for ind in population:
        I_bin = next(partial_decode(ind.genome))  # decode I from genome
        indx = np.argsort(I_bin)  # index of sorted I
        indx = indx[::-1]  # larger first

        # split I from A_b chromosomes
        I_float, A_b_float = np.split(ind.genome, 2)
        # split into chromosomes
        I_float = np.array(np.split(I_float, vars.FBGN))
        A_b_float = np.array(np.split(A_b_float, vars.FBGN))
        # sort chromosomes according to indx
        I_float = np.take(I_float, indx, axis=0)
        A_b_float = np.take(A_b_float, indx, axis=0)
        # combine sorted chromosomes
        I_float = np.concatenate(I_float)
        A_b_float = np.concatenate(A_b_float)
        ind.genome = np.concatenate((I_float, A_b_float))

        yield ind


@curry
def stoc_pair_selection(population: Iterator) -> Iterator:
    """
    Stocastic selection.
    Samples pairs with probabilities proportional to fitness
    """

    fitness = [ind.fitness for ind in population]
    p = fitness/np.sum(fitness)

    while True:
        pair = np.random.choice(population, size=2, replace=False, p=p)
        for ind in pair:
            yield ind


@curry
def one_point_crossover(next_individual: Iterator,
                        p=1.0) -> Iterator:
    """ Do crossover between individuals between one crossover points.

    :param next_individual: where we get the next individual from the pipeline
    :return: two recombined
    """

    def _one_point_crossover(child1, child2):

        x = np.random.randint(1, len(child1.genome))
        tmp = child1.genome[:x].copy()
        child2.genome[:x], child1.genome[:x] = child1.genome[:x], tmp

        return child1, child2

    for parent1, parent2 in zip(next_individual, next_individual):

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
@listiter_op
# differential step of differential evolution algorithm
def diff(population: List, F: float = 0.8, p_diff: float = 0.9, FBGN: int = 2
         ) -> Iterator:
    # F: Differential weight $F /in [0,2]$
    # p_diff: crossover probability $p_diff /in [0,1]
    # n: number of genes

    N = len(population)
    for i in range(N):
        individual = population[i]

        id_diff = select_samples(i, N)
        A, B, C = [population[i] for i in id_diff]
        V = individual.clone()
        V.genome = A.genome + F*(B.genome-C.genome)

        id_swap = np.random.rand(FBGN) < p_diff
        # TODO add random index?
        individual.genome[id_swap] = V.genome[id_swap]
        yield individual


@curry
def one_on_one_compare(populations, parents=None):
    '''Compare each parent with its child'''
    if parents:
        populations = (populations, parents)

    for ind1, ind2 in zip(*populations):
        yield max(ind1, ind2)

# ----------------------- Particle Swarm Optimization ---------------------- #


@curry
def update_position(parents, velocities, lr, w, pa, ga):
    i = context['leap']['current_subpopulation']
    velocities[i] = update_velocities(parents, velocities[i], w, pa, ga)

    for ind, v in zip(parents, velocities[i]):
        ind.genome += v * lr
        yield ind


@curry
def clip(population, bounds):
    for individual in population:
        individual.genome = individual.genome.clip(*bounds)
        yield individual


def random_migrate(subpopulations, pop_size, swarms):
    # flatten
    population = list(chain(*subpopulations))
    # shuffle
    random.shuffle(population)
    # deflatten
    subpopulations = [population[i:i+pop_size]
                      for i in range(0, swarms*pop_size, pop_size)]

    return subpopulations


# --------------------------------------------------------------------------- #
#                              Auxiliary Functions                            #
# --------------------------------------------------------------------------- #

# -------------------------- Binary Representation -------------------------- #

to_float = np.exp2(np.arange(10))  # transform array
to_float = to_float/np.sum(to_float)  # normalize


def bool2float(a):
    # Transforms size 10 bool to float between 0 and 1
    return np.dot(a, to_float)


def partial_decode(genome):
    """ return I and then A_b from binary genome """
    I_bin, A_b_bin = np.split(genome, 2)
    I_bin = np.stack(np.split(I_bin, vars.FBGN))
    I_float = bool2float(I_bin)
    yield I_float
    A_b_bin = np.stack(np.split(A_b_bin, vars.FBGN))
    A_b_float = vars.A0 - vars.D + 2*vars.D*bool2float(A_b_bin)
    yield A_b_float

# -------------------------- Differential Evolution ------------------------- #


def select_samples(candidate, N):
    """
    obtain 3 random integers from range(N),
    without replacement. You can't have the original candidate.
    """
    idx = list(range(N))
    idx.pop(candidate)  # remove candidate
    id_list = random.sample(idx, k=3)
    return id_list

# ------------------------ Particle Swarm Optimization ---------------------- #


def update_velocities(parents, velocities, w, pa, ga):

    N = len(parents)
    r = np.random.rand(N, vars.FBGN)
    q = np.random.rand(N, vars.FBGN)

    parents_genome = get_genome(parents)
    particle_best_genome = get_genome_best(parents)
    i = context['leap']['current_subpopulation']
    subpop_best = context['leap']['population_best'][i]

    # previous velocity component
    new_velocities = w*velocities
    # attraction to previous particle best
    new_velocities += pa*r*(particle_best_genome - parents_genome)
    # attraction to previous subpopulation best
    new_velocities += ga*q*(subpop_best.genome - parents_genome)

    return new_velocities

# --------------------------------- General --------------------------------- #


def get_genome(population):
    ''' Get genome from population
    '''
    return np.array([individual.genome for individual in population])


def get_genome_best(population):
    ''' Get best genome of each Individual_hist of a population
    '''
    return np.array([individual.best.genome for individual in population])


def get_genome_pop_best(population):
    '''Get best genome of population
    '''
    best_individual = max(population)
    return best_individual.genome


# --------------------------------------------------------------------------- #
#                         Genetic Algorithm Estimators                        #
# --------------------------------------------------------------------------- #

class GeneticAlgo():
    def __init__(self, pop_size=20, max_generation=500, FBGN=vars.FBGN,
                 threshold=1*vars.n):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.FBGN = FBGN
        self.threshold = threshold

    @stack
    def predict(self, x, verbose=False):
        bounds = ((self.bounds, ) * vars.FBGN)
        parents = Individual_numpy.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=FBGProblem(x)
                                   )

        # Evaluate initial population
        parents = Individual_numpy.evaluate_population(parents)

        best_individual = self.loop(parents, verbose)
        return best_individual.genome

    def loop(self, parents, verbose):

        # print initial, random population
        if verbose:
            util.print_population(parents, generation=0)

        # start generation counter
        generation_counter = util.inc_generation(context=context)

        static_counter = 0
        prev_best_fitness = nan

        # main loop
        while generation_counter.generation() < self.max_generation:

            # print current generation
            if verbose == 2:
                util.print_population(parents, context['leap']['generation'])

            offspring = self.pipeline(parents)

            best_individual = max(offspring)

            parents = offspring

            generation_counter()  # increment to the next generation

            # check if threshold has been reached
            if best_individual.fitness < self.threshold:
                # Check fitness plateau
                if isclose(best_individual.fitness, prev_best_fitness,
                           abs_tol=10**-3):
                    static_counter += 1
                else:
                    static_counter = 0

                if static_counter > 200:
                    if verbose:
                        util.print_population(parents,
                                              context['leap']['generation'])

                    return best_individual

                prev_best_fitness = best_individual.fitness

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return best_individual


class genetic_algorithm_binary(GeneticAlgo):
    def __init__(self, pop_size=20, max_generation=500, FBGN=vars.FBGN,
                 threshold=1*vars.n, p_swap=1, p_mut=0.1):
        super().__init__(pop_size, max_generation, FBGN, threshold)
        self.p_swap = p_swap
        self.N = self.FBGN*2*10  # number of chromosomes
        self.ex_mut = p_mut*self.N

        def pipeline(parents):
            return pipe(
                        parents,
                        stoc_pair_selection(),
                        ops.clone,
                        one_point_crossover(p=self.p_swap),
                        mutate_bitflip(expected_num_mutations=self.ex_mut),
                        sort_genome,
                        ops.evaluate,
                        ops.pool(size=self.pop_size),
                        ops.truncation_selection(size=self.pop_size,
                                                 parents=parents)
                        )

        self.pipeline = pipeline

    @stack
    def predict(self, x, verbose=False):
        parents = Individual_numpy.create_population(
                                   self.pop_size,
                                   initialize=create_binary_sequence(self.N),
                                   decoder=FBGDecoder_binary(),
                                   problem=FBGProblem(x)
                                   )

        # Evaluate initial population
        parents = Individual_numpy.evaluate_population(parents)

        best_individual = self.loop(parents, verbose)

        _, y_hat = partial_decode(best_individual.genome)

        return y_hat


class genetic_algorithm_real(GeneticAlgo):
    def __init__(self, pop_size=30, max_generation=100, FBGN=vars.FBGN,
                 threshold=0.01, bounds=vars.bounds, p_swap=0.3,
                 std=0.01*vars.n):
        super().__init__(pop_size, max_generation, FBGN, threshold)
        self.bounds = bounds
        self.p_swap = p_swap
        self.std = std

        def pipeline(parents):
            return pipe(
                        parents,
                        stoc_pair_selection(),
                        ops.clone,
                        ops.uniform_crossover(p_swap=self.p_swap),
                        mutate_gaussian(std=self.std,
                                        expected_num_mutations=1,
                                        hard_bounds=self.bounds),
                        to_np_array,
                        single_swap(p_swap=0.3),
                        ops.evaluate,
                        ops.pool(size=self.pop_size),
                        ops.truncation_selection(size=self.pop_size,
                                                 parents=parents)
                        )

        self.pipeline = pipeline


class DistributedEstimation(GeneticAlgo):
    def __init__(self, pop_size=100, max_generation=500, FBGN=vars.FBGN,
                 threshold=0.01, bounds=vars.bounds, m=2, top_size=10):
        super().__init__(pop_size, max_generation, FBGN, threshold)
        self.bounds = bounds
        self.m = m
        self.top_size = top_size

        def pipeline(parents):
            return pipe(
                        parents,
                        update_sample(dist=self.dist,
                                      sample_size=self.pop_size),
                        clip(bounds=self.bounds),
                        ops.evaluate,
                        ops.pool(size=self.pop_size),
                        ops.truncation_selection(size=self.top_size,
                                                 parents=parents)
                        )

        self.pipeline = pipeline

    @stack
    def predict(self, x, verbose=False):
        bounds = ((self.bounds, ) * self.FBGN)  # repeat bounds for each FBG
        parents = Individual_numpy.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=FBGProblem(x)
                                   )

        # Evaluate initial population/
        parents = Individual_numpy.evaluate_population(parents)
        parents = ops.truncation_selection(parents, size=self.top_size,
                                           parents=parents)

        self.dist = GaussianMixture(warm_start=False,
                                    n_components=self.m, reg_covar=1e-10)

        best_individual = self.loop(parents, verbose)

        return best_individual.genome


class swap_differential_evolution(GeneticAlgo):
    def __init__(self, pop_size=30, max_generation=100, FBGN=vars.FBGN,
                 threshold=0.01, bounds=vars.bounds, F=0.8, p_diff=0.9):
        super().__init__(pop_size, max_generation, FBGN, threshold)
        self.bounds = bounds
        self.threshold = threshold
        self.F = F
        self.p_diff = p_diff

        def pipeline(parents):
            return pipe(
                        iter(parents),
                        ops.clone(),
                        ops.pool(size=self.pop_size),
                        diff(F=self.F, p_diff=self.p_diff, FBGN=self.FBGN),
                        ops.evaluate,
                        ops.pool(size=self.pop_size),
                        juxt(  # Parallel operations
                            identity,  # pass as is
                            compose(ops.evaluate, single_swap, ops.clone,
                                    iter),  # clone, swap and evaluate
                            ),
                        one_on_one_compare,
                        one_on_one_compare(parents=parents),
                        ops.pool(size=self.pop_size)
                        )

        self.pipeline = pipeline


class particle_swarm_optimization(GeneticAlgo):
    def __init__(self, pop_size=50, max_generation=1000, FBGN=vars.FBGN,
                 threshold=0.01, bounds=vars.bounds, w=0.6, pa=2, ga=2,
                 lr=0.1):
        super().__init__(pop_size, max_generation, FBGN, threshold)
        self.bounds = bounds
        self.w = w  # previous velocity weight constant
        self.pa = pa  # population acceleration
        self.ga = ga  # group acceleration
        self.lr = lr  # learning rate

        def pipeline(parents):
            return pipe(parents,
                        update_position(velocities=self.velocities,
                                        lr=self.lr, w=self.w, pa=self.pa,
                                        ga=self.ga),
                        clip(bounds=self.bounds),
                        ops.evaluate,
                        ops.pool(size=self.pop_size)
                        )

        self.pipeline = pipeline

    @stack
    def predict(self, x, verbose=False):
        bounds = ((self.bounds, ) * vars.FBGN)
        parents = Individual_hist.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=FBGProblem(x)
                                   )

        context['leap']['population_best'] = [Individual_simple.clone(
                                                                parents[0])]

        context['leap']['current_subpopulation'] = 0

        # Evaluate initial population
        parents = Individual_hist.evaluate_population(parents)

        # Initialize random velocities in ]0,1]
        self.velocities = np.random.rand(1, self.pop_size, vars.FBGN)
        # Move to ]-diff(bounds),diff(bounds)]
        self.velocities = (self.velocities-0.5)*2*np.diff(self.bounds)

        best_individual = self.loop(parents, verbose)

        return best_individual.genome


class dynamic_multi_swarm_particle_swarm_optimization(GeneticAlgo):
    def __init__(self, pop_size=30, max_generation=1000, FBGN=vars.FBGN,
                 threshold=0.01, bounds=vars.bounds, w=0.6, pa=2, ga=2, lr=0.1,
                 swarms=10, migration_gap=15):
        super().__init__(pop_size, max_generation, FBGN, threshold)
        self.bounds = bounds
        self.w = w  # previous velocity weight constant
        self.pa = pa  # population acceleration
        self.ga = ga  # group acceleration
        self.lr = lr  # learning rate
        self.swarms = swarms  # number of swarms
        self.migration_gap = migration_gap

        def pipeline(parents):
            return pipe(
                        parents,
                        update_position(velocities=self.velocities,
                                        lr=self.lr, w=self.w,
                                        pa=self.pa, ga=self.ga),
                        ops.evaluate,
                        ops.pool(size=self.pop_size)
                        )

        self.pipeline = pipeline

    @stack
    def predict(self, x, verbose=False):
        bounds = ((self.bounds, ) * vars.FBGN)
        subpopulations = [Individual_hist.create_population(
                                   self.pop_size,
                                   initialize=create_real_vector(bounds),
                                   decoder=FBGDecoder(),
                                   problem=FBGProblem(x)
                                   ) for _ in range(self.swarms)]

        # keep track of best individual of current subpop
        context['leap']['population_best'] = [Individual_simple.clone(pop[0])
                                              for pop in subpopulations]

        # Evaluate initial population
        for i, population in enumerate(subpopulations):
            # required to assign population_best to current subpop
            context['leap']['current_subpopulation'] = i

            Individual.evaluate_population(population)

        # Initialize random velocities in ]0,1]
        self.velocities = np.random.rand(self.swarms, self.pop_size, vars.FBGN)
        # Move to ]-diff(bounds),diff(bounds)]
        self.velocities = (self.velocities-0.5)*2*np.diff(self.bounds)

        best_individual = self.loop(subpopulations, verbose)

        return best_individual.genome

    def loop(self, subpopulations, verbose):
        # print initial, random population
        if verbose:
            for population in subpopulations:
                util.print_population(population, generation=0)

        generation_counter = util.inc_generation(context=context)

        static_counter = 0
        prev_best_fitness = nan

        while generation_counter.generation() < self.max_generation:

            for i, parents in enumerate(subpopulations):

                context['leap']['current_subpopulation'] = i

                parents = self.pipeline(parents)

            if verbose:
                for population in subpopulations:
                    util.print_population(population, generation=0)

            generation_counter()  # increment to the next generation

            population_best = max(chain(*subpopulations))

            # Check fitness plateau
            if isclose(population_best.fitness, prev_best_fitness,
                       abs_tol=10**-3):
                static_counter += 1
            else:
                static_counter = 0

            if static_counter > 200:
                break

            prev_best_fitness = population_best.fitness

            # Migration
            if generation_counter.generation() % self.migration_gap == 0:
                subpopulations = random_migrate(subpopulations, self.pop_size,
                                                self.swarms)

        # swap step
        population = chain(*subpopulations)
        population = list(swap(population))
        population = Individual.evaluate_population(population)

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return max(population)
