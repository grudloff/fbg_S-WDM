from toolz import pipe

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
from leap_ec.ops import iteriter_op
from typing import Iterator
from itertools import permutations  

from simulation import X
import numpy as np
import variables as vars
import random
from math import factorial

from sklearn.mixture import GaussianMixture

class Individual_hist(Individual_np):
    ''' Individual with memory of best state genome and fitness
    '''

    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)
        self.best = Individual_placeholder(genome, nan)

    def evaluate_imp(self):
        """ This is the evaluate 'implementation' called by
            self.evaluate().   It's intended to be optionally over-ridden by
            sub-classes to give an opportunity to pass in ancillary data to
            the evaluate process either by tailoring the problem interface or
            that of the given decoder.
        """
        self.fitness = super().evaluate_imp()
        #update individual best
        if self > self.best:
            self.best = Individual_placeholder.clone(self)

        #update subpopulation best
        i = context['leap']['current_subpopulation']
        subpop_best = context['leap']['population_best'][i]
        if self > subpop_best:
            context['leap']['population_best'][i] = Individual_placeholder.clone(self)

        return self.fitness
class FBGDecoder(Decoder):

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        #genome = np.array(genome)
        phenome = X(genome)
        return phenome

    def __repr__(self):
        return type(self).__name__ + "()"

to_float = np.exp2(np.arange(10)) #transform array
to_float = to_float/np.sum(to_float) #normalize

def bool2float(a):
    # Transforms size 10 bool to float between 0 and 1
    return np.dot(a, to_float)

def partial_decode(genome):
    """ return I and then A_b from genome """
    I, A_b = np.split(genome, 2)
    I = np.stack(np.split(I, vars.FBGN))
    I = bool2float(I)
    yield I
    A_b = np.stack(np.split(A_b, vars.FBGN))
    A_b = vars.A0 - vars.D + 2*vars.D*bool2float(A_b)
    yield A_b


class FBGDecoder_binary(Decoder):

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        I, A_b = partial_decode(genome)
        phenome = X(A_b, I)
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
def stoc_swap(individual_itr: Iterator, p_swap: int = 0.5) -> Iterator:

    for individual in individual_itr:
        genome_perm = permutations(individual.genome)
        # always yield original
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
def sort_genome(population: Iterator)-> Iterator:
    """ Sort genome to place larger I first """
    for ind in population:
        I = next(partial_decode(ind.genome)) # decode I from genome
        indx = np.argsort(I) # indeces of sorted I
        indx = indx[::-1] # larger first\

        I, A_b = np.split(ind.genome, 2) #split I from A_b chromosomes
        I = np.array(np.split(I, vars.FBGN)) # split into chromosomes
        A_b = np.array(np.split(A_b, vars.FBGN))
        I = np.take(I, indx, axis=0) #sort chromosomes according to indx
        A_b = np.take(A_b, indx, axis=0)
        I = np.concatenate(I) #combine sorted chromosomes
        A_b = np.concatenate(A_b)
        
        ind.genome = np.concatenate((I, A_b))
        yield ind


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

        id_swap = np.random.rand(n) < p_diff
        # TODO add random index?
        individual.genome[id_swap] = V.genome[id_swap] 
        individual.evaluate()
        if original > individual:
            yield original
        else:
            yield individual

@curry
@listiter_op
# differential step and swap step of swap differential evolution algorithm
def diff_swap(pop: List, F: float = 0.8, p_diff:float = 0.9, n: int = 2) -> Iterator:
    # F: Differential weight $F /in [0,2]$
    # p_diff: crossover probability $p_diff /in [0,1]
    # n: number of genes 

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

@curry
def update_position(parents, velocities, bounds, lr):
    for ind, v in zip(parents, velocities):
        ind.genome += v * lr
        ind.genome = ind.genome.clip(*bounds)
        yield ind
        
def random_migrate(subpopulations, pop_size, swarms, migration_gap):
    #flatten
    population = list(chain(*subpopulations))
    #shuffle
    random.shuffle(population)
    #deflatten
    subpopulations = [population[i:i+pop_size] for i in range(0, swarms*pop_size, pop_size)]

    return subpopulations
def _select_samples(candidate, N):
    """
    obtain 3 random integers from range(N),
    without replacement. You can't have the original candidate.
    """
    idx = list(range(N))
    idx.pop(candidate) #remove candidate
    id_list = random.sample(idx, k=3) 
    return id_list

def get_genome(population):
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
def get_top(dist):
    #get most probable cluster
    top_cluster_idx = np.argmax(dist.weights_)
    #return center of top cluster
    return dist.means_[top_cluster_idx]

def sample(dist):
    return dist.sample()[0]

def update_velocities(parents, velocities, w, pa, ga):
    N = len(parents)
    r = np.random.rand(N, vars.FBGN)
    q = np.random.rand(N, vars.FBGN)
    parents_genome = get_genome(parents)
    particle_best_genome = get_genome_best(parents)
    i = context['leap']['current_subpopulation']
    subpop_best = context['leap']['population_best'][i]
    return w*velocities + pa*r*(particle_best_genome - parents_genome) + ga*q*(subpop_best.genome - parents_genome)

class DistributedEstimation():
    def __init__(self, pop_size=30, max_generation=100, bounds = vars.bounds, n = vars.FBGN, threshold = 1*vars.n, k = 10):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.bounds = bounds
        self.n = n
        self.threshold = threshold
        self.k = k

    def predict(self, x, verbose=False):
        if len(x.shape)==1:
            return self._predict(x, verbose)
        else:
            return np.stack([self._predict(k, verbose) for k in x])


    def _predict(self, x, verbose):
        parents = Individual.create_population(self.pop_size,
                                        initialize=create_real_vector(((self.bounds, ) * self.n) ),
                                        decoder=FBGDecoder(),
                                        problem=FBGProblem(x))

        # Evaluate initial population/
        parents = Individual.evaluate_population(parents)
        best_individual = ops.truncation_selection(parents, 1)[0]

        # print initial, random population
        if verbose:
            util.print_population(parents, generation=0)

        generation_counter = util.inc_generation(context=context)

        #TODO Authors use n_components=10 when we know there is one global minima
        # and !n-1 local minimas, so there should be !n clusters
        dist = GaussianMixture(warm_start=False, n_components=factorial(self.n), reg_covar=1e-10)

        while generation_counter.generation() < self.max_generation:
        
            if verbose == 2:
                util.print_population(parents, context['leap']['generation'])

            offspring = pipe(
                                parents,
                                ops.truncation_selection(size = self.k),
                            )

            dist.fit(get_genome(offspring))
            
            #TODO In the paper they sample only (n-m) 
            # and keep the previous top n, to not loose info
            # But this shouldn't be necessary as the distribution
            # is updated not reconstructed from scratch
            if np.any(dist.weights_ < 1e-15):
                dist.weights_[0]=1

            new_parents_genome = dist.sample(self.pop_size)[0]
            new_parents = [Individual(genome, decoder=FBGDecoder(), problem=FBGProblem(x)) for genome in new_parents_genome]
            new_parents = Individual.evaluate_population(new_parents)
            parents = offspring + new_parents

            best_individual = Individual(get_top(dist), decoder=FBGDecoder(),
                                         problem=FBGProblem(x))
            best_individual.evaluate()

            #if best_individual.fitness < self.threshold:
            #    return best_individual.genome

            generation_counter()  # increment to the next generation

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return np.array(best_individual.genome)


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
                                parents,
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
 

class genetic_algorithm_real():
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
                                parents,
                                ops.stoc_pair_selection(),
                                ops.clone,
                                ops.uniform_crossover(p_swap = self.p_swap),
                                mutate_gaussian(std = self.std, expected_num_mutations = 1, hard_bounds = self.bounds),
                                swap,
                                ops.evaluate,
                                ops.pool(size = -1),
                                ops.truncation_selection(size = self.pop_size, parents = parents)
                            )

            parents = offspring

            best_individual = parents[0] #truncation_selection sorts parents

            #if best_individual.fitness < self.threshold:
            #    return best_individual.genome

            generation_counter()  # increment to the next generation

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return np.array(best_individual.genome)


class genetic_algorithm_binary():
    def __init__(self, pop_size=20, max_generation=500, FBGN = vars.FBGN, threshold = 1*vars.n, p_swap=1, p_mut=0.1):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.FBGN = FBGN
        self.threshold = threshold
        self.p_swap = p_swap
        self.p_mut = p_mut

    def predict(self, x, verbose=False):
        if len(x.shape)==1:
            return self._predict(x, verbose)
        else:
            return np.stack([self._predict(k, verbose) for k in x])


    def _predict(self, x, verbose):
        N=self.FBGN*2*10 # Two 10-bit chromosomes per FBG
        parents = Individual.create_population(self.pop_size,
                                                initialize=create_binary_sequence(N),
                                                decoder=FBGDecoder_binary(),
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
                                parents,
                                ops.stoc_pair_selection(),
                                ops.clone,
                                ops.one_point_crossover(p = self.p_swap),
                                mutate_bitflip(expected_num_mutations = self.p_mut*N),
                                sort_genome,
                                ops.evaluate,
                                ops.pool(size = self.pop_size),
                                ops.truncation_selection(size = self.pop_size, parents = parents)
                            )

            parents = offspring

            best_individual = parents[0] #truncation_selection sorts parents

            #if best_individual.fitness < self.threshold:
            #    return best_individual.genome

            generation_counter()  # increment to the next generation

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        _, y_hat = partial_decode(best_individual.genome)

        return y_hat


class particle_swarm_optimization():
    def __init__(self, pop_size=50, max_generation=1000, bounds = vars.bounds, threshold = 1*vars.n, w=0.6, pa=2, ga=2, lr=0.1):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.bounds = bounds
        self.threshold = threshold
        self.w = w # previous velocity weight constant 
        self.pa = pa # population acceleration
        self.ga = ga # group acceleration
        self.lr = lr # learning rate

    @stack
    def predict(self, x, verbose=False):
        population = Individual_hist.create_population(self.pop_size,
                                                initialize=create_real_vector(((self.bounds, ) * vars.FBGN) ),
                                                decoder=FBGDecoder(),
                                                problem=FBGProblem(x))

        context['leap']['population_best']=[Individual_placeholder.clone(population[0])]

        context['leap']['current_subpopulation'] = 0

        # Evaluate initial population
        population = Individual_hist.evaluate_population(population)

        # Initialize velocities 
        velocities = np.random.rand(self.pop_size, vars.FBGN) # Initialize random velocities in ]0,1]
        velocities = (velocities-0.5)*2*np.diff(self.bounds) # Move to ]-diff(bounds),diff(bounds)]

        # print initial, random population
        if verbose:
            util.print_population(population, generation=0)

        generation_counter = util.inc_generation(context=context)

        while generation_counter.generation() < self.max_generation:

        #util.print_population(parents, context['leap']['generation'])

            velocities = update_velocities(population, velocities, self.w, self.pa, self.ga)

            population = pipe(population,
                           update_position(velocities = velocities, bounds = self.bounds, lr = self.lr),
                           ops.evaluate,
                           ops.pool(size = -1))

            #if best_individual.fitness < self.threshold:
            #    return best_individual.genome

            generation_counter()  # increment to the next generation

        if verbose:
            util.print_population(population, context['leap']['generation'])
        
        return get_genome_pop_best(population)


class dynamic_multi_swarm_particle_swarm_optimization():
    def __init__(self, pop_size=30, max_generation=1000, bounds = vars.bounds, w=0.6, pa=2, ga=2, lr=0.1, swarms=10, migration_gap=15):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.bounds = bounds
        self.w = w # previous velocity weight constant 
        self.pa = pa # population acceleration
        self.ga = ga # group acceleration
        self.lr = lr # learning rate
        self.swarms = swarms # number of swarms
        self.migration_gap = migration_gap

    @stack
    def predict(self, x, verbose=False):
        subpopulations = [Individual_hist.create_population(self.pop_size,
                                                initialize=create_real_vector(((self.bounds, ) * vars.FBGN) ),
                                                decoder=FBGDecoder(),
                                                problem=FBGProblem(x)) for _ in range(self.swarms)]

        context['leap']['population_best']=[Individual_placeholder.clone(pop[0]) for pop in subpopulations]

        # Evaluate initial population
        for i, population in enumerate(subpopulations):
            context['leap']['current_subpopulation'] = i #required to assign population_best to current subpop
            Individual.evaluate_population(population)

        # Initialize velocities 
        velocities = np.random.rand(self.swarms, self.pop_size, vars.FBGN) # Initialize random velocities in ]0,1]
        velocities = (velocities-0.5)*2*np.diff(self.bounds) # Move to ]-diff(bounds),diff(bounds)]

        

        # print initial, random population
        if verbose:
            util.print_population(parents, generation=0)

        generation_counter = util.inc_generation(context=context)
        
        static_counter = 0
        prev_best_fitness = nan

        while generation_counter.generation() < self.max_generation:
        
            if verbose:
                util.print_population(parents, context['leap']['generation'])

            for i, parents in enumerate(subpopulations):

                context['leap']['current_subpopulation'] = i

                velocities[i] = update_velocities(parents, velocities[i], self.w, self.pa, self.ga)

                parents = pipe(
                               parents,
                               update_position(velocities = velocities[i], bounds = self.bounds, lr = self.lr),
                               ops.evaluate,
                               ops.pool(size = -1)
                               )

            population_best = max(chain(*subpopulations))

            if isclose(population_best.fitness, prev_best_fitness, abs_tol=10**-3):
                static_counter += 1
            else:
                static_counter = 0

            prev_best_fitness = population_best.fitness

            if static_counter > 200:
                break

            if generation_counter.generation()%self.migration_gap == 0:
                subpopulations = random_migrate(subpopulations, self.pop_size, self.swarms)

            generation_counter()  # increment to the next generation

        #swap step
        population = list(chain(*subpopulations))
        population = list(swap(population))
        population = Individual.evaluate_population(population)

        if verbose:
            util.print_population(parents, context['leap']['generation'])

        return get_genome_pop_best(population)


def main():

    import simulation as sim
    import evolutionary as ev

    y = np.array([1549.5*vars.n, 1550.5*vars.n])
    x = np.sum(sim.R(vars.A[:, None], y[None, :], vars.I[None, :], vars.dA[None, :]) ,axis=-1)

    model = ev.dynamic_multi_swarm_particle_swarm_optimization()
    y_hat = model.predict(x)
    print("y = "+str(y))
    print("y_hat = "+str(y_hat))
    print(np.mean(np.abs(y_hat-y)))



if __name__ == "__main__":
    main()