import random
from functools import partial

import numpy as np
from deap import creator, base, tools

from .config import Config, NhotConfig, LimConfig


class GenericAlgorithm:
    def __init__(self, df):
        self.config = Config()
        self.config.init_df = df

    def set_lim(self, lower, upper):
        lc = LimConfig(lower, upper)
        self.config.lim = lc

    def set_nhot(self, cols, n, value):
        nhotconfig = NhotConfig(n=n, cols=cols, value=value)
        self.config.nhot.append(nhotconfig)

    def set_func(self, func, weights):
        self.config.func = func
        self.config.weight = weights

    @property
    def func(self):
        return partial(_multiobject_funcs, self.config.func)

    def run(self, n_population, n_generation, p_crossover, p_mutation):
        creator.create("Fitness", base.Fitness, weights=self.config.weight)
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox = self._make_toolbox()
        stats = self._make_stats()
        pop = toolbox.population(n=n_population)
        pop = self._set_constraints(pop)
        print('Start of evolution')
        pop = self._initialize_pop(pop, toolbox)
        print('Evaluated {} individuals'.format(len(pop)))

        for generation in range(n_generation):
            print('-- Generation {} --'.format(generation+1))

            # Mate
            for child1, child2 in zip(pop[::2], pop[1::2]):
                if random.random() < p_crossover:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutate
            for mutant in pop:
                if random.random() < p_mutation:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 適応度を削除した個体について適応度の再評価を行う
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            invalid_ind = self._initialize_pop(invalid_ind, toolbox)

            # Generate individual
            random_ind = toolbox.population(len(pop)-len(invalid_ind))
            random_ind = self._initialize_pop(random_ind, toolbox)

            pop = pop + invalid_ind + random_ind
            offspring = toolbox.select(pop, n_population)
            pop = list(map(toolbox.clone, offspring))

            pop = self._set_constraints(pop)
            pop = self._initialize_pop(pop, toolbox)
            record = stats.compile(pop)
            print(record)

        result = self._convert_pop2result(pop)
        return result

    def _set_constraints(self, pop):
        if self.config.nhot:
            pop = _nhot_constraints(pop, self.config)
        return pop

    def _initialize_pop(self, pop, toolbox):
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        return pop

    def _convert_pop2result(self, pop):
        result = [list(ind)+[val for val in ind.fitness.values] for ind in pop]
        return result

    def _make_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("create_ind", _create_ind_uniform,
                         self.config.lims_lower, self.config.lims_upper)
        toolbox.register("individual", tools.initIterate,
                         creator.Individual, toolbox.create_ind)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("evaluate", self.func)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def _make_stats(self):
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats


def _nhot_constraints(individuals, config):
    individuals_new = []
    for individual in individuals:
        for indice, nhot in zip(config.nhot_index, config.nhot):
            if not (sum([individual[i] for i in indice]) == nhot.value*nhot.n):
                index_sel = random.sample(indice, nhot.n)
                for i in indice:
                    individual[i] = 0
                for i in index_sel:
                    individual[i] = nhot.value
                del individual.fitness.values
        individuals_new.append(individual)
    return individuals_new


def _create_ind_uniform(min_boundary, max_boundary):
    individual = []
    for lower, upper in zip(min_boundary, max_boundary):
        individual.append(random.uniform(lower, upper))
    return individual


def _multiobject_funcs(funcs, individual):
    y_list = []
    for func in funcs:
        y_list.append(func(individual))
    return tuple(y_list)


if __name__ == '__main__':
    import pandas as pd
    cols = ['a', 'b', 'c', 'd', 'e', 'f']
    df = pd.DataFrame(columns=cols)
    ga = GenericAlgorithm(df)

    def func(x):
        return -sum([i**2 for i in x])+sum(x)

    weights = (1.0, )
    ga.set_func([func], weights)
    lower = [0, 0, 0, 0, 0, 0]
    upper = [1, 1, 1, 1, 1, 1]
    ga.set_lim(lower, upper)
    ga.set_nhot(['a', 'c', 'e'], 1, 1)
    result = ga.run(100, 100, 0.5, 0.2)
    for r in result:
        print(r)
