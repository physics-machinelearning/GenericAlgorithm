import pytest

import pandas as pd

from ga_tools.optimization import GenericAlgorithm


class TestOptimization:
    def setup_method(self):
        cols = ['a', 'b', 'c', 'd', 'e', 'f']
        df = pd.DataFrame(columns=cols)
        self.ga = GenericAlgorithm(df)

        def func(x):
            return -sum([i**2 for i in x])+sum(x)

        weights = (1.0, )
        self.ga.set_func([func], weights)
        self.lower = [0, 0, 0, 0, 0, 0]
        self.upper = [1, 1, 1, 1, 1, 1]
        self.ga.set_lim(self.lower, self.upper)
        self.ga.set_nhot(['a', 'c', 'e'], 1, 1)

    def test_limit_constraint(self):
        results = self.ga.run(100, 100, 0.5, 0.2)
        for result in results:
            for low, value in zip(self.lower, result):
                assert value >= low
                
