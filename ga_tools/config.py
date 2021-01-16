from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Config:
    init_df: pd.DataFrame = None
    nhot: list = field(default_factory=list)
    func: list = field(default_factory=list)


@dataclass
class NhotConfig:
    n: int = 0
    cols: list = field(default_factory=list)


class SetConfig:
    def __init__(self, df):
        self.config = Config()
        self.df = df

    def set_nhot(self, cols, n):
        nhotconfig = NhotConfig(n=n, cols=cols)
        self.config.nhot.append(nhotconfig)

    def set_func(self, funcs):
        for func in funcs:
            self.config.func.append(func)
