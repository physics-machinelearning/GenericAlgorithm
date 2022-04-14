from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class NhotConfig:
    n: int
    cols: list
    value: float


@dataclass
class LimConfig:
    lower: list
    upper: list

@dataclass
class DiscreteConfig:
    col: str
    values: List


@dataclass
class TotalConfig:
    cols: List
    total: float


@dataclass
class Config:
    init_df: pd.DataFrame = None
    lim: LimConfig = None
    nhot: list = field(default_factory=list)
    discrete: DiscreteConfig = None
    total: TotalConfig = None
    func: list = field(default_factory=list)
    weight: tuple = field(default_factory=tuple)

    @property
    def cols(self):
        return list(self.init_df.columns)

    @property
    def lims_lower(self):
        return self.lim.lower

    @property
    def lims_upper(self):
        return self.lim.upper

    @property
    def nhot_index(self):
        indice_list = []
        for nhot in self.nhot:
            indice = []
            for col in nhot.cols:
                index = self.cols.index(col)
                indice.append(index)
            indice_list.append(indice)
        return indice_list
