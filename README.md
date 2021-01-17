## 概要
本プログラムは遺伝的アルゴリズムのライブラリdeapのラッパーである。本来煩雑なプログラムが以下のように簡単にかける。以下はone max problemの例。目的関数と制約を設定すれば、あとは`ga.run()`で実行可能。将来的にいろいろな制約(離散制約、総和制約など)も加えられるようにしたいが、現状最小値最大値の制約と、nhot制約のみ

```
#　初期個体（なければcolumnsだけ定義)
cols = ['a', 'b','c', 'd', 'e']
df = pd.DataFrame(columns=cols)
ga = GenericAlgorithm(df)

# one max problem
def func(x):
    return sum(x)

#　最大化したいのでweightは1
weights = (1.0, )
ga.set_func([func], weights)

# 最小値と最大値
lower = [0, 0, 0, 0, 0]
upper = [1, 1, 1, 1, 1]
ga.set_lim(lower, upper)


n_population = 100
n_generation = 100
p_crossover = 0.5
p_mutation = 0.1
results = ga.run(n_population, n_generation, p_crossover, p_mutation)
```

## Usage
### jupyter notebookで試す
- `pip install pipenv`
- `pipenv install --dev`
- `pipenv run python -m ipykernel install --user --name="ga"`
- `jupyter notebook`
- kernelにgaを選択、jupyter notebookでプログラム作成
