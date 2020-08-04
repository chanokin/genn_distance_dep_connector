from pynn_genn.random import NativeRNG
from pyNN.random import NumpyRNG
np_rng = NumpyRNG()
rng = NativeRNG(np_rng)

dist = "wrong distribution"
assert rng._supports_dist(dist) == False

dist = "uniform"
assert rng._supports_dist(dist) == True


params = {'a': 1, 'b': 2, 'c': 3}
assert rng._check_params(dist, params) == False

params = {'low': 0, 'high': 1}
assert rng._check_params(dist, params) == True

rng1 = NativeRNG(np_rng, seed=1)
assert rng1 is rng
assert rng1.seed == rng.seed

rng2 = NativeRNG(np_rng, seed=9)
assert rng1.seed == rng.seed and rng2.seed == rng.seed
assert rng1 is rng2
assert rng is rng2

print(id(rng))
print(id(rng1))
print(id(rng2))