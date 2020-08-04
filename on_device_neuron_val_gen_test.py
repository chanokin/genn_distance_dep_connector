import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

np_rng = NumpyRNG()
rng = NativeRNG(np_rng, seed=1)

timestep = 1.
sim.setup(timestep)

n_neurons = 1000
params = copy.copy(sim.IF_curr_exp.default_parameters)
print(params)
# dist_params = {'low': -70.0, 'high': -60.0}
dist_params = {'low': 10.0, 'high': 20.0}
dist = 'uniform'
var = 'tau_m'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
params[var] = rand_dist
params['cm'] = 2.

pop = sim.Population(n_neurons, sim.IF_curr_exp, params,
                     label='rand pop')

sim.run(10)

gen_var = np.asarray( pop.get(var) )

sim.end()

# print(f"Values for v_reset = {v_reset}")
v_min = gen_var.min()
v_max = gen_var.max()
v_avg = gen_var.mean()
print(f"Stats for sampled {var} = {v_min}, {v_avg}, {v_max}")

half_range = dist_params['low'] + \
             (dist_params['high'] - dist_params['low']) / 2.
print(f"Stats for ideal {var} = {dist_params['low']}, "
      f"{half_range}, {dist_params['high']}")

epsilon = 1.
assert np.abs(v_min - dist_params['low']) < epsilon
assert np.abs(v_max - dist_params['high']) < epsilon
assert np.abs(v_avg - half_range) < epsilon