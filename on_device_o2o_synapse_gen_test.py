import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

np_rng = NumpyRNG(seed=1)
rng = NativeRNG(np_rng, seed=1)

timestep = 1.
sim.setup(timestep)

n_neurons = 10000
params = copy.copy(sim.IF_curr_exp.default_parameters)
pre = sim.Population(n_neurons, sim.IF_curr_exp, params,
                     label='pre')
post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                      label='post')

dist_params = {'low': 0.0, 'high': 10.0}
dist = 'uniform'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
var = 'weight'
on_device_init = bool(1)
conn = sim.OneToOneConnector()
syn = sim.StaticSynapse(weight=rand_dist, delay=1)#rand_dist)
proj = sim.Projection(pre, post, conn, synapse_type=syn)

sim.run(10)

comp_var = np.asarray( proj.getWeights(format='array') )
shape = np.copy( comp_var.shape )
connected = np.where(~np.isnan(comp_var))
comp_var = comp_var[connected]
num_active = comp_var.size
sim.end()

assert num_active == n_neurons
assert np.all(connected[0] == connected[1])

from scipy import stats
scale = dist_params['high'] - dist_params['low']
s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
assert p > 0.05

v_min = comp_var.min()
v_max = comp_var.max()
v_avg = comp_var.mean()
print(f"Stats for sampled {var} = {v_min}, {v_avg}, {v_max}")

half_range = dist_params['low'] + \
             (dist_params['high'] - dist_params['low']) / 2.
print(f"Stats for ideal {var} = {dist_params['low']}, "
      f"{half_range}, {dist_params['high']}")

epsilon = 0.1
assert np.abs(v_min - dist_params['low']) < epsilon
assert np.abs(v_max - dist_params['high']) < epsilon
assert np.abs(v_avg - half_range) < epsilon