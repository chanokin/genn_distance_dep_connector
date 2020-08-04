import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

np_rng = NumpyRNG()
rng = NativeRNG(np_rng)

timestep = 1.
sim.setup(timestep)

n_pre = 1000
n_post = 1000
params = copy.copy(sim.IF_curr_exp.default_parameters)
pre = sim.Population(n_pre, sim.IF_curr_exp, params,
                     label='pre')
post = sim.Population(n_post, sim.IF_curr_exp, params,
                      label='post')

dist_params = {'low': 0.0, 'high': 10.0}
dist = 'uniform'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
var = 'weight'
on_device_init = bool(1)
p_conn = 0.3
conn = sim.FixedProbabilityConnector(p_connect=p_conn,
                                     rng=rng)
syn = sim.StaticSynapse(weight=rand_dist, delay=1)#rand_dist)
proj = sim.Projection(pre, post, conn, synapse_type=syn)

sim.run(10)

comp_var = np.asarray( proj.getWeights(format='array') )
shape = np.copy( comp_var.shape )
connected = np.where(~np.isnan(comp_var))
comp_var = comp_var[connected]
num_active = comp_var.size
sim.end()
ideal = n_pre * n_post * p_conn
one_pc = ideal * 0.01
print(one_pc)
abs_diff = np.abs(num_active - ideal)
print("abs({} - {}*{}*{}) = abs({} - {}) = {}".format(
    num_active, n_pre, n_post, p_conn, num_active, ideal, abs_diff
))
assert  abs_diff <= one_pc


v_min = comp_var.min()
v_max = comp_var.max()
v_avg = comp_var.mean()
print(f"Stats for sampled {var} = {v_min}, {v_avg}, {v_max}")

half_range = dist_params['low'] + \
             (dist_params['high'] - dist_params['low']) / 2.
print(f"Stats for ideal {var} = {dist_params['low']}, "
      f"{half_range}, {dist_params['high']}")

from scipy import stats
scale = dist_params['high'] - dist_params['low']
s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
assert p > 0.05
