import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

np_rng = NumpyRNG()
rng = NativeRNG(np_rng, seed=1)

timestep = 1.
sim.setup(timestep)

n_neurons = 10000
params = copy.copy(sim.IF_curr_exp.default_parameters)
print(params)
dist_params = {'mu': -70.0, 'sigma': 1.0}
dist = 'normal'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
var = 'v'

post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                      label='rand pop')
post.initialize(**{var: rand_dist})
post.record(var)

sim.run(10)

comp_var = post.get_data(var)
comp_var = comp_var.segments[0].analogsignals[0]
comp_var = np.asarray([float(x) for x in comp_var[0, :]])
# print(dir(comp_var))
sim.end()
from scipy import stats
s, p = stats.kstest(comp_var - dist_params['mu'], 'norm')


# print(f"Values for v_reset = {v_reset}")
v_std = comp_var.std()
v_mean = comp_var.mean()
print(f"Stats for sampled {var} = {v_mean}, {v_std}")

print(f"Stats for ideal {var} = {dist_params['mu']}, "
      f"{dist_params['sigma']}")

epsilon = 1
assert np.abs(v_mean - dist_params['mu']) < epsilon
assert np.abs(v_std - dist_params['sigma']) < epsilon
