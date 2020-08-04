import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution
import matplotlib.pyplot as plt
from pyNN.space import Grid3D
import sys
from MaxDistanceFixedProbabilityConnector import MaxDistanceFixedProbabilityConnector

np_rng = NumpyRNG(seed=1)
rng = NativeRNG(np_rng, seed=1)

timestep = 1.
sim.setup(timestep)

n_z = 5#20
n_pre = (10, 10)
shape_post = (10, 10, n_z)
n_post = int(np.prod(shape_post))
ratioXY = float(shape_post[0])/shape_post[1]
ratioXZ = float(shape_post[0])/shape_post[2]
structure = Grid3D(ratioXY, ratioXZ, dz=0.001 * (1./n_z))
params = copy.copy(sim.IF_curr_exp.default_parameters)
pre = sim.Population(n_pre, sim.IF_curr_exp, params,
                     label='pre')
post = sim.Population(n_post, sim.IF_curr_exp, params,
                      label='post', structure=structure)

dist_params = {'low': 0.0, 'high': 10.0}
dist = 'uniform'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
var = 'weight'
conn = MaxDistanceFixedProbabilityConnector(3, 0.95, rng=rng)
syn = sim.StaticSynapse(weight=3, delay=1)
proj = sim.Projection(pre, post, conn, synapse_type=syn)

sim.run(10)

comp_var = np.asarray( proj.getWeights(format='array') )
shape = np.copy( comp_var.shape )
print(shape)
connected = np.where(~np.isnan(comp_var))
# print(connected)
img = np.zeros(n_pre)
s = comp_var.shape[1]
ncols = n_z
nrows = s // ncols + int(s%ncols > 0)
fig = plt.figure(figsize=(ncols, nrows))

for col in range(s):
    sys.stdout.write("\r{}/{}".format(col+1, s))
    sys.stdout.flush()
    wcol = comp_var[:, col]
    pres = np.where(~np.isnan(wcol))[0]
    img[:] = 0
    img[pres//n_pre[1], pres%n_pre[1]] = 1
    ax = plt.subplot(nrows, ncols, col + 1)
    ax.imshow(img.copy())
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("dist_weights.pdf")
# plt.show()

comp_var = comp_var[connected]
num_active = comp_var.size
sim.end()

# assert num_active == n_neurons**2

# print(f"Values for v_reset = {v_reset}")
v_min = comp_var.min()
v_max = comp_var.max()
v_avg = comp_var.mean()
print(f"Stats for sampled {var} = {v_min}, {v_avg}, {v_max}")

half_range = dist_params['low'] + \
             (dist_params['high'] - dist_params['low']) / 2.
print(f"Stats for ideal {var} = {dist_params['low']}, "
      f"{half_range}, {dist_params['high']}")

epsilon = 1
assert np.abs(v_min - dist_params['low']) < epsilon
assert np.abs(v_max - dist_params['high']) < epsilon
assert np.abs(v_avg - half_range) < epsilon