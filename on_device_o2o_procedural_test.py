import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

np_rng = NumpyRNG(seed=1)
rng = NativeRNG(np_rng, seed=1)

timestep = 1.
sim.setup(timestep)

n_neurons = 100
params = copy.copy(sim.IF_curr_exp.default_parameters)
pre = sim.Population(n_neurons, sim.SpikeSourceArray,
                     {'spike_times': [[1 + i] for i in range(n_neurons)]},
                     label='pre')
params['tau_syn_E'] = 5.
post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                      label='post')
post.record('spikes')

dist_params = {'low': 0.0, 'high': 10.0}
dist = 'uniform'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
var = 'weight'
on_device_init = bool(1)
conn = sim.OneToOneConnector()
syn = sim.StaticSynapse(weight=5, delay=1)#rand_dist)
proj = sim.Projection(pre, post, conn, synapse_type=syn, use_procedural=bool(0))

sim.run(2 * n_neurons)
data = post.get_data()
spikes = np.asarray(data.segments[0].spiketrains)

sim.end()

all_at_appr_time = 0
sum_spikes = 0
for i, times in enumerate(spikes):
    sum_spikes += len(times)
    if int(times[0]) == (i + 9):
        all_at_appr_time += 1

assert sum_spikes == n_neurons
assert all_at_appr_time == n_neurons
#each neuron spikes once because