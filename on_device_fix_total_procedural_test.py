import numpy as np
import pynn_genn as sim
import copy
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

np_rng = NumpyRNG()
rng = NativeRNG(np_rng)

timestep = 1.
sim.setup(timestep)

n_pre =  100
n_post = 50000
params = copy.copy(sim.IF_curr_exp.default_parameters)
times = [[1] for _ in range(n_pre)]
pre = sim.Population(n_pre, sim.SpikeSourceArray,
                     {'spike_times': times},
                     label='pre')
post = sim.Population(n_post, sim.IF_curr_exp, params,
                      label='post')
post.record('spikes')

n = 2
dist_params = {'low': 4.99, 'high': 5.01}
dist = 'uniform'
rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
var = 'weight'
on_device_init = bool(1)
conn = sim.FixedNumberPostConnector(n, with_replacement=True, rng=rng)
syn = sim.StaticSynapse(weight=rand_dist, delay=1)#rand_dist)
proj = sim.Projection(pre, post, conn, synapse_type=syn, use_procedural=bool(0),
                      num_threads_per_spike=1)

sim.run(100)
data = post.get_data()
spikes = np.asarray(data.segments[0].spiketrains)
sim.end()

all_at_appr_time = 0
sum_spikes = 0
for i, times in enumerate(spikes):
    sum_spikes += (1 if len(times) else 0)
    if len(times) == 1 and times[0] == 9:
        all_at_appr_time += 1

print(all_at_appr_time)
print(sum_spikes)
assert np.abs(sum_spikes - (n_pre * n)) < 3
assert np.abs(all_at_appr_time - (n_pre * n)) < 3
