# encoding: utf-8

from nose.tools import assert_equal, assert_raises, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pyNN.utility import assert_arrays_equal, assert_arrays_almost_equal

from nose.plugins.skip import SkipTest
from pyNN.utility import init_logging
from pyNN.random import RandomDistribution
import pynn_genn
import numpy
from pyNN.errors import InvalidParameterValueError


def scenario3(sim):
    """
    Simple feed-forward network network with additive STDP. The second half of
    the presynaptic neurons fires faster than the second half, so their
    connections should be potentiated more.
    """

    init_logging(logfile=None, debug=True)
    second = 1000.0
    duration = 10
    tau_m = 20  # ms
    cm = 1.0  # nF
    v_reset = -60
    cell_parameters = dict(
        tau_m=tau_m,
        cm=cm,
        v_rest=-70,
        e_rev_E=0,
        e_rev_I=-70,
        v_thresh=-54,
        v_reset=v_reset,
        tau_syn_E=5,
        tau_syn_I=5,
    )
    g_leak = cm / tau_m  # µS

    w_min = 0.0 * g_leak
    w_max = 0.05 * g_leak

    r1 = 5.0
    r2 = 40.0

    sim.setup()
    pre = sim.Population(100, sim.SpikeSourcePoisson())
    post = sim.Population(10, sim.IF_cond_exp())

    pre.set(duration=duration * second)
    pre.set(start=0.0)
    pre[:50].set(rate=r1)
    pre[50:].set(rate=r2)
    assert_equal(pre[49].rate, r1)
    assert_equal(pre[50].rate, r2)
    post.set(**cell_parameters)
    post.initialize(v=RandomDistribution('normal', mu=v_reset, sigma=5.0))

    stdp = sim.STDPMechanism(
                sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                  A_plus=0.01, A_minus=0.01),
                sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max),
                #dendritic_delay_fraction=0.5))
                dendritic_delay_fraction=1)

    connections = sim.Projection(pre, post, sim.AllToAllConnector(),
                                 synapse_type=stdp,
                                 receptor_type='excitatory')

    initial_weight_distr = RandomDistribution('uniform', low=w_min, high=w_max)
    connections.randomizeWeights(initial_weight_distr)
    initial_weights = connections.get('weight', format='array', gather=False)
    # assert initial_weights.min() >= w_min
    # assert initial_weights.max() < w_max
    # assert initial_weights[0, 0] != initial_weights[1, 0]

    pre.record('spikes')
    post.record('spikes')
    post[0:1].record('v')

    sim.run(duration * second)

    actual_rate = pre.mean_spike_count() / duration
    expected_rate = (r1 + r2) / 2
    errmsg = "actual rate: %g  expected rate: %g" % (actual_rate, expected_rate)
    assert abs(actual_rate - expected_rate) < 1, errmsg
    #assert abs(pre[:50].mean_spike_count()/duration - r1) < 1
    #assert abs(pre[50:].mean_spike_count()/duration- r2) < 1
    final_weights = connections.get('weight', format='array', gather=False)
    assert initial_weights[0, 0] != final_weights[0, 0]

    try:
        import scipy.stats
    except ImportError:
        raise SkipTest
    t, p = scipy.stats.ttest_ind(initial_weights[:50, :].flat, initial_weights[50:, :].flat)
    assert p > 0.05, p
    t, p = scipy.stats.ttest_ind(final_weights[:50, :].flat, final_weights[50:, :].flat)
    assert p < 0.01, p
    assert final_weights[:50, :].mean() < final_weights[50:, :].mean()
    sim.end()
    return initial_weights, final_weights, pre, post, connections
    
def scenario2(sim):
    """
    Array of neurons, each injected with a different current.

    firing period of a IF neuron injected with a current I:

    T = tau_m*log(I*tau_m/(I*tau_m - v_thresh*cm))

    (if v_rest = v_reset = 0.0)

    we set the refractory period to be very large, so each neuron fires only
    once (except neuron[0], which never reaches threshold).
    """
    n = 83
    t_start = 25.0
    duration = 100.0
    t_stop = 150.0
    tau_m = 20.0
    v_thresh = 10.0
    cm = 1.0
    cell_params = {"tau_m": tau_m, "v_rest": 0.0, "v_reset": 0.0,
                   "tau_refrac": 100.0, "v_thresh": v_thresh, "cm": cm}
    I0 = (v_thresh * cm) / tau_m
    sim.setup(timestep=0.01, min_delay=0.1, spike_precision="off_grid")
    neurons = sim.Population(n, sim.IF_curr_exp(**cell_params))
    neurons.initialize(v=0.0)
    I = numpy.arange(I0, I0 + 1.0, 1.0 / n)
    currents = [sim.DCSource(start=t_start, stop=t_start + duration, amplitude=amp)
                for amp in I]
    for j, (neuron, current) in enumerate(zip(neurons, currents)):
        if j % 2 == 0:                      # these should
            neuron.inject(current)        # be entirely
        else:                             # equivalent
            current.inject_into([neuron])
    neurons.record(['spikes', 'v'])

    sim.run(t_stop)

    spiketrains = neurons.get_data().segments[0].spiketrains
    assert_equal(len(spiketrains), n)
    assert_equal(len(spiketrains[0]), 0)  # first cell does not fire
    assert_equal(len(spiketrains[1]), 1)  # other cells fire once
    assert_equal(len(spiketrains[-1]), 1)  # other cells fire once
    expected_spike_times = t_start + tau_m * numpy.log(I * tau_m / (I * tau_m - v_thresh * cm))
    a = spike_times = [numpy.array(st)[0] for st in spiketrains[1:]]
    b = expected_spike_times[1:]
    max_error = abs((a - b) / b).max()
    print("max error =", max_error)
    assert max_error < 0.005, max_error
    sim.end()
    return a, b, spike_times

def issue511(sim):
    """Giving SpikeSourceArray an array of non-ordered spike times should produce an InvalidParameterValueError error"""
    sim.setup()
    celltype = sim.SpikeSourceArray(spike_times=[[2.4, 4.8, 6.6, 9.4], [3.5, 6.8, 9.6, 8.3]])
    assert_raises(InvalidParameterValueError, sim.Population, 2, celltype)

def test_reset(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    dt = 1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
    p.record('v')

    for i in range(repeats):
        sim.run(10.0)
        sim.reset()
    data = p.get_data(clear=False)
    sim.end()

    assert len(data.segments) == repeats
    for segment in data.segments[1:]:
        assert_array_almost_equal(segment.analogsignals[0],
                                  data.segments[0].analogsignals[0], 10)


def test_reset_with_clear(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    dt = 1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
    p.record('v')

    data = []
    for i in range(repeats):
        sim.run(10.0)
        data.append(p.get_data(clear=True))
        sim.reset()

    sim.end()

    for rec in data:
        assert len(rec.segments) == 1
        assert_arrays_almost_equal(rec.segments[0].analogsignals[0],
                                   data[0].segments[0].analogsignals[0], 1e-11)


def test_setup(sim):
    """
    Run the same simulation n times, recreating the network each time,
    and check the results are the same each time.
    """
    n = 3
    data = []
    dt = 1

    for i in range(n):
        sim.setup(timestep=dt, min_delay=dt)
        p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
        p.record('v')
        print('starting run ', i)
        sim.run(10.0)
        print('finished run ', i)

        print('start get_data run ', i)
        data.append(p.get_data())
        print('finished get_data run ', i)

        print('start sim end ', i)
        sim.end()
        print('finished sim end ', i)

    assert len(data) == n
    for block in data:
        assert len(block.segments) == 1
        signals = block.segments[0].analogsignals
        assert len(signals) == 1
        assert_array_equal(signals[0], data[0].segments[0].analogsignals[0])


def test_run_until(sim):
    sim.setup(timestep=0.1)
    p = sim.Population(1, sim.IF_cond_exp())
    sim.run_until(12.7)
    assert_almost_equal(sim.get_current_time(), 12.7, 10)
    sim.run_until(12.7)
    assert_almost_equal(sim.get_current_time(), 12.7, 10)
    sim.run_until(99.9)
    assert_almost_equal(sim.get_current_time(), 99.9, 10)
    assert_raises(ValueError, sim.run_until, 88.8)
    sim.end()


if __name__ == '__main__':
    sim = pynn_genn
    # test_setup(sim)
    # test_reset_with_clear(sim)
    # test_run_until(sim)
    # test_reset(sim)
    # scenario3(sim)
    scenario2(sim)
    # issue511(sim)
