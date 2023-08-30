from scripts.Network import *
import numpy as np
import time
from utils.utils import *


# args: (dirname_1, dirname_2, rng, metric, max_iter)
def runNetworkMM(args, results, metricsArr, pos):
    """
        creates a Network with Maximum Metrix evaluation

        Parameters
        ----------
        args : (dirname_1, dirname_2, rng, metric, max_iter)
            args necessary to the Network to be created:
        results
            shared array containing the best epsilons
        metricsArr
            shared array containing the Metric samples every 10 iterations
        pos : int
            integer identifying the run
    """
    max_i = args[4]
    met = args[3]
    V = []
    network = Network(args[0], args[1], False, args[2])
    metric_vals = np.zeros(max_i // 10)

    initial_error = network.get_error()
    start = time.time()
    print("PID:", pos, "- initial error:", initial_error)
    startIteration = time.time()
    for i in range(max_i):
        network.update()
        if i % 10 == 0:
            metric_vals[i // 10] = network.validate(met)
        V.append(network.validate(met))
        if (i+1) % 10 == 0:
            endIteration = time.time()
            print(f"PID: {pos} - iteration {i + 1},\t{met.value} = {V[-1]},\ttime = {endIteration - startIteration}")
            startIteration = time.time()
    end = time.time()
    print("Total time: ", end - start)
    results[pos] = (V.index(min(V)) if met == EvaluationMetric.RMSE else V.index(max(V)))
    len_m = max_i // 10
    for i in range(len_m):
        metricsArr[pos*len_m + i] = metric_vals[i]


# args: (dirname_1, dirname_2, rng, metric, max_iter)
def runNetworkRE(args, results, metricsArr, pos):
    """
        creates a Network with Relative Error evaluation

        Parameters
        ----------
        args : (dirname_1, dirname_2, rng, metric, max_iter)
            args necessary to the Network to be created:
        results
            shared array containing the best epsilons
        metricsArr
            shared array containing the Metric samples every 10 iterations
        pos : int
            integer identifying the run
    """
    epsilon = 0
    max_i = args[4]
    met = args[3]
    error = []
    V = []
    metric_vals = np.zeros(max_i // 10)
    # mask is set by default at 1
    network = Network(args[0], args[1], False, args[2])

    initial_error = network.get_error()
    # print("\nRun number " + str(j + 1) + " of the algorithm")
    start = time.time()
    print("PID:", pos, "- initial error:", initial_error)
    eps_iter = []
    startIteration = time.time()
    for i in range(max_i):
        network.update()
        error.append(network.get_error())
        V.append(network.validate(met))
        if i % 10 == 0:
            metric_vals[i // 10] = network.validate(met)
        if i > 1:
            epsilon = abs((error[-1] - error[-2]) / error[-2])
            if epsilon < 0.001:
                eps_iter.append(i)
        if (i+1) % 10 == 0:
            endIteration = time.time()
            print(f"PID: {pos} - iteration {i + 1},\trelative error = {epsilon},\ttime = {endIteration - startIteration}")
            startIteration = time.time()
    end = time.time()
    print("PID:", pos, "- Total time: ", end - start)
    results[pos] = eps_iter[0]
    len_m = max_i // 10
    for i in range(len_m):
        metricsArr[pos*len_m + i] = metric_vals[i]
