import warnings
from scripts import Network
import numpy as np
import matplotlib
from utils.utils import *  # EvaluationMetric, StopCriterion
import pylab as plt
import statistics
import os
import yaml
import multiprocessing
from scripts.processNetwork import runNetworkRE, runNetworkMM

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    matplotlib.use('agg')

    # current directory
    current = os.getcwd()

    _, filename_1, filename_2 = sys.argv  # directory of your data and setting file, name of your setting file
    dirname_1 = os.path.join(filename_1, filename_2)
    dirname_2 = filename_1  # os.path.join(current, filename_1)


    def plot_iteration(max_it, met_val):
        X = np.arange(1, max_it, 10)
        plt.plot(X, met_val)


    def complete_plot(m):
        plt.xlabel('Iteration')
        if m == EvaluationMetric.APS:
            plt.ylabel('Average Precision Score (APS)')
            plt.ylim(0, 1)
        elif m == EvaluationMetric.AUROC:
            plt.ylabel('Area Under ROC Curve')
            plt.ylim(0, 1)
        elif m == EvaluationMetric.RMSE:
            plt.ylabel('RMSE')


    def predict(num_iterations, th, network=None, rng=np.random.default_rng()):
        if network is None:
            network = Network(dirname_1, dirname_2, True, rng, mask=0)
        for i in range(num_iterations):
            network.update()
            print("iteration "+str(i)+", error =", network.get_error())

        rebuilt_association_matrix = np.linalg.multi_dot(
            [network.get_main().G_left, network.get_main().S, network.get_main().G_right.transpose()])
        new_relations_matrix = rebuilt_association_matrix - network.get_main().original_matrix
        n, m = new_relations_matrix.shape
        with open(f"{dirname_2}/results/results.txt", "w") as outF:
            for i in range(n):
                for j in range(m):
                    if new_relations_matrix[i, j] > th:
                        line = network.get_main().left_sorted_term_list[i] + "  "\
                               + network.get_main().right_sorted_term_list[j] + "  " + str(new_relations_matrix[i, j])
                        outF.write(line)
                        outF.write("\n")

    with open(dirname_1, 'r') as f:
        graph_topology = yaml.load(f, Loader=yaml.FullLoader)
        metric = EvaluationMetric(graph_topology["metric"].upper())
        stop_criterion = StopCriterion(graph_topology["stop.criterion"].upper())

        try:
            max_iter_value = graph_topology["number.of.iterations"]
            max_iter = int(max_iter_value)
            if max_iter > MAX_ITER:
                raise ValueError()
        except ValueError:
            max_iter = MAX_ITER
            print(f"Invalid number of iteration {max_iter_value}, set default value {MAX_ITER}")

        try:
            threshold = graph_topology["score.threshold"]
            threshold = float(threshold)
            if not (0 < threshold < 1):
                raise ValueError()
        except ValueError:
            print(f"Invalid threshold {threshold}, set default value {default_threshold}")
            threshold = default_threshold

        try:
            initialization = graph_topology["initialization"]
        except ValueError:
            print(f"No initialization method given")

    print("\nmetric :", metric.value)
    print("initialization : ", initialization)
    print(f"number of iterations : {max_iter}")
    print("stop criterion : ", stop_criterion.value)
    print("threshold : ", threshold)

    metric_vals = np.zeros(max_iter // 10)
    if stop_criterion == StopCriterion.MAXIMUM_METRIC:
        best_iter = 0
        # contains the iterations with the best performance from each of N_ITERATIONS validation runs (j cycle)
        best_iter_arr = []

        ss = np.random.SeedSequence()
        # Spawn off 10 child SeedSequences to pass to child processes.
        child_seeds = ss.spawn(N_ITERATIONS + 1)
        streams = [np.random.default_rng(s) for s in child_seeds]
        processes = list()
        results = multiprocessing.Array('i', range(N_ITERATIONS))
        metricsArr = multiprocessing.Array('d', range(N_ITERATIONS * (max_iter // 10)))
        for i in range(N_ITERATIONS):
            p = multiprocessing.Process(target=runNetworkMM, args=(
                [dirname_1, dirname_2, streams[i], metric, max_iter], results, metricsArr, i))
            p.start()
            processes.append(p)
        for i in range(N_ITERATIONS):
            processes[i].join()
            best_iter_arr.append(results[i])
            plot_iteration(max_iter, metricsArr[i * (max_iter // 10):(i + 1) * (max_iter // 10)])

        complete_plot(metric)

        res_best_iter = statistics.median(best_iter_arr)
        plt.axvline(x=res_best_iter, color='k', label='selected stop iteration', linestyle='dashed')
        plt.legend(loc=4)
        plt.savefig(f'{dirname_2}/results/{metric.value}_{graph_topology["initialization"]}_{stop_criterion.value}.png')
        plt.close("all")

        predict(res_best_iter, threshold, rng=streams[N_ITERATIONS])

    elif stop_criterion == StopCriterion.RELATIVE_ERROR:
        best_epsilon_arr = []
        ss = np.random.SeedSequence()
        # Spawn off 10 child SeedSequences to pass to child processes.
        # completamente indipendenti dato che usiamo numpy = 1.18 farò riferimento a
        # https://albertcthomas.github.io/good-practices-random-number-generators/
        # è necessitata SeedSequence spawing https://numpy.org/doc/1.18/reference/random/parallel.html
        # essa implementa un algoritmo che garantisce un'alta probabilità che due seed genrati vicini sian
        # molto diversi.
        # SeedSequence avoids these problems by using successions of integer hashes with good avalanche properties
        child_seeds = ss.spawn(N_ITERATIONS + 1)
        streams = [np.random.default_rng(s) for s in child_seeds]
        processes = list()
        results = multiprocessing.Array('i', range(N_ITERATIONS))
        metricsArr = multiprocessing.Array('d', range(N_ITERATIONS*(max_iter // 10)))
        for i in range(N_ITERATIONS):
            p = multiprocessing.Process(target=runNetworkRE, args=([dirname_1, dirname_2, streams[i], metric, max_iter], results, metricsArr, i))
            p.start()
            processes.append(p)
        for i in range(N_ITERATIONS):
            processes[i].join()
            best_epsilon_arr.append(results[i])
            plot_iteration(max_iter, metricsArr[i*(max_iter // 10):(i+1)*(max_iter // 10)])

        print("best_epsilon_arr: "+str(best_epsilon_arr))

        complete_plot(metric)

        res_best_epsilon = statistics.median(best_epsilon_arr)
        plt.axvline(x=res_best_epsilon, color='k', label='selected stop iteration', linestyle='dashed')
        plt.legend(loc=4)
        plt.savefig(f'{dirname_2}/results/{metric.value}_{graph_topology["initialization"]}_{stop_criterion.value}.png')
        plt.close("all")

        print("\nFinal run without masking, stop at iteration: " + str(res_best_epsilon))
        predict(res_best_epsilon, threshold, rng=streams[N_ITERATIONS])

    elif stop_criterion == StopCriterion.MAXIMUM_ITERATIONS:
        network = Network(dirname_1, dirname_2, True, np.random.default_rng(), mask=0)
        initial_error = network.get_error()
        print("initial error: {}".format(initial_error))
        print("\nUnique run of the algorithm without masking")
        predict(max_iter, threshold, network=network)


#%%
