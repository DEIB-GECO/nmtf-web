import warnings
import multiprocessing

warnings.filterwarnings('ignore')
import sys

#  sys.path.insert(0, 'scripts/')
from scripts.AssociationMatrix import AssociationMatrix, EvaluationMetric
import numpy as np
import os
import yaml
from contextlib import contextmanager
import sys
from scipy import linalg as la
from sklearn.cluster import KMeans


class Network:
    """
    The Network class is the representation of all the graph
    And it contains a instance of AssociationMatrix for each
    file (every AssociationMatrix contains the decomposition)

    Attributes
    ----------
    graph_topology_file : str
        File name describing the topology of the network (setting file).
    init_strategy : str
        Strategy of initialization of the AssociationMatrix in the network. Strategy can be: random, svd, kmeans.
    integration_strategy : str
        Integration strategy of nodes. Strategy can be: union, intersection.
    rng : Generator
        Random generator
    type_of_masking : int
        How mask is generated. 0 means fully_random, 1 means random_per_row.
    association_matrices : List[AssociationMatrix]
        Association matrix of the Network
    datasets : Set[str]
        Set with names of all data sets
    dataset_ks : Dict[str, int]
        Dictionary that associate data set name with its rank
    files : Dict[str, list]
        Dictionary that associate to every filename the list [main, ds_left, ds_right]. Main is 1 if is the main matrix,
        0 otherwise. ds_left is the dataset on the left of the relationship and ds_right is the dataset on the right.
    """

    def __init__(self, graph_topology_file, dirfilename, verbose, rng, mask=1):
        """
        Constructor of the matrix

        Parameters
        ----------
        graph_topology_file : str
            File name describing the topology of the network (setting file).
        dirfilename : str
            Directory where association files are present
        verbose : bool
            True to show logs
        rng : Generator
            Random generator
        mask : int, default=1
            Parameter passed to AssociationMatrix
        """
        self.graph_topology_file = graph_topology_file
        self.init_strategy = "random"
        self.integration_strategy = lambda x, y: x.intersection(y)
        self.rng = rng
        self.type_of_masking = 1

        self.association_matrices = []
        self.datasets = {}
        self.dataset_ks = {}
        self.files = {}

        with open(self.graph_topology_file) as f:
            graph_topology = yaml.load(f, Loader=yaml.FullLoader)

            integration_strategy = graph_topology["integration.strategy"]
            if integration_strategy == "union":
                self.integration_strategy = lambda x, y: x.union(y)
            elif integration_strategy == "intersection":
                self.integration_strategy = lambda x, y: x.intersection(y)
            else:
                print("Option '{}' not supported".format(integration_strategy))
                exit(-1)

            initialization = graph_topology["initialization"]
            if (initialization == "random") or (initialization == "kmeans") or (initialization == "svd"):
                self.init_strategy = initialization
            else:
                print("Option '{}' not supported".format(initialization))
                exit(-1)
            if verbose:
                print("Initialization strategy is " + self.init_strategy + "\n")

            type_of_masking = graph_topology["type.of.masking"]
            if type_of_masking == "fully_random":
                self.type_of_masking = 0
            elif type_of_masking == "per_row_random":
                self.type_of_masking = 1
            else:
                print("Option '{}' not supported".format(type_of_masking))
                exit(-1)

            if "ranks" in graph_topology:
                for data in graph_topology["ranks"]:
                    dsname = data["dsname"]
                    k = data["k"]
                    self.dataset_ks[dsname.upper()] = int(k)
            if "k_svd" in graph_topology:
                self.k_svd = graph_topology["k_svd"]

            # For each category of nodes, compute the intersection or union between the different matrices
            for element in graph_topology["graph.datasets"]:
                filename = element["filename"]
                filename = os.path.join(dirfilename, filename)
                ds1_name = element["nodes.left"].upper()
                ds2_name = element["nodes.right"].upper()
                main = int(element["main"])
                ds1_entities = set()
                ds2_entities = set()

                with open(filename) as af:
                    for edge in af:
                        s_edge = edge.strip().split("\t")
                        ds1_entities.add(s_edge[0])
                        ds2_entities.add(s_edge[1])

                self.files[filename] = [main, ds1_name, ds2_name]

                if ds1_name in self.datasets:
                    self.datasets[ds1_name] = self.integration_strategy(self.datasets[ds1_name], ds1_entities)
                else:
                    self.datasets[ds1_name] = ds1_entities
                if ds2_name in self.datasets:
                    self.datasets[ds2_name] = self.integration_strategy(self.datasets[ds2_name], ds2_entities)
                else:
                    self.datasets[ds2_name] = ds2_entities

        # sort the nodes, such that each matrix receives the same ordered list of nodes
        for k in self.datasets.keys():
            self.datasets[k] = list(sorted(list(self.datasets[k])))

        if verbose:
            print('All specified nodes\' categories: ' + "{}".format(
                str(list(self.datasets.keys()))) + "\n")

        for file in self.files.keys():
            self.association_matrices.append(
                AssociationMatrix(file,                                 # file = files.key = filename
                                  self.files[file][1],                  # files[file][1] = ds1_name
                                  self.files[file][2],                  # file[file][2] = ds2_name
                                  self.datasets[self.files[file][1]],
                                  self.datasets[self.files[file][2]],
                                  self.files[file][0],                  # files[file][0] = main
                                  self.rng,
                                  mask,
                                  self.type_of_masking,
                                  verbose))

        for m1 in self.association_matrices:
            for m2 in self.association_matrices:
                if m1 != m2:
                    if m1.leftds == m2.leftds:
                        m1.dep_own_left_other_left.append(m2)
                    elif m1.rightds == m2.rightds:
                        m1.dep_own_right_other_right.append(m2)
                    elif m1.rightds == m2.leftds:
                        m1.dep_own_right_other_left.append(m2)
                    elif m1.leftds == m2.rightds:
                        m1.dep_own_left_other_right.append(m2)

        # This part is setting the ranks for the different AssociationMatrix
        # ( remember am = G_left^{M,k1} @ S^{k1,k2} @ G_right^{k2,N} )
        # NOT CLEAR how k is used
        for k in self.datasets.keys():
            rank = self.select_rank(k)
            for am in self.association_matrices:
                if am.leftds == k:
                    am.k1 = int(rank)
                elif am.rightds == k:
                    am.k2 = int(rank)

        # self.association_matrices = list(self.pool.starmap(initialize, [(am, self.init_strategy, False) for am in self.association_matrices]))
        """processes = list()
        results = multiprocessing.Manager().list()
        for am in self.association_matrices:
            p = multiprocessing.Process(target=initialize, args=(am, self.init_strategy, False, results))
            p.start()
            processes.append(p)
        for i in range(len(self.association_matrices)):
            processes[i].join()
            self.association_matrices[i] = results[i]"""

        for am in self.association_matrices:
            am.initialize(self.init_strategy, verbose)

        """for am in self.association_matrices:
            am.create_update_rules()"""

    # Method to calculate rank for each datatype. In case of k-means and spherical k-means initialization represents
    # number of clusters. TODO: Besides in case of svd rank represents the "compression" magnitude
    def select_rank(self, ds_name):
        """
        Method to calculate rank for each datatype. In case of k-means and spherical k-means initialization represents
        number of clusters. In case of svd rank represents the "compression" magnitude.

        Parameters
        ----------
        ds_name : str
            name of the data set
        Returns
        -------
        int
            rank
        """
        # TODO: rank set by the user
        if ds_name in self.dataset_ks:
            rank = self.dataset_ks[ds_name]
        else:
            if self.init_strategy == "kmeans":
                el_num = len(self.datasets[ds_name])
                if el_num > 200:
                    el_num = int(el_num / 5)
                # Rank should be less than the number of unique elements in rows and in columns of any matrix where
                # datatype is present
                for am in self.association_matrices:
                    if am.leftds == ds_name:
                        el_num = min([el_num, np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=1), 0)),
                                      np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=0), 1)),
                                      len(self.datasets[am.rightds])])
                    elif am.rightds == ds_name:
                        el_num = min([el_num, np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=1), 0)),
                                      np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=0), 1)),
                                      len(self.datasets[am.leftds])])
                rank = el_num
            elif self.init_strategy == "random":
                rank = 100
            elif self.init_strategy == "svd":
                rank = self.k_svd
            else:
                print("select_rank error: rank set to -1. Rank initialization not supported for " +
                      self.init_strategy + ".\n")
                rank = -1
        return rank

    def get_error(self):
        """
        Method to get the current error of the network

        Returns
        -------
        float :
            current error of the network
        """
        # print(sum([am.get_error() for am in self.association_matrices]))
        return sum([am.get_error() for am in self.association_matrices])

    def update(self):
        """
        Update the network
        """
        for am in self.association_matrices:
            am.update()

    def validate(self, metric=EvaluationMetric.APS):
        """
        Validate the network, producing performance metrics (APS, AUROC, RMSE)

        Parameters
        ----------
        metric : EvaluationMetric, default=EvaluationMetric.APS
            Metric to produce

        Returns
        -------
        Method to produce performance metrics
        """
        for am in self.association_matrices:
            if am.main == 1 and am.validation == 1:
                return am.validate(metric)

    def get_main(self):
        """
        Get the main matrix of the network

        Returns
        -------
        AssociationMatrix
            main matrix
        """
        for am in self.association_matrices:
            if am.main == 1:
                return am
