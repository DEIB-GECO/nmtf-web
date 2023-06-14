import multiprocessing
import warnings

warnings.filterwarnings('ignore')
import sys
# import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from scipy import stats as stats
from scipy import linalg as la
import copy

from contextlib import contextmanager
import os
from utils import EvaluationMetric
from scripts.processAssociationMatrix import *

import math


# poolAM = multiprocessing.Pool(5)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Function to parse edges in association files
def parse_line(line):
    s = line.strip().split("\t")
    # if only names are written than weight of connection is set to 1
    if len(s) == 2:
        return [s[0], [s[1], 1]]
    elif len(s) == 3:
        return [s[0], [s[1], s[3]]]
    else:
        raise Exception()


def list_to_dict(list):
    dic = dict()
    i = 0
    for el in list:
        dic[el] = i
        i += 1
    return dic


class AssociationMatrix:

    def __init__(self, filename, leftds, rightds, left_sorted_terms, right_sorted_terms, main, rng,
                 mask, type_of_masking, verbose):

        self.M = None
        self.filename = filename
        self.safe_filename = filename.rsplit('/')[-1]
        self.leftds = leftds
        self.rightds = rightds
        self.intra_data_matrices = []
        self.dep_own_right_other_right = []
        self.dep_own_right_other_left = []
        self.dep_own_left_other_right = []
        self.dep_own_left_other_left = []
        self.left_sorted_term_list = left_sorted_terms
        self.right_sorted_term_list = right_sorted_terms
        self.k1 = 0
        self.k2 = 0
        self.main = main
        self.rightds_intra = None
        self.leftds_intra = None
        self.rng = rng
        self.validation = mask
        if self.main == 1:
            self.type_of_masking = type_of_masking

        with open(self.filename, "r") as f:
            data_graph = [parse_line(element) for element in f.readlines()]
        self.edges = dict()
        for el in self.left_sorted_term_list:
            self.edges[el] = list()
        setleft = set(self.left_sorted_term_list)
        setright = set(self.right_sorted_term_list)
        for els in data_graph:
            if els[0] in setleft and els[1][0] in setright:
                self.edges[els[0]].append(els[1])

        len_terms_left = len(self.left_sorted_term_list)
        len_terms_right = len(self.right_sorted_term_list)
        right_sorted_dict = list_to_dict(self.right_sorted_term_list)
        ass_mat = np.zeros((len_terms_left, len_terms_right))
        i = 0
        for eleft in self.left_sorted_term_list:
            for eright in self.edges[eleft]:
                ass_mat[i][right_sorted_dict[eright[0]]] = eright[1]
            i += 1
        self.association_matrix = ass_mat

        self.original_matrix = copy.deepcopy(self.association_matrix)  # for all to use in select_rank
        if self.main == 1 and self.validation == 1:  # this is the matrix which we try to investigate
            self.mask_matrix()
        self.G_left = None
        self.G_left_primary = False
        self.G_right = None
        self.G_right_primary = False
        self.S = None

    def initialize(self, initialize_strategy, verbose):
        if initialize_strategy == "random":
            if verbose:
                print("Association matrix filename: " + self.safe_filename)
                print("Used parameters: k1 = " + str(self.k1) + " and" + " k2 = " + str(
                    self.k2))
                print("Non-zero elements of the association matrix = {}".format(
                    np.count_nonzero(self.association_matrix)))
            if self.G_left is None:
                self.G_left = self.rng.random((self.association_matrix.shape[0], self.k1))
                self.G_left_primary = True
            if self.G_right is None:
                self.G_right = self.rng.random((self.association_matrix.shape[1], self.k2))
                self.G_right_primary = True
            self.S = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])
        elif initialize_strategy == "kmeans":
            if verbose:
                print("Association matrix filename: " + self.safe_filename)
                print("Used parameters: k1 = " + str(self.k1) + " and k2 = " + str(
                    self.k2))
                print("Non-zero elements of the association matrix = {}".format(
                    np.count_nonzero(self.association_matrix)))
            if self.G_left is None:
                with suppress_stdout():
                    self.G_left = KMeans(n_clusters=self.k1, algorithm='full').fit_transform(self.association_matrix)
                    self.G_left_primary = True
            if self.G_right is None:
                with suppress_stdout():  # TODO: full is lloyd nelle versioni successive
                    self.G_right = KMeans(n_clusters=self.k2, algorithm='full').fit_transform(
                        self.association_matrix.transpose())
                    self.G_right_primary = True
            self.S = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])
        # TODO: added svd initialization
        elif initialize_strategy == "svd":
            # full_matrices is false because we want an approximation so la.svd(...) will compute only
            # leading eigenvalues TODO: è ok?
            # TODO: sarebbe da calcolare il tempo di
            u, s, vh = la.svd(self.association_matrix, full_matrices=False)
            # TODO
            k_svd = min(self.k1, len(s))
            self.k1 = k_svd
            self.k2 = k_svd
            if verbose:
                print("Association matrix filename: " + self.safe_filename)
                print("Used parameters: k1 = " + str(self.k1) + " and k2 = " + str(
                    self.k2))
                print("Non-zero elements of the association matrix = {}".format(
                    np.count_nonzero(self.association_matrix)))
            if self.G_left is None:
                with suppress_stdout():
                    # reduction of G_left
                    self.G_left = u[:self.association_matrix.shape[0], :self.k1]
                    # all negative values set to 0
                    for i in range(len(self.G_left)):
                        for j in range(self.k1):
                            if math.isnan(self.G_left[i][j]):
                                input("Nan in G_left " + self.filename + " >")
                            if self.G_left[i][j] < 0:
                                self.G_left[i][j] = 0
                    self.G_left_primary = True  # TODO: a cosa serve questa variabile?

            if self.G_right is None:
                with suppress_stdout():
                    # reduction of G_right. Need to transpose to stay consistent with other initialization methods
                    self.G_right = vh[:self.k2, :self.association_matrix.shape[1]].transpose()
                    # all negative values set to 0
                    for i in range(len(self.G_right)):
                        for j in range(self.k2):
                            if math.isnan(self.G_right[i][j]):
                                input("Nan in G_right " + self.filename + " >")
                            if self.G_right[i][j] < 0:
                                self.G_right[i][j] = 0
                    self.G_right_primary = True
            # s became an array with all eigenvalues in it
            s = s[:self.k1]
            # Need to discard all negative eigenvalues
            for i in range(self.k1):
                if math.isnan(s[i]):
                    input("Nan in S " + self.filename + " >")
                if s[i] < 0:
                    s[i] = 0
            self.S = np.diag(s)  # S became a diagonal matrix

        for am in self.dep_own_left_other_left:
            if am.G_left is None:
                am.G_left = self.G_left
        for am in self.dep_own_left_other_right:
            if am.G_right is None:
                am.G_right = self.G_left
        for am in self.dep_own_right_other_left:
            if am.G_left is None:
                am.G_left = self.G_right
        for am in self.dep_own_right_other_right:
            if am.G_right is None:
                am.G_right = self.G_right
        if verbose:
            print(self.leftds, self.rightds, self.association_matrix.shape)
            print("Shape Factor Matrix left " + str(self.G_left.shape))
            print("Shape Factor Matrix right " + str(self.G_right.shape) + "\n")

    # Method to mask the matrix. Used in validation phase to be able later in validate method to assess performance of
    # the algorithm. self.type_of_masking is set in the settings file and read in the method open of the class network.
    # This parameter can be "fully_random"(0) or "per_row_random"(1). In the first case masking elements in the created
    # mask are distributed uniformly randomly and in the second case mask has same number of masking elements per row,
    # distributed randomly within the row.
    def mask_matrix(self):
        self.M = np.zeros_like(self.association_matrix)
        if self.type_of_masking == 0:
            a = np.ones(self.association_matrix.shape, dtype=self.association_matrix.dtype)
            n = self.association_matrix.size * 0.05
            a = a.reshape(a.size)
            a[:int(n)] = 0
            self.rng.shuffle(a)
            a = a.reshape(self.association_matrix.shape)
            self.association_matrix = np.multiply(self.association_matrix, a)
            self.M = a
        else:
            for i in range(0, self.association_matrix.shape[0] - 1):
                nc = self.association_matrix.shape[1]  # nc is row size ( number of columns)
                a = np.ones(nc, dtype=int)  # get array of dimension of 1 row
                n = self.association_matrix.shape[1] * 0.05
                a[:int(n)] = 0
                self.rng.shuffle(a)
                self.association_matrix[i, :] = np.multiply(self.association_matrix[i, :], a)
                self.M[i, :] = a

    # Method to produce performance metrics (APS, AUROC). Produces output only if the matrix is the matrix for which
    # predictions are searched and the network is in validation mode.
    def validate(self, metric=EvaluationMetric.APS):
        if self.main == 1 and self.validation == 1:
            self.rebuilt_association_matrix = np.linalg.multi_dot([self.G_left, self.S, self.G_right.transpose()])

            R12_2 = list(self.original_matrix[self.M == 0])
            R12_found_2 = list(self.rebuilt_association_matrix[self.M == 0])
            # TODO: R12_found_2 è tutti nan in svd generando un errore (errore vecchio)
            if metric == EvaluationMetric.AUROC:
                fpr, tpr, _ = metrics.roc_curve(R12_2, R12_found_2)
                return metrics.auc(fpr, tpr)
            elif metric == EvaluationMetric.APS:
                return metrics.average_precision_score(R12_2, R12_found_2)
            elif metric == EvaluationMetric.RMSE:
                return (metrics.mean_squared_error(R12_2, R12_found_2)) ** (.5)
            elif metric == EvaluationMetric.LOG_RMSE:
                return np.log10((metrics.mean_squared_error(R12_2, R12_found_2)) ** (.5))
            elif metric == EvaluationMetric.PEARSON:
                return stats.pearsonr(R12_2, R12_found_2)[0]

    def get_error(self):
        self.rebuilt_association_matrix = np.linalg.multi_dot([self.G_left, self.S, self.G_right.transpose()])
        # print(self.rebuilt_association_matrix) # TODO: rebuilt_association_matrix a  volte è nan
        return np.linalg.norm(self.association_matrix - self.rebuilt_association_matrix, ord='fro') ** 2

    def update_G_right(self):
        num = np.linalg.multi_dot([self.association_matrix.transpose(), self.G_left, self.S])
        den = np.linalg.multi_dot(
            [self.G_right, self.S.transpose(), self.G_left.transpose(), self.G_left, self.S])
        for am in self.dep_own_right_other_right:
            num += np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
            den += np.linalg.multi_dot(
                [self.G_right, am.S.transpose(), am.G_left.transpose(), am.G_left, am.S])
        for am in self.dep_own_right_other_left:
            num += np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
            den += np.linalg.multi_dot(
                [self.G_right, am.S, am.G_right.transpose(), am.G_right, am.S.transpose()])
        div = np.divide(num, den + 0.000001)  #+ sys.float_info.min) # TODO: per questo si generava l'errore
        return np.multiply(self.G_right, div)

    def update_G_left(self):
        num = np.linalg.multi_dot([self.association_matrix, self.G_right, self.S.transpose()])
        den = np.linalg.multi_dot([self.G_left, self.S, self.G_right.transpose(), self.G_right, self.S.transpose()])

        for am in self.dep_own_left_other_left:
            num += np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
            den += np.linalg.multi_dot(
                [self.G_left, am.S, am.G_right.transpose(), am.G_right, am.S.transpose()])
        for am in self.dep_own_left_other_right:
            num += np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
            den += np.linalg.multi_dot(
                [self.G_left, am.S.transpose(), am.G_left.transpose(), am.G_left, am.S])
        # print(den)
        div = np.divide(num, den + 0.000001)  # TODO: secondo me il problema è qui dove si fa un 0/0
        # print(div)
        return np.multiply(self.G_left, div)

    def update_S(self):
        num = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])
        den = np.linalg.multi_dot(
            [self.G_left.transpose(), self.G_left, self.S, self.G_right.transpose(), self.G_right])
        div = np.divide(num, den + 0.000001)
        return np.multiply(self.S, div)

    def update(self):
        if self.G_right_primary:
            self.G_right = self.update_G_right()  # update_G_right(self)  #
        if self.G_left_primary:
            self.G_left = self.update_G_left()  # update_G_left(self)  #
        self.S = self.update_S()  # update_S(self)  #

