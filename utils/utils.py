import sys
from enum import Enum


def bold(s):
    return f'\033[1m {s} \033[0m'


class EvaluationMetric(Enum):
    APS = 'APS'
    AUROC = 'AUROC'
    PEARSON = 'PEARSON'
    RMSE = 'RMSE'
    LOG_RMSE = 'LOG_RMSE'

    @classmethod
    def _missing_(cls, value):
        print(f'Invalid metric option <{value}>, switched to default RMSE',
              file=sys.stderr)
        return EvaluationMetric.RMSE


class StopCriterion(Enum):
    MAXIMUM_METRIC = 'MAXIMUM_METRIC'
    MAXIMUM_ITERATIONS = 'MAXIMUM_ITERATIONS'
    RELATIVE_ERROR = 'RELATIVE_ERROR'

    @classmethod
    def _missing_(cls, value):
        print(f'Invalid metric option <{value}>, switched to default MAXIMUM_METRIC',
              file=sys.stderr)
        return StopCriterion.MAXIMUM_METRIC


# Baseline parameters
default_threshold = 0.1
threshold = default_threshold
metric = 'aps'
MAX_ITER = 300
stop_criterion = 'calculate'
# number of iterations to find the stop criterion value
N_ITERATIONS = 5
