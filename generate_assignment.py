import argparse

parser = argparse.ArgumentParser(
    description='Generate Experimental Assignments')
parser.add_argument('file',
                    help='subject ids and covariates as a csv file')
parser.add_argument('--dynamic', type=bool, default=False,
                    help='dynamic treatment assignment')
parser.add_argument('--output', type=str, default=None,
                    help='output assignment file, use original file if '
                         'unspecified')
parser.add_argument('--balance-objective', type=str, help='balance objective',
                    default='mahalanobis')
parser.add_argument('--K', type=int, help='number of rerandomizations',
                    default=None)
parser.add_argument('--target-quantile', type=int, default=None,
                    help='target balance quantile | best is 100')


if __name__ == '__main__':
    args = parser.parse_args()
