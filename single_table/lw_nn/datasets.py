import os

import numpy as np

import lw_nn.common as common

def LoadCensus(filename='census.csv'):
	csv_file = './datasets/{}'.format(filename)
	cols = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	#       [0, 1, 1, 0, 1, 1, 1, 1, 1, 0,  0, 0,  1,  1]
	type_casts = {}
	return common.CsvTable('Adult', csv_file, cols, type_casts, header=None)
