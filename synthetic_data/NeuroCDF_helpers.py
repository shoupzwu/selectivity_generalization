import copy
import random
import numpy as np
import torch

random.seed(42)
candi_points = np.linspace(0, 1, 51, endpoint=False)[1:]

def query2cdfs(query, num_cols):
	# query: a list of [col, op, val], col should be the id
	# col: column index
	# val: normalized value
	queried_cols = []
	query_range = []
	upper_list = []
	lower_list = []

	for _ in range(num_cols):
		query_range.append([0., 1.])
		upper_list.append(1.)
		lower_list.append(0.)

	for col, op, val in query:
		if op == '<=':
			query_range[col][1] = val
			upper_list[col] = val
		elif op == '>=':
			query_range[col][0] = val
			lower_list[col] = val

		if col not in queried_cols:
			queried_cols.append(col)

	cdfs = [[]]
	lower_counts = [0]
	signs = []

	for i in range(num_cols):
		if lower_list[i] == 0. and upper_list[i] == 1.:
			for cdf in cdfs:
				cdf.append(1.)
		else:
			if upper_list[i] != 1.:
				for cdf in cdfs:
					cdf.append(upper_list[i])
			else:
				cdfs_copy = copy.deepcopy(cdfs)
				lower_counts_copy = copy.deepcopy(lower_counts)

				for cdf in cdfs:
					cdf.append(lower_list[i])
				for j in range(len(lower_counts)):
					lower_counts[j] += 1

				for cdf in cdfs_copy:
					cdf.append(1.)

				cdfs.extend(cdfs_copy)
				lower_counts.extend(lower_counts_copy)

	for count in lower_counts:
		signs.append((-1.) ** count)

	return cdfs, signs

def prepare_batch(queries_batch, num_cols, encoding_size):
	max_num_cdfs = 0
	batch_cdfs = []
	batch_signs = []
	for query in queries_batch:
		cdfs, signs = query2cdfs(query, num_cols)
		batch_cdfs.append(cdfs)
		batch_signs.append(signs)

		if len(cdfs) > max_num_cdfs:
			max_num_cdfs = len(cdfs)

	for cdfs, signs in zip(batch_cdfs, batch_signs):
		if len(cdfs) < max_num_cdfs:
			for _ in range(max_num_cdfs - len(cdfs)):
				cdfs.append([0.] * (encoding_size))
				signs.append(0.)

	return batch_cdfs, batch_signs, max_num_cdfs
