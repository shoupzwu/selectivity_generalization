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
				cdf.extend([0, 1.])
		else:
			if upper_list[i] != 1.:
				for cdf in cdfs:
					cdf.extend([0, upper_list[i]])
			else:
				cdfs_copy = copy.deepcopy(cdfs)
				lower_counts_copy = copy.deepcopy(lower_counts)

				for cdf in cdfs:
					cdf.extend([0, lower_list[i]])
				for j in range(len(lower_counts)):
					lower_counts[j] += 1

				for cdf in cdfs_copy:
					cdf.extend([0, 1.])

				cdfs.extend(cdfs_copy)
				lower_counts.extend(lower_counts_copy)

	for count in lower_counts:
		signs.append((-1.) ** count)

	return cdfs, signs

def prepare_batch(queries_batch, num_cols):
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
				cdfs.append([0.] * (2*num_cols))
				signs.append(0.)

	return batch_cdfs, batch_signs, max_num_cdfs

def generate_query(num_dim):
	query = []
	interval = []
	lower = [0.] * num_dim
	higher = [1.] * num_dim

	num_dim_queried = random.randint(1, 6)
	dims_queried = random.sample(list(range(num_dim)), num_dim_queried)

	for dim in dims_queried:
		value = random.choice(candi_points)
		op = random.choice([0, 1])

		if op == 0:
			query.append((dim, '>=', value))
			lower[dim] = value
			higher[dim] = 1.
		else:
			query.append((dim, '<=', value))
			lower[dim] = 0.
			higher[dim] = value

	interval.append(lower)
	interval.append(higher)

	return query, interval

def get_consistency_pairs(num_queries, num_dim, model):
	queries = []
	intervals = []
	for _ in range(num_queries):
		query, interval_tmp = generate_query(num_dim)
		interval =  []

		for i in range(len(interval_tmp[0])):
			interval.extend([interval_tmp[0][i], interval_tmp[1][i]])

		queries.append(query)
		intervals.append(interval)

	batch_queries_tensor = np.stack(intervals, axis=0)
	batch_queries_tensor = torch.FloatTensor(batch_queries_tensor)

	outputs = model(batch_queries_tensor)
	sel_predicts = torch.squeeze(outputs)

	### get cdfs!
	batch_cdfs, batch_signs, max_num_cdfs = prepare_batch(queries, num_dim)

	batch_cdfs_tensor = np.stack(batch_cdfs, axis=0)
	batch_cdfs_tensor = torch.FloatTensor(batch_cdfs_tensor)

	batch_signs_tensor = np.stack(batch_signs, axis=0)
	batch_signs_tensor = torch.FloatTensor(batch_signs_tensor)

	batch_cdfs_tensor = batch_cdfs_tensor.view(-1, 2 * num_dim)
	cdf_outputs = model(batch_cdfs_tensor)  # [bs*max_num_cdfs, num_dims]

	cdf_sel_predicts = torch.squeeze(cdf_outputs)
	cdf_sel_predicts = cdf_sel_predicts.view(-1, max_num_cdfs)  # [bs, max_num_cdfs]
	cdf_sel_predicts = torch.sum(cdf_sel_predicts * batch_signs_tensor, dim=-1)

	return sel_predicts, cdf_sel_predicts