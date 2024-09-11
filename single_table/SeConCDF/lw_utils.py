import random
import numpy as np
import copy
import torch
import csv
from lw_nn.utils import *

seed = 42
random.seed(seed)

dataset_name = 'census'
file_name_column_min_max_vals = "./data/column_min_max_vals_census.csv"


with open(file_name_column_min_max_vals, 'rU') as f:
	data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
	column_min_max_vals = {}
	for i, row in enumerate(data_raw):
		if i == 0:
			continue
		column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

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
		col_name = cols[col]
		if op == '<=':
			query_range[col][1] = val
			upper_list[col] = val
		elif op == '>=':
			query_range[col][0] = val
			lower_list[col] = val
		elif op == '=':
			if val > 0:
				query_range[col][0] = find_next_var(unnormalize_var(val, col_name), col_name, 'down')
				query_range[col][1] = val
				lower_list[col] = find_next_var(unnormalize_var(val, col_name), col_name, 'down')
				upper_list[col] = val
			else:
				query_range[col][0] = val
				query_range[col][1] = find_next_var(unnormalize_var(val, col_name), col_name, 'up')
				lower_list[col] = val
				upper_list[col] = find_next_var(unnormalize_var(val, col_name), col_name, 'up')

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
			if upper_list[i] != 1. and lower_list[i] == 0.:
				for cdf in cdfs:
					cdf.extend([0, upper_list[i]])
			elif upper_list[i] == 1. and lower_list[i] != 0.:
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
			else:
				cdfs_copy = copy.deepcopy(cdfs)
				lower_counts_copy = copy.deepcopy(lower_counts)

				for cdf in cdfs:
					cdf.extend([0, lower_list[i]])
				for j in range(len(lower_counts)):
					lower_counts[j] += 1

				for cdf in cdfs_copy:
					cdf.extend([0, upper_list[i]])

				cdfs.extend(cdfs_copy)
				lower_counts.extend(lower_counts_copy)

	for count in lower_counts:
		signs.append((-1.) ** count)
	return cdfs, signs

def load_queries(file_name):
	tables, joins, filters = [], [], []
	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		for qid, row in enumerate(data_raw):
			q_filters = []
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))

			filter_items = row[2].split(',')

			if len(filter_items) < 3:
				q_filters.append('')
			else:
				for pid in range(int(len(filter_items)/3)):
					col = filter_items[pid*3]
					op = filter_items[pid * 3+1]
					val = filter_items[pid * 3+2]

					if op == '<':
						op = '<='
						if isinstance(val, str):
							val = str(int(val) - 1)
						else:
							val = int(val) - 1
					elif op == '>':
						op = '>='
						if isinstance(val, str):
							val = str(int(val) + 1)
						else:
							val = int(val) + 1
					q_filters.append([col, op, str(val)])

			filters.append(q_filters)

	return tables, joins, filters

def generate_query(num_dim, col_to_valid_list):
	query = []
	interval = []
	lower = [0.] * num_dim
	higher = [1.] * num_dim

	num_dim_queried = random.randint(1, 8)
	dims_queried = random.sample(list(range(num_dim)), num_dim_queried)

	for dim in dims_queried:
		col_name = cols[dim]

		if col_name in num_min_max:
			var = random.randint(num_min_max[col_name][0], num_min_max[col_name][1])
			nor_var = normalize_var(var, col_name)

			op = random.choice(['>=', '<='])
			if op == '>=':
				query.append((dim, '>=', nor_var))
				lower[dim] = nor_var
				higher[dim] = 1.
			elif op == '<=':
				query.append((dim, '<=', nor_var))
				lower[dim] = 0.
				higher[dim] = nor_var
			else:
				query.append((dim, '=', nor_var))

				if nor_var > 0:
					higher[dim] = nor_var
					lower[dim] = find_next_var(var, col_name, direction='down')
				else:
					print('wait!')
					higher[dim] = find_next_var(var, col_name, direction='up')
					lower[dim] = nor_var
		else:
			candi_points = col_to_valid_list[col_name]
			var = random.choice(list(candi_points))
			nor_var = normalize_var(var, col_name)

			query.append((dim, '=', nor_var))
			if nor_var > 0:
				higher[dim] = nor_var
				lower[dim] = find_next_var(var, col_name, direction='down')
			else:
				higher[dim] = find_next_var(var, col_name, direction='up')
				lower[dim] = nor_var

	interval.append(lower)
	interval.append(higher)

	return query, interval


def get_consistency_pairs(num_queries, num_dim, col_to_valid_list, model):
	queries = []
	intervals = []
	for _ in range(num_queries):
		query, interval_tmp = generate_query(num_dim, col_to_valid_list)
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