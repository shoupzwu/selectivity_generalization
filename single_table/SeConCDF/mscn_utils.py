import random
import numpy as np
import copy
import csv
from mscn.utils import *
from mscn.data import make_aug_dataset, make_aug_dataset_cdf

seed = 42
random.seed(seed)
file_name_column_min_max_vals = "./data/column_min_max_vals_census.csv"

with open(file_name_column_min_max_vals, 'rU') as f:
	data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
	column_min_max_vals = {}
	for i, row in enumerate(data_raw):
		if i == 0:
			continue
		column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

def sample_queries(num_queries, column_min_max_vals, max_num_preds=5):

	filters = []
	num_generated = 0

	while True:
		if num_generated >= num_queries:
			break

		num_filters =  random.randint(1, max_num_preds)
		sampled_cols = random.sample(list(column_min_max_vals.keys()), num_filters)

		q_filters = []

		for col in sampled_cols:
			if col in ['0', '3', '9', '10', '11']:
				op = random.choice(['<=', '>='])
			else:
				op = '='
			val =  random.randint(column_min_max_vals[col][0], column_min_max_vals[col][1])
			q_filters.append([col, op, val])

		filters.append(q_filters)

		num_generated += 1

	return filters


def query2cdfs(q_filters, column_min_max_vals):
	cdfs = [[]]
	lower_counts = [0]
	signs = []

	for filter in q_filters:
		col = filter[0]
		op = filter[1]
		val = filter[2]

		col_min = int(column_min_max_vals[col][0])
		col_max = int(column_min_max_vals[col][1])

		if op == '<=':
			if int(val) == col_max:
				continue

			for cdf in cdfs:
				cdf.append([col, '<=', val])
		elif op == '>=':
			if int(val) == col_min:
				continue

			cdfs_copy = copy.deepcopy(cdfs)
			lower_counts_copy = copy.deepcopy(lower_counts)

			for cdf in cdfs:
				cdf.append([col, '<=', int(val) - 1])
			for j in range(len(lower_counts)):
				lower_counts[j] += 1

			cdfs.extend(cdfs_copy)
			lower_counts.extend(lower_counts_copy)

		else:  # the case for =
			if int(val) == col_min:
				for cdf in cdfs:
					cdf.append([col, '<=', int(val)])
			else:
				cdfs_copy = copy.deepcopy(cdfs)
				lower_counts_copy = copy.deepcopy(lower_counts)

				for cdf in cdfs:
					cdf.append([col, '<=', int(val) - 1])
				for j in range(len(lower_counts)):
					lower_counts[j] += 1

				for cdf in cdfs_copy:
					cdf.append([col, '<=', int(val)])

				cdfs.extend(cdfs_copy)
				lower_counts.extend(lower_counts_copy)

	for count in lower_counts:
		signs.append((-1) ** count)

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



def sample_batch(aug_bs, column_min_max_vals, max_num_predicates,
                 shared_batch, column2vec, op2vec, cuda, lock, stop_event, updated_event):
	while not stop_event.is_set():
		# Simulate generating a new data batch
		consistent_filters = sample_queries(aug_bs, column_min_max_vals, max_num_predicates)
		consistent_cdfs = []
		consistent_signs = []
		for filters in consistent_filters:
			q_cdfs, q_signs = query2cdfs(filters, column_min_max_vals)
			consistent_cdfs.append(q_cdfs)
			consistent_signs.append(q_signs)

		predicates_consis = encode_data(consistent_filters, column_min_max_vals, column2vec, op2vec)
		cdf_predicates_consis = encode_cdf_data(consistent_cdfs,
		                                        column_min_max_vals,
		                                        column2vec, op2vec)
		aug_predicate_tensors, aug_predicate_masks, = make_aug_dataset(predicates_consis,
		                                                               max_num_predicates)
		aug_cdf_predicate_tensors, aug_cdf_predicate_masks, aug_cdf_signs_tensors = make_aug_dataset_cdf(
			cdf_predicates_consis, consistent_signs, max_num_predicates)
		with lock:
			shared_batch[:] = []
			shared_batch.append(aug_predicate_tensors.share_memory_())
			shared_batch.append(aug_predicate_masks.share_memory_())
			shared_batch.append(aug_cdf_predicate_tensors.share_memory_())
			shared_batch.append(aug_cdf_predicate_masks.share_memory_())
			shared_batch.append(aug_cdf_signs_tensors.share_memory_())

			updated_event.set()