import copy
import csv
import random

import numpy as np
import torch
from torch.utils.data import dataset

from mscn.util import *

def load_data(file_name, num_materialized_samples, num_train, dataset='imdb', column_min_max_vals=None, trans_op=False, workload_type='in', shift='center'):
	joins = []
	predicates = []
	tables = []
	samples = []
	label = []

	q_lens = []
	q_centers = []

	if dataset == 'imdb':
		shifting_var = 't.production_year'
	elif dataset == 'dsb':
		shifting_var = 'ss.ss_list_price'


	shift_col_min = int(column_min_max_vals[shifting_var][0])
	shift_col_max = int(column_min_max_vals[shifting_var][1])

	random.seed(42)

	valid_qids = [] # for filtering samples

	num_joins = []
	num_predicates = []
	table_sets = []
	table2numjoins = {}

	numerical_cols = []

	q_types = {}
	q_ids_per_type = {}

	min_max_numpreds_per_joingraph = {}

	num_all_qs = 0
	# Load queries
	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		for qid, row in enumerate(data_raw):
			# check neg
			num_all_qs += 1
			a_predicate = row[2]
			contain_neg = False
			if a_predicate:
				a_predicate = row[2].split(',')
				a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
				for an_item in a_predicate:
					op = an_item[1]
					if op == '!=':
						contain_neg = True
			if contain_neg:
				continue

			valid_qids.append(qid)
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))
			table_sets.append(frozenset(row[0].split(',')))

			if trans_op:
				transformed_predicate = []
				a_predicate = row[2]

				if a_predicate:
					a_predicate = row[2].split(',')
					a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
					for an_item in a_predicate:
						col = an_item[0]
						op = an_item[1]
						val = an_item[2]
						if op in ['<', '>', '<=', '>=']:
							if col not in numerical_cols:
								numerical_cols.append(col)

						if op == '<':
							if isinstance(val, str):
								transformed_predicate.extend([col, '<=', str(int(val) - 1)])
							else:
								transformed_predicate.extend([col, '<=', val - 1])
						elif op == '>':
							if isinstance(val, str):
								transformed_predicate.extend([col, '>=', str(int(val) + 1)])
							else:
								transformed_predicate.extend([col, '>=', val + 1])
						else:
							transformed_predicate.extend([col, op, val])

					if transformed_predicate:
						predicates.append(transformed_predicate)

					else:
						predicates.append([''])
				else:
					predicates.append([''])
			else:
				predicates.append(row[2].split(','))
				a_predicate = row[2]
				if a_predicate:
					a_predicate = row[2].split(',')
					a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
					for an_item in a_predicate:
						col = an_item[0]
						op = an_item[1]
						if op in ['<', '>', '<=', '>=']:
							if col not in numerical_cols:
								numerical_cols.append(col)

			if int(row[3]) < 1:
				print("Queries must have non-zero cardinalities")
				exit(1)
			label.append(row[3])

			#### get query range and shift col center
			col_len = None
			col_range = [0, 1]

			a_predicate = predicates[-1]
			if len(a_predicate) > 1:
				for qid in range(int(len(a_predicate) / 3)):
					col = a_predicate[qid*3]
					op = a_predicate[qid*3+1]
					val = a_predicate[qid*3+2]

					if col == shifting_var:
						if op == '=':
							col_len = 1. / (shift_col_max - shift_col_min)
							col_range = [(int(val) - shift_col_min) / (shift_col_max - shift_col_min),
							              (int(val) - shift_col_min) / (shift_col_max - shift_col_min)]
						elif op == '<=':
							col_len = (int(val) - shift_col_min +1) / (shift_col_max - shift_col_min)
							col_range[1] = (int(val) - shift_col_min) / (shift_col_max - shift_col_min)
						elif op == '>=':
							col_len = (shift_col_max - int(val) +1) / (shift_col_max - shift_col_min)
							col_range[0] = (int(val) - shift_col_min) / (shift_col_max - shift_col_min)

			q_lens.append(col_len)
			q_centers.append(np.mean(col_range))

			if row[1] == '':
				num_join = 0
			elif ',' not in row[1]:
				num_join = 1
			else:
				num_join = len(row[1].split(','))
			num_joins.append(num_join)

			if frozenset(row[0].split(',')) not in table2numjoins:
				table2numjoins[frozenset(row[0].split(','))] = num_join

			t_set = frozenset(row[0].split(','))
			if ',' not in row[2]:
				num_predicates.append(0)
				if t_set in min_max_numpreds_per_joingraph:
					if 0 < min_max_numpreds_per_joingraph[t_set][0]:
						min_max_numpreds_per_joingraph[t_set][0] = 0
				else:
					min_max_numpreds_per_joingraph[t_set] = [0, 0]
			else:
				num_predicates.append(len(row[2].split(',')) / 3)
				num_p = len(row[2].split(',')) / 3

				if t_set in min_max_numpreds_per_joingraph:
					if num_p < min_max_numpreds_per_joingraph[t_set][0]:
						min_max_numpreds_per_joingraph[t_set][0] = num_p
					if num_p > min_max_numpreds_per_joingraph[t_set][1]:
						min_max_numpreds_per_joingraph[t_set][1] = num_p
				else:
					min_max_numpreds_per_joingraph[t_set] = [num_p, num_p]
			if len(valid_qids) >= num_train:
				break
	print("Loaded queries")

	candi_joins = []
	candi_predicates = []
	candi_tables = []
	candi_samples = []
	candi_label = []
	candi_num_joins = []
	candi_num_predicates = []
	candi_table_sets = []
	candi_q_center = []
	candi_q_lens = []
	candi_ori_qs = []

	candi_file_name = './workloads/imdb-test'
	if dataset == 'dsb':
		candi_file_name = './workloads/dsb'

	### load in-distribution workload
	with open(candi_file_name + ".csv", 'rU') as f:
		if dataset == 'dsb':
			data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))[:80000]
		else:
			if shift == 'center':
				data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))[:10000]
			else:
				data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))[:5000]
		num_all_candi_qs = len(data_raw)
		for qid, row in enumerate(data_raw):
			candi_ori_qs.append('#'.join(row))
			candi_tables.append(row[0].split(','))
			candi_joins.append(row[1].split(','))
			candi_table_sets.append(frozenset(row[0].split(',')))
			if int(row[3]) < 1:
				print("Queries must have non-zero cardinalities")
				exit(1)
			candi_label.append(row[3])

			if trans_op:
				transformed_predicate = []
				a_predicate = row[2]

				if a_predicate:
					a_predicate = row[2].split(',')
					a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
					for an_item in a_predicate:
						col = an_item[0]
						op = an_item[1]
						val = an_item[2]
						if op in ['<', '>', '<=', '>=']:
							if col not in numerical_cols:
								numerical_cols.append(col)

						if op == '<':
							if isinstance(val, str):
								transformed_predicate.extend([col, '<=', str(int(val) - 1)])
							else:
								transformed_predicate.extend([col, '<=', val - 1])
						elif op == '>':
							if isinstance(val, str):
								transformed_predicate.extend([col, '>=', str(int(val) + 1)])
							else:
								transformed_predicate.extend([col, '>=', val + 1])
						else:
							transformed_predicate.extend([col, op, val])

					if transformed_predicate:
						candi_predicates.append(transformed_predicate)

					else:
						candi_predicates.append([''])
				else:
					candi_predicates.append([''])
			else:
				candi_predicates.append(row[2].split(','))

			if row[1] == '':
				num_join = 0
			elif ',' not in row[1]:
				num_join = 1
			else:
				num_join = len(row[1].split(','))
			candi_num_joins.append(num_join)

			if frozenset(row[0].split(',')) not in table2numjoins:
				table2numjoins[frozenset(row[0].split(','))] = num_join

			t_set = frozenset(row[0].split(','))


			if ',' not in row[2]:
				candi_num_predicates.append(0)

				if t_set in min_max_numpreds_per_joingraph:
					if 0 < min_max_numpreds_per_joingraph[t_set][0]:
						min_max_numpreds_per_joingraph[t_set][0] = 0
				else:
					min_max_numpreds_per_joingraph[t_set] = [0, 0]
			else:
				candi_num_predicates.append(len(row[2].split(',')) / 3)

				num_p = len(row[2].split(',')) / 3

				if t_set in min_max_numpreds_per_joingraph:
					if num_p < min_max_numpreds_per_joingraph[t_set][0]:
						min_max_numpreds_per_joingraph[t_set][0] = num_p
					if num_p > min_max_numpreds_per_joingraph[t_set][1]:
						min_max_numpreds_per_joingraph[t_set][1] = num_p
				else:
					min_max_numpreds_per_joingraph[t_set] = [num_p, num_p]

			a_predicate = candi_predicates[-1]
			q_range = copy.deepcopy(column_min_max_vals)
			col_range = [0, 1]
			col_len = None

			if len(a_predicate) > 1:
				for qid in range(int(len(a_predicate) / 3)):
					col = a_predicate[qid * 3]
					op = a_predicate[qid * 3 + 1]
					val = a_predicate[qid * 3 + 2]

					if op == '>=':
						q_range[col][0] = val
					elif op == '<=':
						q_range[col][1] = val
					elif op == '=':
						q_range[col][1] = q_range[col][0]

					if col == shifting_var:
						if op == '=':
							col_len = 1. / (shift_col_max - shift_col_min)
							col_range = [(int(val) - shift_col_min) / (shift_col_max - shift_col_min),
							             (int(val) - shift_col_min) / (shift_col_max - shift_col_min)]
						elif op == '<=':
							col_len = (int(val) - shift_col_min + 1) / (shift_col_max - shift_col_min)
							col_range[1] = (int(val) - shift_col_min) / (shift_col_max - shift_col_min)
						elif op == '>=':
							col_len = (shift_col_max - int(val) + 1) / (shift_col_max - shift_col_min)
							col_range[0] = (int(val) - shift_col_min) / (shift_col_max - shift_col_min)

			candi_q_center.append(np.mean(col_range))
			candi_q_lens.append(col_len)

	print("Loaded candidate queries")
	print(len(candi_tables))

	num_types = 0
	for qid, (table_set, num_p) in enumerate(zip(table_sets, num_predicates)):
		if table_set not in q_types:
			q_types[table_set] = {}
			q_ids_per_type[table_set] = {}
		if num_p not in q_types[table_set]:
			q_types[table_set][num_p] = 0
			q_ids_per_type[table_set][num_p] = []
			num_types += 1

		q_types[table_set][num_p] += 1
		q_ids_per_type[table_set][num_p].append(qid)

	print("all number of queries: {}".format(len(table_sets)))
	print("all number of types: {}".format(num_types))

	# Load bitmaps
	num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
	with open(file_name.replace("workloads", "bitmaps") + ".bitmaps", 'rb') as f:
		for i in range(num_all_qs):
			four_bytes = f.read(4)
			if not four_bytes:
				print("Error while reading 'four_bytes'")
				exit(1)
			num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
			bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
			for j in range(num_bitmaps_curr_query):
				# Read bitmap
				bitmap_bytes = f.read(num_bytes_per_bitmap)
				if not bitmap_bytes:
					print("Error while reading 'bitmap_bytes'")
					exit(1)
				bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
			samples.append(bitmaps)
	print("Loaded bitmaps")

	# Load candi bitmaps
	num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
	with open(candi_file_name.replace("workloads", "bitmaps") + ".bitmaps", 'rb') as f:
		for i in range(num_all_candi_qs):
			four_bytes = f.read(4)
			if not four_bytes:
				print("Error while reading 'four_bytes'")
				exit(1)
			num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
			bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
			for j in range(num_bitmaps_curr_query):
				# Read bitmap
				bitmap_bytes = f.read(num_bytes_per_bitmap)
				if not bitmap_bytes:
					print("Error while reading 'bitmap_bytes'")
					exit(1)
				bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
			candi_samples.append(bitmaps)
	print("Loaded candi bitmaps")

	samples = [samples[qid] for qid in valid_qids]
	train_ids = []
	test_size = 0

	for table_set in q_ids_per_type:
		q_ids_per_table_set = q_ids_per_type[table_set]
		keys_list = list(q_ids_per_table_set.keys())
		q_ids_type = []

		for num_p in keys_list:
			q_ids = q_ids_per_table_set[num_p]
			q_ids_type.extend(q_ids)

		if shift == 'center': # center move
			temp_labels = [q_centers[qid] for qid in q_ids_type]
			paired_sorted = zip(q_ids_type, temp_labels)

			if workload_type == 'in':
				filtered_paired_sorted = [id for id, label in paired_sorted if (label < 0.25 or label > 0.75)]
			elif workload_type == 'ood':
				filtered_paired_sorted = [id for id, label in paired_sorted if (label > 0.25 and label < 0.75 and label != 0.5)]

			train_ids.extend(filtered_paired_sorted)
		else: # granularity shift
			temp_labels = [q_lens[qid] for qid in q_ids_type]
			num_q_per_join_graph = int(len(q_ids_type) * 0.2)
			paired_sorted = zip(q_ids_type, temp_labels)

			if workload_type == 'in':
				filtered_paired_sorted = [id for id, label in paired_sorted if label is not None]
			elif workload_type == 'ood':
				filtered_paired_sorted = [id for id, label in paired_sorted if label is None]

			random.shuffle(filtered_paired_sorted)
			filtered_paired_sorted = filtered_paired_sorted[:num_q_per_join_graph]
			train_ids.extend(filtered_paired_sorted)

	if shift == 'center':  # center move
		if workload_type == 'in':
			candi_q_ids = [id for id, label in enumerate(candi_q_center) if
			               (label < 0.25 or label > 0.75)]
		elif workload_type == 'ood':
			candi_q_ids = [id for id, label in enumerate(candi_q_center) if
			               (label > 0.25 and label < 0.75 and label != 0.5)]
	else:  # granularity shift
		if workload_type == 'in':
			candi_q_ids = [id for id, label in enumerate(candi_q_lens) if label is not None]
		elif workload_type == 'ood':
			candi_q_ids = [id for id, label in enumerate(candi_q_lens) if label is None]


	print('len of train_ids {}'.format(len(train_ids)))
	random.shuffle(train_ids)
	new_joins = [joins[qid] for qid in train_ids]
	new_predicates = [predicates[qid] for qid in train_ids]
	new_tables = [tables[qid] for qid in train_ids]
	new_samples = [samples[qid] for qid in train_ids]
	new_label = [label[qid] for qid in train_ids]
	new_num_joins = [num_joins[qid] for qid in train_ids]
	new_num_predicates = [num_predicates[qid] for qid in train_ids]
	new_table_sets = [table_sets[qid] for qid in train_ids]

	# Split predicates
	new_predicates = [list(chunks(d, 3)) for d in new_predicates]
	candi_predicates = [list(chunks(d, 3)) for d in candi_predicates]

	if trans_op:
		for d in new_predicates:
			for pred in d:
				if len(pred) > 1:
					if pred[1] == '<':
						pred[1] = '<='
						if isinstance(pred[2], str):
							pred[2] = str(int(pred[2]) - 1)
						else:
							pred[2] = pred[2] - 1
					elif pred[1] == '>':
						pred[1] = '>='
						if isinstance(pred[2], str):
							pred[2] = str(int(pred[2]) + 1)
						else:
							pred[2] = pred[2] + 1

		for d in candi_predicates:
			for pred in d:
				if len(pred) > 1:
					if pred[1] == '<':
						pred[1] = '<='
						if isinstance(pred[2], str):
							pred[2] = str(int(pred[2]) - 1)
						else:
							pred[2] = pred[2] - 1
					elif pred[1] == '>':
						pred[1] = '>='
						if isinstance(pred[2], str):
							pred[2] = str(int(pred[2]) + 1)
						else:
							pred[2] = pred[2] + 1

	if dataset == 'dsb':
		candi_q_ids = [qid for qid in candi_q_ids if qid >= num_train]

	candi_joins = [candi_joins[qid] for qid in candi_q_ids]
	candi_predicates = [candi_predicates[qid] for qid in candi_q_ids]
	candi_tables = [candi_tables[qid] for qid in candi_q_ids]
	candi_samples = [candi_samples[qid] for qid in candi_q_ids]
	candi_label = [candi_label[qid] for qid in candi_q_ids]
	candi_num_joins = [candi_num_joins[qid] for qid in candi_q_ids]
	candi_num_predicates = [candi_num_predicates[qid] for qid in candi_q_ids]
	candi_table_sets = [candi_table_sets[qid] for qid in candi_q_ids]

	return new_joins, new_predicates, new_tables, new_samples, new_label, new_num_joins, \
	       new_num_predicates, new_table_sets, numerical_cols, test_size, candi_joins, candi_predicates, candi_tables, candi_samples, candi_label, candi_num_joins, \
	       candi_num_predicates, candi_table_sets, min_max_numpreds_per_joingraph

def load_ood_data(file_name, num_materialized_samples, num_train, dataset='imdb', column_min_max_vals=None, trans_op=False, shift='center'):
	joins = []
	predicates = []
	tables = []
	samples = []
	label = []

	q_centers = []
	q_lens = []
	original_qids = []

	if dataset == 'imdb':
		shifting_var = 't.production_year'
	elif dataset == 'dsb':
		shifting_var = 'ss.ss_list_price'

	shift_col_min = int(column_min_max_vals[shifting_var][0])
	shift_col_max = int(column_min_max_vals[shifting_var][1])

	valid_qids = [] # for filtering samples

	num_predicates = []
	table_sets = []

	numerical_cols = []

	q_types = {}
	q_ids_per_type = {}

	num_all_qs = 0
	# Load queries
	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		for qid, row in enumerate(data_raw):
			# check neg
			num_all_qs += 1
			a_predicate = row[2]
			contain_neg = False
			if a_predicate:
				a_predicate = row[2].split(',')
				a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
				for an_item in a_predicate:
					op = an_item[1]
					if op == '!=':
						contain_neg = True
			if contain_neg:
				continue

			valid_qids.append(qid)
			original_qids.append(qid)
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))
			table_sets.append(frozenset(row[0].split(',')))

			if trans_op:
				transformed_predicate = []
				a_predicate = row[2]

				if a_predicate:
					a_predicate = row[2].split(',')
					a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
					for an_item in a_predicate:
						col = an_item[0]
						op = an_item[1]
						val = an_item[2]
						if op in ['<', '>', '<=', '>=']:
							if col not in numerical_cols:
								numerical_cols.append(col)

						if op == '<':
							if isinstance(val, str):
								transformed_predicate.extend([col, '<=', str(int(val) - 1)])
							else:
								transformed_predicate.extend([col, '<=', val - 1])
						elif op == '>':
							if isinstance(val, str):
								transformed_predicate.extend([col, '>=', str(int(val) + 1)])
							else:
								transformed_predicate.extend([col, '>=', val + 1])
						else:
							transformed_predicate.extend([col, op, val])

					if transformed_predicate:
						predicates.append(transformed_predicate)

					else:
						predicates.append([''])
				else:
					predicates.append([''])
			else:
				predicates.append(row[2].split(','))
				a_predicate = row[2]
				if a_predicate:
					a_predicate = row[2].split(',')
					a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
					for an_item in a_predicate:
						col = an_item[0]
						op = an_item[1]
						if op in ['<', '>', '<=', '>=']:
							if col not in numerical_cols:
								numerical_cols.append(col)

			if int(row[3]) < 1:
				print("Queries must have non-zero cardinalities")
				exit(1)

			#### get query range and year center
			q_range = copy.deepcopy(column_min_max_vals)
			a_predicate = predicates[-1]
			col_range = [0, 1]
			col_len = None

			if len(a_predicate) > 1:
				for qid in range(int(len(a_predicate) / 3)):
					col = a_predicate[qid * 3]
					op = a_predicate[qid * 3 + 1]
					val = a_predicate[qid * 3 + 2]

					if op == '>=':
						q_range[col][0] = val
					elif op == '<=':
						q_range[col][1] = val
					elif op == '=':
						q_range[col][1] = q_range[col][0]

					if col == shifting_var:
						if op == '=':
							col_len = 1. / (shift_col_max - shift_col_min)
							col_range = [(int(val) - shift_col_min) / (shift_col_max - shift_col_min),
							             (int(val) - shift_col_min) / (shift_col_max - shift_col_min)]
						elif op == '<=':
							col_len = (int(val) - shift_col_min + 1) / (shift_col_max - shift_col_min)
							col_range[1] = (int(val) - shift_col_min) / (shift_col_max - shift_col_min)
						elif op == '>=':
							col_len = (shift_col_max - int(val) + 1) / (shift_col_max - shift_col_min)
							col_range[0] = (int(val) - shift_col_min) / (shift_col_max - shift_col_min)

			q_lens.append(col_len)
			q_centers.append(np.mean(col_range))

			label.append(row[3])

			if ',' not in row[2]:
				num_predicates.append(0)
			else:
				num_predicates.append(len(row[2].split(','))/3)

			if len(valid_qids) >= num_train:
				break

	num_types = 0
	for qid, (table_set, num_p) in enumerate(zip(table_sets, num_predicates)):
		if table_set not in q_types:
			q_types[table_set] = {}
			q_ids_per_type[table_set] = {}
		if num_p not in q_types[table_set]:
			q_types[table_set][num_p] = 0
			q_ids_per_type[table_set][num_p] = []
			num_types += 1

		q_types[table_set][num_p] += 1
		q_ids_per_type[table_set][num_p].append(qid)

	# Load bitmaps
	num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
	with open(file_name.replace("workloads", "bitmaps") + ".bitmaps", 'rb') as f:
		for i in range(num_all_qs):
			four_bytes = f.read(4)
			if not four_bytes:
				print("Error while reading 'four_bytes'")
				exit(1)
			num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
			bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
			for j in range(num_bitmaps_curr_query):
				# Read bitmap
				bitmap_bytes = f.read(num_bytes_per_bitmap)
				if not bitmap_bytes:
					print("Error while reading 'bitmap_bytes'")
					exit(1)
				bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
			samples.append(bitmaps)

	samples = [samples[qid] for qid in valid_qids]
	train_ids = []

	for table_set in q_ids_per_type:
		q_ids_per_table_set = q_ids_per_type[table_set]
		keys_list = list(q_ids_per_table_set.keys())
		q_ids_type = []

		for num_p in keys_list:
			q_ids = q_ids_per_table_set[num_p]
			q_ids_type.extend(q_ids)

		if shift == 'center':
			opp_num_q_per_join_graph = int(len(q_ids_type) * 0.01)
			temp_labels = [q_centers[qid] for qid in q_ids_type]
			paired_sorted = zip(q_ids_type, temp_labels)
			filtered_paired_sorted = [id for id, label in paired_sorted if (label > 0.25 and label < 0.75 and label != 0.5)]
			random.shuffle(filtered_paired_sorted)
			if dataset == 'imdb':
				train_ids.extend(filtered_paired_sorted[:3 * opp_num_q_per_join_graph])
			elif dataset == 'dsb':
				train_ids.extend(filtered_paired_sorted[:4 * opp_num_q_per_join_graph])
		else:
			temp_labels = [q_lens[qid] for qid in q_ids_type]
			opp_num_q_per_join_graph = int(len(q_ids_type) * 0.01)
			paired_sorted = zip(q_ids_type, temp_labels)
			filtered_paired_sorted = [id for id, label in paired_sorted if label is None]
			random.shuffle(filtered_paired_sorted)

			train_ids.extend(filtered_paired_sorted[:4 * opp_num_q_per_join_graph])

	joins = [joins[qid] for qid in train_ids]
	predicates = [predicates[qid] for qid in train_ids]
	tables = [tables[qid] for qid in train_ids]
	samples = [samples[qid] for qid in train_ids]
	label = [label[qid] for qid in train_ids]

	# Split predicates
	predicates = [list(chunks(d, 3)) for d in predicates]
	if trans_op:
		for d in predicates:
			for pred in d:
				if len(pred) > 1:
					if pred[1] == '<':
						pred[1] = '<='
						if isinstance(pred[2], str):
							pred[2] = str(int(pred[2]) - 1)
						else:
							pred[2] = pred[2] - 1
					elif pred[1] == '>':
						pred[1] = '>='
						if isinstance(pred[2], str):
							pred[2] = str(int(pred[2]) + 1)
						else:
							pred[2] = pred[2] + 1

	label = np.array([float(l) for l in label])
	return joins, predicates, tables, samples, label

def ori_load_data(file_name, num_materialized_samples):
	joins = []
	predicates = []
	tables = []
	samples = []
	label = []

	num_joins = []
	num_predicates = []
	table_sets = []
	table2numjoins = {}

	numerical_cols = []

	# Load queries
	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		num_all_qs = len(data_raw)
		for qid, row in enumerate(data_raw):
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))
			predicates.append(row[2].split(','))
			table_sets.append(frozenset(row[0].split(',')))

			a_predicate = row[2]
			if a_predicate:
				a_predicate = row[2].split(',')
				a_predicate = [a_predicate[i:i + 3] for i in range(0, len(a_predicate), 3)]
				for an_item in a_predicate:
					col = an_item[0]
					op = an_item[1]
					if op in ['<', '>', '<=', '>=']:
						if col not in numerical_cols:
							numerical_cols.append(col)

			if int(row[3]) < 1:
				print("Queries must have non-zero cardinalities")
				exit(1)
			label.append(row[3])

			if row[1] == '':
				num_join = 0
			elif ',' not in row[1]:
				num_join = 1
			else:
				num_join = len(row[1].split(','))
			num_joins.append(num_join)

			if frozenset(row[0].split(',')) not in table2numjoins:
				table2numjoins[frozenset(row[0].split(','))] = num_join

			if ',' not in row[2]:
				num_predicates.append(0)
			else:
				num_predicates.append(len(row[2].split(','))/3)

	# Load bitmaps
	num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
	with open(file_name.replace("workloads", "bitmaps") + ".bitmaps", 'rb') as f:
		for i in range(num_all_qs):
			four_bytes = f.read(4)
			if not four_bytes:
				print("Error while reading 'four_bytes'")
				exit(1)
			num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
			bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
			for j in range(num_bitmaps_curr_query):
				# Read bitmap
				bitmap_bytes = f.read(num_bytes_per_bitmap)
				if not bitmap_bytes:
					print("Error while reading 'bitmap_bytes'")
					exit(1)
				bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
			samples.append(bitmaps)
	print("Loaded bitmaps")

	# Split predicates
	predicates = [list(chunks(d, 3)) for d in predicates]

	return joins, predicates, tables, samples, label, num_joins, \
	       num_predicates, table_sets, numerical_cols

def load_and_encode_train_data(num_train, num_materialized_samples, dataset='imdb', trans_op=False, add_one=False, workload_type='in', shift='center'):

	file_name_queries = "workloads/imdb-train"
	file_name_column_min_max_vals = "data/column_min_max_vals_imdb.csv"
	if dataset == 'dsb':
		file_name_queries = "workloads/dsb"
		file_name_column_min_max_vals = "data/column_min_max_vals_dsb.csv"

	# Get min and max values for each column
	with open(file_name_column_min_max_vals, 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
		column_min_max_vals = {}
		for i, row in enumerate(data_raw):
			if i == 0:
				continue
			column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

	joins, predicates, tables, samples, label, num_joins, num_predicates, table_sets, numerical_cols, num_test, candi_joins, candi_predicates, candi_tables, candi_samples, candi_label, candi_num_joins, \
	candi_num_predicates, candi_table_sets, min_max_numpreds_per_joingraph = load_data(file_name_queries, num_materialized_samples, num_train, dataset, column_min_max_vals, trans_op=trans_op, workload_type=workload_type, shift=shift)

	# Get column name dict
	column_names = get_all_column_names(predicates)
	column2vec, idx2column = get_set_encoding(column_names)

	# Get table name dict
	table_names = get_all_table_names(tables)
	table2vec, idx2table = get_set_encoding(table_names)

	# Get operator name dict
	operators = get_all_operators(predicates)
	op2vec, idx2op = get_set_encoding(operators)

	# Get join name dict
	join_set = get_all_joins(joins)
	join2vec, idx2join = get_set_encoding(join_set)

	# Get feature encoding and proper normalization
	samples_enc = encode_samples(tables, samples, table2vec)
	predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
	label_norm, min_val, max_val = normalize_labels(label, dataset=dataset, add_one=add_one)

	candi_samples_enc = encode_samples(candi_tables, candi_samples, table2vec)
	candi_predicates_enc, candi_joins_enc = encode_data(candi_predicates, candi_joins, column_min_max_vals, column2vec, op2vec, join2vec)
	candi_label_norm, min_val, max_val = normalize_labels(candi_label, min_val, max_val, add_one=add_one)
	candi_label_norm = np.array(candi_label_norm)
	candi_query_typeids = {}

	for qid, (num_p, table_set) in enumerate(zip(candi_num_predicates, candi_table_sets)):
		table_set_str = '.'.join(sorted(table_set))
		type_str = '{},{}'.format(table_set_str, int(num_p))
		if type_str not in candi_query_typeids:
			candi_query_typeids[type_str] = []
		candi_query_typeids[type_str].append(qid)

	# Split in training and validation samples
	num_queries = len(samples_enc)
	num_train = num_queries - num_test
	num_train_oracle = num_train

	samples_train = samples_enc[:num_train_oracle]
	predicates_train = predicates_enc[:num_train_oracle]
	joins_train = joins_enc[:num_train_oracle]
	labels_train = list(label_norm[:num_train_oracle])

	ori_predicates_train = predicates[:num_train_oracle]
	ori_samples_train = samples[:num_train_oracle]
	ori_tables_train = tables[:num_train_oracle]
	ori_join_train = joins[:num_train_oracle]

	ori_predicates_test = predicates[num_train:num_train + num_test]
	ori_samples_test = samples[num_train:num_train + num_test]
	ori_tables_test = tables[num_train:num_train + num_test]

	num_joins_train = num_joins[:num_train_oracle]
	num_predicates_train = num_predicates[:num_train_oracle]
	table_sets_train = table_sets[:num_train_oracle ]

	labels_train = np.array(labels_train)

	samples_test = samples_enc[num_train:num_train + num_test]
	predicates_test = predicates_enc[num_train:num_train + num_test]
	joins_test = joins_enc[num_train:num_train + num_test]
	labels_test = label_norm[num_train:num_train + num_test]
	labels_test = np.array(labels_test)

	num_joins_test = num_joins[num_train:num_train + num_test]
	num_predicates_test = num_predicates[num_train:num_train + num_test]
	table_sets_test = table_sets[num_train:num_train + num_test]

	print("Number of training samples: {}".format(len(labels_train)))
	print("Number of validation samples: {}".format(len(labels_test)))

	if len(candi_joins_enc):
		max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in candi_predicates_enc]))
		max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in candi_predicates_enc]))
	else:
		max_num_joins = max([len(j) for j in joins_train])
		max_num_predicates = max([len(p) for p in predicates_train])
	max_num_predicates = max_num_predicates + 1

	dicts = [table2vec, column2vec, op2vec, join2vec]
	train_data = [samples_train, predicates_train, joins_train]
	test_data = [samples_test, predicates_test, joins_test]
	candi_data = [candi_samples_enc, candi_predicates_enc, candi_joins_enc]

	return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, candi_label_norm, max_num_joins, max_num_predicates, train_data, test_data, candi_data, \
	       ori_predicates_train, ori_samples_train, ori_tables_train, ori_join_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, \
	       num_joins_test, num_predicates_test, table_sets_test, numerical_cols, candi_query_typeids, candi_joins, candi_predicates, candi_tables, candi_samples, min_max_numpreds_per_joingraph

def make_dataset(samples, predicates, joins, labels, max_num_joins, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	sample_masks = []
	sample_tensors = []
	for sample in samples:
		sample_tensor = np.vstack(sample)
		num_pad = max_num_joins + 1 - sample_tensor.shape[0]
		sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
		sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
		sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
		sample_tensors.append(np.expand_dims(sample_tensor, 0))
		sample_masks.append(np.expand_dims(sample_mask, 0))
	sample_tensors = np.vstack(sample_tensors)
	sample_tensors = torch.FloatTensor(sample_tensors)
	sample_masks = np.vstack(sample_masks)
	sample_masks = torch.FloatTensor(sample_masks)

	predicate_masks = []
	predicate_tensors = []
	for predicate in predicates:
		predicate_tensor = np.vstack(predicate)
		num_pad = max_num_predicates - predicate_tensor.shape[0]
		predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
		predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
		predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
		predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
		predicate_masks.append(np.expand_dims(predicate_mask, 0))
	predicate_tensors = np.vstack(predicate_tensors)
	predicate_tensors = torch.FloatTensor(predicate_tensors)
	predicate_masks = np.vstack(predicate_masks)
	predicate_masks = torch.FloatTensor(predicate_masks)

	join_masks = []
	join_tensors = []
	for join in joins:
		join_tensor = np.vstack(join)
		num_pad = max_num_joins - join_tensor.shape[0]
		join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
		join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
		join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
		join_tensors.append(np.expand_dims(join_tensor, 0))
		join_masks.append(np.expand_dims(join_mask, 0))
	join_tensors = np.vstack(join_tensors)
	join_tensors = torch.FloatTensor(join_tensors)
	join_masks = np.vstack(join_masks)
	join_masks = torch.FloatTensor(join_masks)

	target_tensor = torch.FloatTensor(labels)

	ids_tensor = torch.IntTensor(np.arange(len(labels)))

	return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks,
	                             predicate_masks, join_masks, ids_tensor)

def make_aug_dataset(samples, predicates, joins, max_num_joins, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	sample_masks = []
	sample_tensors = []
	for sample in samples:
		sample_tensor = np.vstack(sample)
		num_pad = max_num_joins + 1 - sample_tensor.shape[0]
		sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
		sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
		sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
		sample_tensors.append(np.expand_dims(sample_tensor, 0))
		sample_masks.append(np.expand_dims(sample_mask, 0))
	sample_tensors = np.vstack(sample_tensors)
	sample_tensors = torch.FloatTensor(sample_tensors)
	sample_masks = np.vstack(sample_masks)
	sample_masks = torch.FloatTensor(sample_masks)

	predicate_masks = []
	predicate_tensors = []
	for predicate in predicates:
		predicate_tensor = np.vstack(predicate)
		num_pad = max_num_predicates - predicate_tensor.shape[0]
		predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
		predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
		predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
		predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
		predicate_masks.append(np.expand_dims(predicate_mask, 0))
	predicate_tensors = np.vstack(predicate_tensors)
	predicate_tensors = torch.FloatTensor(predicate_tensors)
	predicate_masks = np.vstack(predicate_masks)
	predicate_masks = torch.FloatTensor(predicate_masks)

	join_masks = []
	join_tensors = []
	for join in joins:
		join_tensor = np.vstack(join)
		num_pad = max_num_joins - join_tensor.shape[0]
		join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
		join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
		join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
		join_tensors.append(np.expand_dims(join_tensor, 0))
		join_masks.append(np.expand_dims(join_mask, 0))
	join_tensors = np.vstack(join_tensors)
	join_tensors = torch.FloatTensor(join_tensors)
	join_masks = np.vstack(join_masks)
	join_masks = torch.FloatTensor(join_masks)

	return sample_tensors, predicate_tensors, join_tensors, sample_masks, predicate_masks, join_masks

def make_dataset_cdf(samples, predicates, joins, cdf_samples, cdf_predicates, cdf_joins, cdf_signs, labels, max_num_joins, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	max_num_cdfs = 0
	for a_cdf_samples in cdf_samples:
		if len(a_cdf_samples) > max_num_cdfs:
			max_num_cdfs = len(a_cdf_samples)

	sample_masks = []
	sample_tensors = []
	for sample in samples:
		sample_tensor = np.vstack(sample)
		num_pad = max_num_joins + 1 - sample_tensor.shape[0]
		sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
		sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
		sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
		sample_tensors.append(np.expand_dims(sample_tensor, 0))
		sample_masks.append(np.expand_dims(sample_mask, 0))
	sample_tensors = np.vstack(sample_tensors)
	sample_tensors = torch.FloatTensor(sample_tensors)
	sample_masks = np.vstack(sample_masks)
	sample_masks = torch.FloatTensor(sample_masks)

	cdf_sample_masks = []
	cdf_sample_tensors = []

	for a_query_samples in cdf_samples:
		a_query_sample_masks = []
		a_query_sample_tensors = []
		for sample in a_query_samples:
			sample_tensor = np.vstack(sample)
			num_pad = max_num_joins + 1 - sample_tensor.shape[0]
			sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
			sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
			sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
			a_query_sample_masks.append(np.expand_dims(sample_mask, 0))
			a_query_sample_tensors.append(np.expand_dims(sample_tensor, 0))

		if len(a_query_sample_masks) < max_num_cdfs:
			for _ in range(max_num_cdfs-len(a_query_sample_masks)):
				a_query_sample_tensors.append(a_query_sample_tensors[0])
				a_query_sample_masks.append(a_query_sample_masks[0])


		a_query_sample_tensors = np.vstack(a_query_sample_tensors)
		a_query_sample_masks = np.vstack(a_query_sample_masks)

		cdf_sample_masks.append(a_query_sample_masks)
		cdf_sample_tensors.append(a_query_sample_tensors)

	cdf_sample_masks = torch.FloatTensor(cdf_sample_masks)
	cdf_sample_tensors = torch.FloatTensor(cdf_sample_tensors)

	#### process signs!
	for a_query_signs in cdf_signs:
		if len(a_query_signs) < max_num_cdfs:
			for _ in range(max_num_cdfs - len(a_query_signs)):
				a_query_signs.append(0)
	cdf_signs_tensors = np.vstack(cdf_signs)
	cdf_signs_tensors = torch.FloatTensor(cdf_signs_tensors)

	#### finished processing signs!

	predicate_masks = []
	predicate_tensors = []
	for predicate in predicates:
		predicate_tensor = np.vstack(predicate)
		num_pad = max_num_predicates - predicate_tensor.shape[0]
		predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
		predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
		predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
		predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
		predicate_masks.append(np.expand_dims(predicate_mask, 0))
	predicate_tensors = np.vstack(predicate_tensors)
	predicate_tensors = torch.FloatTensor(predicate_tensors)
	predicate_masks = np.vstack(predicate_masks)
	predicate_masks = torch.FloatTensor(predicate_masks)

	cdf_predicate_masks = []
	cdf_predicate_tensors = []
	for a_query_predicate in cdf_predicates:
		a_query_predicate_masks = []
		a_query_predicate_tensors = []
		for predicate in a_query_predicate:
			predicate_tensor = np.vstack(predicate)
			num_pad = max_num_predicates - predicate_tensor.shape[0]
			predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
			predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
			predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
			a_query_predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
			a_query_predicate_masks.append(np.expand_dims(predicate_mask, 0))

		if len(a_query_predicate_masks) < max_num_cdfs:
			for _ in range(max_num_cdfs-len(a_query_predicate_masks)):
				a_query_predicate_masks.append(a_query_predicate_masks[0])
				a_query_predicate_tensors.append(a_query_predicate_tensors[0])

		a_query_predicate_tensors = np.vstack(a_query_predicate_tensors)
		a_query_predicate_masks = np.vstack(a_query_predicate_masks)

		cdf_predicate_masks.append(a_query_predicate_masks)
		cdf_predicate_tensors.append(a_query_predicate_tensors)

	cdf_predicate_masks = torch.FloatTensor(cdf_predicate_masks)
	cdf_predicate_tensors = torch.FloatTensor(cdf_predicate_tensors)

	join_masks = []
	join_tensors = []
	for join in joins:
		join_tensor = np.vstack(join)
		num_pad = max_num_joins - join_tensor.shape[0]
		join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
		join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
		join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
		join_tensors.append(np.expand_dims(join_tensor, 0))
		join_masks.append(np.expand_dims(join_mask, 0))
	join_tensors = np.vstack(join_tensors)
	join_tensors = torch.FloatTensor(join_tensors)
	join_masks = np.vstack(join_masks)
	join_masks = torch.FloatTensor(join_masks)

	cdf_join_masks = []
	cdf_join_tensors = []
	for a_query_cdf in cdf_joins:
		q_query_join_masks = []
		q_query_join_tensors = []
		for join in a_query_cdf:
			join_tensor = np.vstack(join)
			num_pad = max_num_joins - join_tensor.shape[0]
			join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
			join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
			join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
			q_query_join_tensors.append(np.expand_dims(join_tensor, 0))
			q_query_join_masks.append(np.expand_dims(join_mask, 0))

		if len(q_query_join_masks) < max_num_cdfs:
			for _ in range(max_num_cdfs-len(q_query_join_masks)):
				q_query_join_masks.append(q_query_join_masks[0])
				q_query_join_tensors.append(q_query_join_tensors[0])

		q_query_join_tensors = np.vstack(q_query_join_tensors)
		q_query_join_masks = np.vstack(q_query_join_masks)

		cdf_join_tensors.append(q_query_join_tensors)
		cdf_join_masks.append(q_query_join_masks)

	cdf_join_tensors = torch.FloatTensor(cdf_join_tensors)
	cdf_join_masks = torch.FloatTensor(cdf_join_masks)

	target_tensor = torch.FloatTensor(labels)

	ids_tensor = torch.IntTensor(np.arange(len(labels)))

	return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks,
	                             predicate_masks, join_masks, ids_tensor,
	                             cdf_sample_tensors, cdf_predicate_tensors, cdf_join_tensors,
	                             cdf_sample_masks, cdf_predicate_masks, cdf_join_masks, cdf_signs_tensors)

def make_aug_dataset_cdf(cdf_samples, cdf_predicates, cdf_joins, cdf_signs, max_num_joins, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	max_num_cdfs = 0
	for a_cdf_samples in cdf_samples:
		if len(a_cdf_samples) > max_num_cdfs:
			max_num_cdfs = len(a_cdf_samples)

	cdf_sample_masks = []
	cdf_sample_tensors = []

	for a_query_samples in cdf_samples:
		a_query_sample_masks = []
		a_query_sample_tensors = []
		for sample in a_query_samples:
			sample_tensor = np.vstack(sample)
			num_pad = max_num_joins + 1 - sample_tensor.shape[0]
			sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
			sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
			sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
			a_query_sample_masks.append(np.expand_dims(sample_mask, 0))
			a_query_sample_tensors.append(np.expand_dims(sample_tensor, 0))

		if len(a_query_sample_masks) < max_num_cdfs:
			for _ in range(max_num_cdfs-len(a_query_sample_masks)):
				a_query_sample_tensors.append(a_query_sample_tensors[0])
				a_query_sample_masks.append(a_query_sample_masks[0])

		a_query_sample_tensors = np.vstack(a_query_sample_tensors)
		a_query_sample_masks = np.vstack(a_query_sample_masks)

		cdf_sample_masks.append(a_query_sample_masks)
		cdf_sample_tensors.append(a_query_sample_tensors)

	cdf_sample_masks = torch.FloatTensor(np.array(cdf_sample_masks))
	cdf_sample_tensors = torch.FloatTensor(np.array(cdf_sample_tensors))

	#### process signs!
	for a_query_signs in cdf_signs:
		if len(a_query_signs) < max_num_cdfs:
			for _ in range(max_num_cdfs - len(a_query_signs)):
				a_query_signs.append(0)
	cdf_signs_tensors = np.vstack(cdf_signs)
	cdf_signs_tensors = torch.FloatTensor(cdf_signs_tensors)

	#### finished processing signs!

	cdf_predicate_masks = []
	cdf_predicate_tensors = []
	for a_query_predicate in cdf_predicates:
		a_query_predicate_masks = []
		a_query_predicate_tensors = []

		for predicate in a_query_predicate:
			predicate_tensor = np.vstack(predicate)
			num_pad = max_num_predicates - predicate_tensor.shape[0]
			predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
			predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
			predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
			a_query_predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
			a_query_predicate_masks.append(np.expand_dims(predicate_mask, 0))

		if len(a_query_predicate_masks) < max_num_cdfs:
			for _ in range(max_num_cdfs-len(a_query_predicate_masks)):
				a_query_predicate_masks.append(a_query_predicate_masks[0])
				a_query_predicate_tensors.append(a_query_predicate_tensors[0])

		a_query_predicate_tensors = np.vstack(a_query_predicate_tensors)
		a_query_predicate_masks = np.vstack(a_query_predicate_masks)

		cdf_predicate_masks.append(a_query_predicate_masks)
		cdf_predicate_tensors.append(a_query_predicate_tensors)

	cdf_predicate_masks = torch.FloatTensor(np.array(cdf_predicate_masks))
	cdf_predicate_tensors = torch.FloatTensor(np.array(cdf_predicate_tensors))


	cdf_join_masks = []
	cdf_join_tensors = []
	for a_query_cdf in cdf_joins:
		q_query_join_masks = []
		q_query_join_tensors = []
		for join in a_query_cdf:
			join_tensor = np.vstack(join)
			num_pad = max_num_joins - join_tensor.shape[0]
			join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
			join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
			join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
			q_query_join_tensors.append(np.expand_dims(join_tensor, 0))
			q_query_join_masks.append(np.expand_dims(join_mask, 0))

		if len(q_query_join_masks) < max_num_cdfs:
			for _ in range(max_num_cdfs-len(q_query_join_masks)):
				q_query_join_masks.append(q_query_join_masks[0])
				q_query_join_tensors.append(q_query_join_tensors[0])

		q_query_join_tensors = np.vstack(q_query_join_tensors)
		q_query_join_masks = np.vstack(q_query_join_masks)

		cdf_join_tensors.append(q_query_join_tensors)
		cdf_join_masks.append(q_query_join_masks)

	cdf_join_tensors = torch.FloatTensor(np.array(cdf_join_tensors))
	cdf_join_masks = torch.FloatTensor(np.array(cdf_join_masks))

	return cdf_sample_tensors, cdf_predicate_tensors, cdf_join_tensors, cdf_sample_masks, cdf_predicate_masks, cdf_join_masks, cdf_signs_tensors

def make_aug_dataset_cdf_no_padding(cdf_samples, cdf_predicates, cdf_joins, cdf_signs, max_num_joins, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	max_num_cdfs = 0
	for a_cdf_samples in cdf_samples:
		if len(a_cdf_samples) > max_num_cdfs:
			max_num_cdfs = len(a_cdf_samples)

	cdf_sample_masks = []
	cdf_sample_tensors = []

	for a_query_samples in cdf_samples:
		a_query_sample_masks = []
		a_query_sample_tensors = []
		for sample in a_query_samples:
			sample_tensor = np.vstack(sample)
			num_pad = max_num_joins + 1 - sample_tensor.shape[0]
			sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
			sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
			sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
			a_query_sample_masks.append(np.expand_dims(sample_mask, 0))
			a_query_sample_tensors.append(np.expand_dims(sample_tensor, 0))

		cdf_sample_masks.append(a_query_sample_masks)
		cdf_sample_tensors.append(a_query_sample_tensors)


	#### process signs!
	for a_query_signs in cdf_signs:
		if len(a_query_signs) < max_num_cdfs:
			for _ in range(max_num_cdfs - len(a_query_signs)):
				a_query_signs.append(0)
	cdf_signs_tensors = np.vstack(cdf_signs)
	cdf_signs_tensors = torch.FloatTensor(cdf_signs_tensors)
	#### finished processing signs!

	cdf_predicate_masks = []
	cdf_predicate_tensors = []
	for a_query_predicate in cdf_predicates:
		a_query_predicate_masks = []
		a_query_predicate_tensors = []

		for predicate in a_query_predicate:
			predicate_tensor = np.vstack(predicate)
			num_pad = max_num_predicates - predicate_tensor.shape[0]
			predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
			predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
			predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
			a_query_predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
			a_query_predicate_masks.append(np.expand_dims(predicate_mask, 0))

		cdf_predicate_masks.append(a_query_predicate_masks)
		cdf_predicate_tensors.append(a_query_predicate_tensors)

	cdf_join_masks = []
	cdf_join_tensors = []
	for a_query_cdf in cdf_joins:
		q_query_join_masks = []
		q_query_join_tensors = []
		for join in a_query_cdf:
			join_tensor = np.vstack(join)
			num_pad = max_num_joins - join_tensor.shape[0]
			join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
			join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
			join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
			q_query_join_tensors.append(np.expand_dims(join_tensor, 0))
			q_query_join_masks.append(np.expand_dims(join_mask, 0))

		cdf_join_tensors.append(q_query_join_tensors)
		cdf_join_masks.append(q_query_join_masks)

	return cdf_sample_tensors, cdf_predicate_tensors, cdf_join_tensors, cdf_sample_masks, cdf_predicate_masks, cdf_join_masks, cdf_signs_tensors, max_num_cdfs

def get_train_datasets(num_queries, num_materialized_samples, dataset='imdb', trans_op=False, workload_type='in', shift='center'):
	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, candi_label, max_num_joins, max_num_predicates, train_data, test_data, candi_data, \
	ori_predicates_train, ori_samples_train, ori_tables_train, ori_tables_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, \
	num_joins_test, num_predicates_test, table_sets_test, numerical_cols, candi_query_typeids, candi_joins, candi_predicates, candi_tables, candi_samples, min_max_numpreds_per_joingraph = load_and_encode_train_data(
		num_queries, num_materialized_samples, dataset, trans_op=trans_op, workload_type=workload_type, shift=shift)
	train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
	                             max_num_predicates=max_num_predicates)
	print("Created TensorDataset for training data")

	if len(labels_test):
		test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
		                            max_num_predicates=max_num_predicates)
	else:
		test_dataset = []
	print("Created TensorDataset for validation data")

	candi_dataset = make_dataset(*candi_data, labels=candi_label, max_num_joins=max_num_joins,
	                            max_num_predicates=max_num_predicates)

	print("Created TensorDataset for candidate data")

	return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset, candi_dataset, \
	       ori_predicates_train, ori_samples_train, ori_tables_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, num_joins_test, num_predicates_test, table_sets_test, numerical_cols, candi_query_typeids, \
			candi_joins, candi_predicates, candi_tables, candi_samples

def get_train_datasets_w_cdf(num_queries, num_materialized_samples, dataset='imdb', add_one=False, workload_type='in', shift='center'):
	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, candi_label, max_num_joins, max_num_predicates, train_data, test_data, candi_data, \
	ori_predicates_train, ori_samples_train, ori_tables_train, ori_joins_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, \
	num_joins_test, num_predicates_test, table_sets_test, numerical_cols, candi_query_typeids, candi_joins, candi_predicates, candi_tables, candi_samples, min_max_numpreds_per_joingraph = load_and_encode_train_data(
		num_queries, num_materialized_samples, dataset, trans_op=True, add_one=add_one, workload_type=workload_type, shift=shift)
	train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
	                             max_num_predicates=max_num_predicates)
	print("Created TensorDataset for training data")

	if len(labels_test):
		test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
		                            max_num_predicates=max_num_predicates)
	else:
		test_dataset = []
	print("Created TensorDataset for validation data")

	candi_dataset = make_dataset(*candi_data, labels=candi_label, max_num_joins=max_num_joins,
	                            max_num_predicates=max_num_predicates)

	print("Created TensorDataset for candidate data")

	return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset, candi_dataset, \
	       ori_predicates_train, ori_samples_train, ori_tables_train, ori_joins_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, num_joins_test, num_predicates_test, table_sets_test, numerical_cols, candi_query_typeids, \
			candi_joins, candi_predicates, candi_tables, candi_samples, min_max_numpreds_per_joingraph
