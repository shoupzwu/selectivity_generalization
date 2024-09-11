import copy
import csv
import random
import numpy as np
import torch
from torch.utils.data import dataset

from mscn.utils import *

def load_data(file_name, num_train, column_min_max_vals=None, trans_op=False, workload_type='in', shift='granularity'):
	predicates = []
	labels = []
	q_size = []

	q_center = []
	col_range = []

	shifting_var = '0'

	shifting_var_min = int(column_min_max_vals[shifting_var][0])
	shifting_var_max = int(column_min_max_vals[shifting_var][1])

	random.seed(42)

	valid_qids = []  # for filtering samples

	num_predicates = []
	numerical_cols = []

	num_all_qs = 0
	# Load queries
	with open(file_name + ".csv", 'r') as f:
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
			num_predicates.append(len(row[2].split(',')) / 3)

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
			labels.append(row[3])

			#### get query range and year center

			q_range = copy.deepcopy(column_min_max_vals)
			a_predicate = predicates[-1]
			shift_col_range = [0., 1.]
			col2range = {}

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
							if int(val) == shifting_var_min:
								shift_col_range = [(int(val) - shifting_var_min + 1) / (shifting_var_max - shifting_var_min + 1),
								              (int(val) - shifting_var_min + 2) / (shifting_var_max - shifting_var_min + 1)]
							else:
								shift_col_range = [(int(val) - shifting_var_min) / (shifting_var_max - shifting_var_min + 1),
								              (int(val) - shifting_var_min + 1) / (shifting_var_max - shifting_var_min + 1)]
						elif op == '<=':
							shift_col_range[1] = (int(val) - shifting_var_min) / (shifting_var_max - shifting_var_min)
						elif op == '>=':
							shift_col_range[0] = (int(val) - shifting_var_min) / (shifting_var_max - shifting_var_min)

					var_min = int(column_min_max_vals[col][0])
					var_max = int(column_min_max_vals[col][1])

					if op == '=':
						if val == var_min:
							col2range[col] = [(int(val) - var_min) / (var_max - var_min + 1),
							                  (int(val) - var_min + 1) / (var_max - var_min + 1)]
					elif op == '<=':
						if col not in col2range:
							col2range[col] = [0, (int(val) - var_min + 1) / (var_max - var_min + 1)]
						else:
							col2range[col][1] = (int(val) - var_min + 1) / (var_max - var_min + 1)
					elif op == '>=':
						if col not in col2range:
							col2range[col] = [(int(val) - var_min + 1) / (var_max - var_min + 1), 1]
						else:
							col2range[col][0] = (int(val) - var_min + 1) / (var_max - var_min + 1)

			range_size = 1
			for col in q_range:
				lower = int(q_range[col][0])
				upper = int(q_range[col][1])
				range_size *= (upper - lower + 1)

			q_size.append(range_size)
			q_center.append(np.mean(shift_col_range))
			col_range.append(shift_col_range[1] - shift_col_range[0])

			if len(valid_qids) >= num_train:
				break

	candi_predicates = []
	candi_label = []
	candi_num_predicates = []
	candi_ori_qs = []

	candi_file_name = './queries/census-mscn-50000'

	### load in-distribution workload
	with open(candi_file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		num_all_candi_qs = len(data_raw)
		for qid, row in enumerate(data_raw):
			candi_ori_qs.append('#'.join(row))
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

			candi_num_predicates.append(len(row[2].split(',')) / 3)

	print("Loaded candidate queries")

	all_train_ids = []
	test_size = 0

	labels = [float(l) for l in labels]

	if shift == 'card': ### card shift
		if workload_type == 'in':
			filtered_paired_sorted = [id for id, center in enumerate(labels) if (center > 6000)]
		elif workload_type == 'ood':
			filtered_paired_sorted = [id for id, center in enumerate(labels) if (center < 1000)]
	else: # granularity shift
		if workload_type == 'in':
			filtered_paired_sorted = [id for id, a_q_range in enumerate(col_range) if (a_q_range > 0.9)]
		elif workload_type == 'ood':
			filtered_paired_sorted = [id for id, a_q_range in enumerate(col_range) if (a_q_range < 0.2)]

	all_train_ids.extend(filtered_paired_sorted)

	all_train_ids = all_train_ids[:20000]
	train_num = int(0.9 * (len(all_train_ids)))
	train_ids = all_train_ids[:train_num]

	print('len of train_ids {}'.format(len(train_ids)))

	new_predicates = [predicates[qid] for qid in train_ids]
	new_num_predicates = [num_predicates[qid] for qid in train_ids]
	new_label = [labels[qid] for qid in train_ids]

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

	candi_q_ids = all_train_ids[train_num:]

	candi_predicates = [candi_predicates[qid] for qid in candi_q_ids]
	candi_label = [candi_label[qid] for qid in candi_q_ids]
	candi_num_predicates = [candi_num_predicates[qid] for qid in candi_q_ids]

	return new_predicates, new_label, new_num_predicates, numerical_cols, test_size, candi_predicates, candi_label, candi_num_predicates

def load_ood_data(file_name, num_train, column_min_max_vals=None, trans_op=False, workload_type='in', shift='center'):
	predicates = []
	labels = []
	q_size = []

	q_center = []
	col_range = []

	shifting_var = '0'

	shift_var_min = int(column_min_max_vals[shifting_var][0])
	shift_var_max = int(column_min_max_vals[shifting_var][1])

	random.seed(42)

	valid_qids = []  # for filtering samples

	num_predicates = []
	numerical_cols = []

	num_all_qs = 0
	# Load queries
	with open(file_name + ".csv", 'r') as f:
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
			num_predicates.append(len(row[2].split(',')) / 3)

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
			labels.append(row[3])

			#### get query range and year center

			q_range = copy.deepcopy(column_min_max_vals)
			a_predicate = predicates[-1]
			year_range = [0, 1]

			col2range = {}

			if len(a_predicate) > 1:
				for qid in range(int(len(a_predicate) / 3)):
					col = a_predicate[qid * 3]
					op = a_predicate[qid * 3 + 1]
					val = a_predicate[qid * 3 + 2]

					if col == shifting_var:
						if op == '=':
							if int(val) == shift_var_min:
								year_range = [(int(val) - shift_var_min + 1) / (shift_var_max - shift_var_min + 1),
								              (int(val) - shift_var_min + 2) / (shift_var_max - shift_var_min + 1)]
							else:
								year_range = [(int(val) - shift_var_min) / (shift_var_max - shift_var_min + 1),
								              (int(val) - shift_var_min + 1) / (shift_var_max - shift_var_min + 1)]
						elif op == '<=':
							year_range[1] = (int(val) - shift_var_min ) / (shift_var_max - shift_var_min )
						elif op == '>=':
							year_range[0] = (int(val) - shift_var_min ) / (shift_var_max - shift_var_min)
					var_min = int(column_min_max_vals[col][0])
					var_max = int(column_min_max_vals[col][1])

					if op == '=':
						if val == var_min:
							col2range[col] = [(int(val) - var_min) / (var_max - var_min + 1),
							                  (int(val) - var_min + 1) / (var_max - var_min + 1)]
					elif op == '<=':
						if col not in col2range:
							col2range[col] = [0, (int(val) - var_min + 1) / (var_max - var_min + 1)]
						else:
							col2range[col][1] = (int(val) - var_min + 1) / (var_max - var_min + 1)
					elif op == '>=':
						if col not in col2range:
							col2range[col] = [(int(val) - var_min + 1) / (var_max - var_min + 1), 1]
						else:
							col2range[col][0] = (int(val) - var_min + 1) / (var_max - var_min + 1)

			range_size = 1
			for col in q_range:
				lower = int(q_range[col][0])
				upper = int(q_range[col][1])
				range_size *= (upper - lower + 1)

			q_size.append(range_size)
			q_center.append(np.mean(year_range))
			col_range.append(year_range[1] - year_range[0])

			if len(valid_qids) >= num_train:
				break

	train_ids = []

	labels = [float(l) for l in labels]
	if shift == 'card':  # card shift
		if workload_type == 'in':
			filtered_paired_sorted = [id for id, center in enumerate(labels) if (center > 6000)]
		elif workload_type == 'ood':
			filtered_paired_sorted = [id for id, center in enumerate(labels) if (center < 1000)]
	else:  # granularity shift
		if workload_type == 'in':
			filtered_paired_sorted = [id for id, a_q_range in enumerate(col_range) if (a_q_range > 0.9)]
		elif workload_type == 'ood':
			filtered_paired_sorted = [id for id, a_q_range in enumerate(col_range) if (a_q_range < 0.2)]

	train_ids.extend(filtered_paired_sorted)

	print('len of train_ids {}'.format(len(train_ids)))

	new_predicates = [predicates[qid] for qid in train_ids]
	new_label = [labels[qid] for qid in train_ids]

	# Split predicates
	new_predicates = [list(chunks(d, 3)) for d in new_predicates]

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
	return new_predicates[:1000], new_label[:1000]

def load_and_encode_train_data(num_train, trans_op=False, add_one=False, workload_type='in', shift='center'):

	file_name_queries = "queries/census-mscn-50000"
	file_name_column_min_max_vals = "data/column_min_max_vals_census.csv"

	# Get min and max values for each column
	with open(file_name_column_min_max_vals, 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
		column_min_max_vals = {}
		for i, row in enumerate(data_raw):
			if i == 0:
				continue
			column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

	predicates, label, num_predicates, numerical_cols, test_size, candi_predicates, candi_label, candi_num_predicates = load_data(
		file_name_queries, num_train, column_min_max_vals, trans_op=trans_op, workload_type=workload_type, shift=shift)

	label = np.array([float(l) for l in label])

	# Get column name dict
	column_names = get_all_column_names(predicates)
	column2vec, idx2column = get_set_encoding(column_names)

	# Get operator name dict
	operators = get_all_operators(predicates)
	op2vec, idx2op = get_set_encoding(operators)

	predicates_enc = encode_data(predicates, column_min_max_vals, column2vec, op2vec)
	label_norm, min_val, max_val = normalize_labels(label, dataset=dataset, add_one=add_one)

	candi_predicates_enc = encode_data(candi_predicates, column_min_max_vals, column2vec, op2vec)
	candi_label_norm, min_val, max_val = normalize_labels(candi_label, min_val, max_val, dataset=dataset,
	                                                      add_one=add_one)
	candi_label_norm = np.array(candi_label_norm)
	candi_query_typeids = {}

	# Split in training and validation samples
	num_queries = len(predicates_enc)
	num_train = num_queries
	num_train_oracle = num_train

	predicates_train = predicates_enc[:num_train_oracle]
	labels_train = list(label_norm[:num_train_oracle])

	ori_predicates_train = predicates[:num_train_oracle]

	num_test = 0
	ori_predicates_test = predicates[num_train:num_train + num_test]

	num_predicates_train = num_predicates[:num_train_oracle]

	labels_train = np.array(labels_train)

	predicates_test = predicates_enc[num_train:num_train + num_test]
	labels_test = label_norm[num_train:num_train + num_test]
	labels_test = np.array(labels_test)

	num_predicates_test = num_predicates[num_train:num_train + num_test]

	print("Number of training samples: {}".format(len(labels_train)))
	print("Number of validation samples: {}".format(len(labels_test)))

	if len(candi_predicates_enc):
		max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in candi_predicates_enc]))
	else:
		max_num_predicates = max([len(p) for p in predicates_train])

	dicts = [column2vec, op2vec]
	train_data = [predicates_train]
	test_data = [predicates_test]
	candi_data = [candi_predicates_enc]

	return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, candi_label_norm, max_num_predicates, train_data, test_data, candi_data, \
	       ori_predicates_train, ori_predicates_test, num_predicates_train, num_predicates_test, numerical_cols, candi_query_typeids, candi_predicates,

def make_dataset(predicates, labels, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

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

	target_tensor = torch.FloatTensor(labels)

	ids_tensor = torch.IntTensor(np.arange(len(labels)))

	return dataset.TensorDataset(predicate_tensors, target_tensor,
	                             predicate_masks, ids_tensor)

def make_aug_dataset(predicates, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""
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

	return predicate_tensors, predicate_masks

def make_aug_dataset_cdf(cdf_predicates, cdf_signs, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	max_num_cdfs = 0
	for a_cdf_samples in cdf_predicates:
		if len(a_cdf_samples) > max_num_cdfs:
			max_num_cdfs = len(a_cdf_samples)

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
			for _ in range(max_num_cdfs - len(a_query_predicate_masks)):
				a_query_predicate_masks.append(a_query_predicate_masks[0])
				a_query_predicate_tensors.append(a_query_predicate_tensors[0])

		a_query_predicate_tensors = np.vstack(a_query_predicate_tensors)
		a_query_predicate_masks = np.vstack(a_query_predicate_masks)

		cdf_predicate_masks.append(a_query_predicate_masks)
		cdf_predicate_tensors.append(a_query_predicate_tensors)

	cdf_predicate_masks = torch.FloatTensor(np.array(cdf_predicate_masks))
	cdf_predicate_tensors = torch.FloatTensor(np.array(cdf_predicate_tensors))

	return cdf_predicate_tensors, cdf_predicate_masks, cdf_signs_tensors

def make_aug_dataset_cdf_no_padding(cdf_predicates, cdf_signs, max_num_predicates):
	"""Add zero-padding and wrap as tensor dataset."""

	max_num_cdfs = 0
	for a_cdf_samples in cdf_predicates:
		if len(a_cdf_samples) > max_num_cdfs:
			max_num_cdfs = len(a_cdf_samples)

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

	return cdf_predicate_tensors, cdf_predicate_masks, cdf_signs_tensors, max_num_cdfs

def get_train_datasets(num_queries, trans_op=False, workload_type='in', shift='center'):
	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, candi_label, max_num_predicates, train_data, test_data, candi_data, \
	ori_predicates_train, ori_predicates_test, num_predicates_train, num_predicates_test, numerical_cols, candi_query_typeids, candi_predicates, = load_and_encode_train_data(
		num_queries, trans_op=trans_op, workload_type=workload_type, shift=shift)
	train_dataset = make_dataset(*train_data, labels=labels_train, max_num_predicates=max_num_predicates)
	print("Created TensorDataset for training data")

	if len(labels_test):
		test_dataset = make_dataset(*test_data, labels=labels_test, max_num_predicates=max_num_predicates)
	else:
		test_dataset = []
	print("Created TensorDataset for validation data")

	candi_dataset = make_dataset(*candi_data, labels=candi_label, max_num_predicates=max_num_predicates)

	print("Created TensorDataset for candidate data")

	return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_dataset, test_dataset, candi_dataset, \
	       ori_predicates_train, ori_predicates_test, num_predicates_train, num_predicates_test, numerical_cols, candi_query_typeids, candi_predicates
