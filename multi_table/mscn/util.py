import copy
import numpy as np

# Helper functions for data processing

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def get_all_column_names(predicates):
	column_names = set()
	for query in predicates:
		for predicate in query:
			if len(predicate) == 3:
				column_name = predicate[0]
				column_names.add(column_name)
	return column_names


def get_all_table_names(tables):
	table_names = set()
	for query in tables:
		for table in query:
			table_names.add(table)
	return table_names


def get_all_operators(predicates):
	operators = set()
	for query in predicates:
		for predicate in query:
			if len(predicate) == 3:
				operator = predicate[1]
				operators.add(operator)
	return operators


def get_all_joins(joins):
	join_set = set()
	for query in joins:
		for join in query:
			join_set.add(join)
	return join_set


def idx_to_onehot(idx, num_elements):
	onehot = np.zeros(num_elements, dtype=np.float32)
	onehot[idx] = 1.
	return onehot


def get_set_encoding(source_set, onehot=True):
	num_elements = len(source_set)
	source_list = list(source_set)
	# Sort list to avoid non-deterministic behavior
	source_list.sort()
	# Build map from s to i
	thing2idx = {s: i for i, s in enumerate(source_list)}
	# Build array (essentially a map from idx to s)
	idx2thing = [s for i, s in enumerate(source_list)]
	if onehot:
		thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
		return thing2vec, idx2thing
	return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
	min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
	for query in predicates:
		for predicate in query:
			if len(predicate) == 3:
				column_name = predicate[0]
				val = float(predicate[2])
				if val < min_max_vals[column_name][0]:
					min_max_vals[column_name][0] = val
				if val > min_max_vals[column_name][1]:
					min_max_vals[column_name][1] = val
	return min_max_vals


def normalize_data(val, column_name, column_min_max_vals, return_array=True):
	min_val = column_min_max_vals[column_name][0]
	max_val = column_min_max_vals[column_name][1]
	val = float(val)
	val_norm = 0.0
	if max_val > min_val:
		val_norm = (val - min_val) / (max_val - min_val)

	if return_array:
		return np.array(val_norm, dtype=np.float32)
	else:
		return val_norm


def unnormalize_data(val_norm, column_name, column_min_max_vals):
	min_val = column_min_max_vals[column_name][0]
	max_val = column_min_max_vals[column_name][1]
	if val_norm == 0:
		return min_val
	if val_norm == 1:
		return max_val

	val_norm = float(val_norm)
	val = (val_norm * (max_val - min_val)) + min_val

	return round(val)


def normalize_labels(labels, min_val=None, max_val=None, dataset='imdb', add_one=False):
	ori_label = np.array([float(l) for l in labels])
	if add_one:
		labels = np.array([np.log(float(l) + 1.) for l in labels])
	else:
		labels = np.array([np.log(float(l)) for l in labels])
	if min_val is None:
		min_val = labels.min()
		min_val = 0.
		print("min (label): {}".format(np.min(ori_label)))
		print("min log(label): {}".format(min_val))
	if max_val is None:
		max_val = labels.max()

		if dataset == 'imdb':
			if add_one:
				max_val = np.log(2127734966462.0 + 1.0)
			else:
				max_val = np.log(2127734966462.0)
		elif dataset == 'dsb':
			if add_one:
				max_val = np.log(100211287.0 + 1.0)
			else:
				max_val = np.log(100211287.0)

		print("max (label): {}".format(np.max(ori_label)))
		print("max log(label): {}".format(max_val))
	labels_norm = (labels - min_val) / (max_val - min_val)
	# Threshold labels
	labels_norm = np.minimum(labels_norm, 1)
	labels_norm = np.maximum(labels_norm, 0)
	return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val, is_cuda=True):
	if is_cuda:
		labels_norm = np.array(labels_norm.cpu(), dtype=np.float32)
	else:
		labels_norm = np.array(labels_norm, dtype=np.float32)
	labels = (labels_norm * (max_val - min_val)) + min_val
	return np.array(np.round(np.exp(labels)), dtype=np.int64)

def get_card_from_sel(labels_norm, min_val, max_val):
	labels_norm = np.array(labels_norm, dtype=np.float32)
	labels = labels_norm * np.exp(max_val)
	return np.array(np.round(labels), dtype=np.int64)

def encode_samples(tables, samples, table2vec):
	samples_enc = []
	for i, query in enumerate(tables):
		samples_enc.append(list())
		for j, table in enumerate(query):
			sample_vec = []
			# Append table one-hot vector
			sample_vec.append(table2vec[table])
			# Append bit vector
			sample_vec.append(samples[i][j])
			sample_vec = np.hstack(sample_vec)
			samples_enc[i].append(sample_vec)
	return samples_enc


def encode_samples2(tables, samples, table2vec, chosen_tables, new_samples):
	samples_enc = []
	sub_list = []
	original_sample = []
	for i, query in enumerate(tables):
		samples_enc.append(list())
		chosen_table = chosen_tables[i]
		for j, table in enumerate(query):
			sample_vec = []
			# Append table one-hot vector
			sample_vec.append(table2vec[table])
			# Append bit vector
			table = table.split(' ')[1]
			if chosen_table != table:
				sample_vec.append(samples[i][j])
			else:
				sample_vec.append(new_samples[i][0])
				sub_list.append(new_samples[i][0])
				original_sample = samples[i][j]
			sample_vec = np.hstack(sample_vec)
			samples_enc[i].append(sample_vec)

	is_bug = False

	return samples_enc, is_bug

def encode_samples3(tables, samples, table2vec, chosen_tables, new_samples):
	samples_enc = []
	for i, query in enumerate(tables):
		samples_enc.append(list())
		chosen_table = chosen_tables[i]
		for j, table in enumerate(query):
			sample_vec = []
			# Append table one-hot vector
			sample_vec.append(table2vec[table])
			# Append bit vector
			table = table.split(' ')[1]
			if chosen_table != table:
				if table == 't':
					sample_vec.append(np.ones(1000, dtype=bool))
				else:
					sample_vec.append(samples[i][j])
			else:
				sample_vec.append(new_samples[i][0])
			sample_vec = np.hstack(sample_vec)
			samples_enc[i].append(sample_vec)

	is_bug = False

	return samples_enc, is_bug

def encode_cdf_samples(cdf_tables, cdf_samples, table2vec, is_ori_table=False):
	cdf_samples_enc = []

	if not is_ori_table:
		for i, q_table_list in enumerate(cdf_tables):
			samples_enc = []
			for j, a_cdf_tables in enumerate(q_table_list):
				samples_enc.append(list())
				for k, table in enumerate(a_cdf_tables):
					sample_vec = []
					# Append table one-hot vector
					sample_vec.append(table2vec[table])
					# Append bit vector
					sample_vec.append(cdf_samples[i][j][k])
					sample_vec = np.hstack(sample_vec)
					samples_enc[j].append(sample_vec)
			cdf_samples_enc.append(samples_enc)
	else:
		for i, q_bitmap_list in enumerate(cdf_samples):
			samples_enc = []
			for j, a_cdf_bitmaps in enumerate(q_bitmap_list):
				samples_enc.append(list())
				for k, bitmap in enumerate(a_cdf_bitmaps):
					sample_vec = []
					# Append table one-hot vector
					table = cdf_tables[i][k]
					sample_vec.append(table2vec[table])
					# Append bit vector
					sample_vec.append(bitmap)
					sample_vec = np.hstack(sample_vec)
					samples_enc[j].append(sample_vec)
			cdf_samples_enc.append(samples_enc)

	# cdf_samples_enc: [num_queries, num_cdfs(varied), num_tables(varied)]
	return cdf_samples_enc


def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
	predicates_enc = []
	joins_enc = []
	for i, query in enumerate(predicates):
		predicates_enc.append(list())
		joins_enc.append(list())
		for predicate in query:
			if len(predicate) == 3:
				# Proper predicate
				column = predicate[0]
				operator = predicate[1]
				val = predicate[2]
				norm_val = normalize_data(val, column, column_min_max_vals)

				pred_vec = []
				pred_vec.append(column2vec[column])
				pred_vec.append(op2vec[operator])
				pred_vec.append(norm_val)
				pred_vec = np.hstack(pred_vec)
			else:
				pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

			predicates_enc[i].append(pred_vec)

		for predicate in joins[i]:
			# Join instruction
			join_vec = join2vec[predicate]
			joins_enc[i].append(join_vec)
	return predicates_enc, joins_enc

def encode_cdf_data(cdf_predicates, cdf_joins, column_min_max_vals, column2vec, op2vec, join2vec, is_join_same_shape=True):
	cdf_predicates_enc = []
	cdf_joins_enc = []

	for k, predicates in enumerate(cdf_predicates):
		predicates_enc = []
		joins_enc = []

		for i, query in enumerate(predicates):
			predicates_enc.append(list())
			joins_enc.append(list())

			if len(query) > 0:
				for predicate in query:
					if len(predicate) == 3:
						# Proper predicate
						column = predicate[0]
						operator = predicate[1]
						val = predicate[2]
						norm_val = normalize_data(val, column, column_min_max_vals)

						pred_vec = []
						pred_vec.append(column2vec[column])
						pred_vec.append(op2vec[operator])
						pred_vec.append(norm_val)
						pred_vec = np.hstack(pred_vec)
					else:
						pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

					predicates_enc[i].append(pred_vec)
			else:
				pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))
				predicates_enc[i].append(pred_vec)

			if is_join_same_shape:
				for predicate in cdf_joins[k][i]:
					# Join instruction
					join_vec = join2vec[predicate]
					joins_enc[i].append(join_vec)
			else:
				for predicate in cdf_joins[k]:
					# Join instruction
					join_vec = join2vec[predicate]
					joins_enc[i].append(join_vec)

		cdf_predicates_enc.append(predicates_enc)
		cdf_joins_enc.append(joins_enc)

	# cdf_predicates_enc: [num_queries, num_cdfs(varied), num_predicates(varied)]
	# cdf_joins_enc: [num_queries, num_cdfs(varied), num_joins(varied)]

	return cdf_predicates_enc, cdf_joins_enc