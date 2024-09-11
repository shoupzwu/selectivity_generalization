import copy
import numpy as np

import torch
from torch.autograd import Variable

rs = np.random.RandomState(42)


# Helper functions

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

		max_val = np.log(48844.0)

		print("max (label): {}".format(np.max(ori_label)))
		print("max log(label): {}".format(max_val))
	labels_norm = (labels - min_val) / (max_val - min_val)
	# Threshold labels
	labels_norm = np.minimum(labels_norm, 1)
	labels_norm = np.maximum(labels_norm, 0)
	return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val, is_cuda=True):
	if is_cuda:
		labels_norm_np = np.array(labels_norm.cpu(), dtype=np.float32)
	else:
		labels_norm_np = np.array(labels_norm, dtype=np.float32)
	labels = (labels_norm_np * (max_val - min_val)) + min_val
	return np.array(np.round(np.exp(labels)), dtype=np.int64)


def get_card_from_sel(labels_norm, min_val, max_val):
	labels_norm = np.array(labels_norm, dtype=np.float32)
	labels = labels_norm * np.exp(max_val)
	return np.array(np.round(labels), dtype=np.int64)


def encode_data(predicates, column_min_max_vals, column2vec, op2vec):
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

	return predicates_enc


def encode_cdf_data(cdf_predicates, column_min_max_vals, column2vec, op2vec):
	cdf_predicates_enc = []

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

		cdf_predicates_enc.append(predicates_enc)

	# cdf_predicates_enc: [num_queries, num_cdfs(varied), num_predicates(varied)]
	# cdf_joins_enc: [num_queries, num_cdfs(varied), num_joins(varied)]

	return cdf_predicates_enc


def unnormalize_torch(vals, min_val, max_val, is_selectivity=False):
	vals = (vals * (max_val - min_val)) + min_val
	if is_selectivity:
		return torch.exp(vals) /  np.exp(max_val)
	else:
		return torch.exp(vals)


def normalize_torch(vals, min_val, max_val):
	vals = torch.log(vals)
	labels_norm = (vals - min_val) / (max_val - min_val)
	return labels_norm


def qerror_loss(preds, targets, min_val, max_val):
	qerror = []
	preds = unnormalize_torch(preds, min_val, max_val)
	targets = unnormalize_torch(targets, min_val, max_val)

	for i in range(len(targets)):
		if (preds[i] > targets[i]).cpu().data.numpy()[0]:
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])
	return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
	preds = []
	t_total = 0.

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		predicates, targets, predicate_masks, train_ids = data_batch

		if cuda:
			predicates, targets = predicates.cuda(), targets.cuda()
			predicate_masks = predicate_masks.cuda()
		predicates, targets = Variable(predicates), Variable(targets)
		predicate_masks = Variable(predicate_masks)
		outputs = model(predicates, predicate_masks)

		for i in range(outputs.data.shape[0]):
			preds.append(outputs.data[i])

	preds = torch.vstack(preds)
	return preds, t_total


def predict_and_get_labels(model, data_loader, cuda):
	preds = []
	labels = []

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		predicates, targets, predicate_masks, train_ids = data_batch

		if cuda:
			predicates, targets = predicates.cuda(), targets.cuda()
			predicate_masks = predicate_masks.cuda()
		predicates, targets = Variable(predicates), Variable(targets)
		predicate_masks = Variable(predicate_masks)
		outputs = model(predicates, predicate_masks)

		for i in range(outputs.data.shape[0]):
			preds.append(outputs.data[i])
			labels.append(targets.data[i])

	preds = torch.vstack(preds)
	labels = torch.vstack(labels)

	return preds, labels


def print_qerror(preds_unnorm, labels_unnorm, table_card, workload_type='In-Distribution'):
	qerror_res = []

	preds_unnorm = np.squeeze(preds_unnorm)
	labels_unnorm = np.squeeze(labels_unnorm)

	for i in range(len(preds_unnorm)):

		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror_res.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror_res.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

	print("{} Workload, Num of queries {}".format(workload_type, len(preds_unnorm)))
	print("RMSE: {}".format(np.sqrt(np.mean(np.square(preds_unnorm / table_card - labels_unnorm / table_card)))))
	print("Median Qerror: {}".format(np.median(qerror_res)))
	print("90th Qerror: {}".format(np.percentile(qerror_res, 90)))
	print("Max Qerror: {}".format(np.max(qerror_res)))
	print("Mean Qerror: {}".format(np.mean(qerror_res)))

	return None
