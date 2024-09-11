import pandas as pd
import numpy as np
import lw_nn.datasets as datasets
import json
import torch
import copy
import random

dataset = 'census'
table = datasets.LoadCensus()

q_file = open('./queries/census_workload_50000.txt', 'r')
j_obj = json.loads(q_file.readline())

queries = j_obj['query_list']
cards = j_obj['card_list']

cols = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

num_min_max = {0:[17, 90],
               4:[1, 16],
               10:[0, 99999],
               11:[0, 4356],
               12:[1, 99]}

col_to_list = {}
col_to_valid_list = {}
table_card = 48842
for col in range(len(cols)):
	col_val_list = table.columns[col].all_distinct_values
	if pd.isna(col_val_list[0]):
		col_to_list[cols[col]] = col_val_list
		col_to_valid_list[cols[col]] = col_val_list[1:]
	else:
		col_to_list[cols[col]] = col_val_list
		col_to_valid_list[cols[col]] = col_val_list

def extract_sublist(original_list, indices):
	return [original_list[i] for i in indices]

def normalize_var(var, col_name):
	if col_name in num_min_max:
		nor_var = (int(var) - num_min_max[col_name][0])/(num_min_max[col_name][1]- num_min_max[col_name][0])
		return nor_var
	else:
		try:
			nor_var = (np.where(col_to_list[col_name] == var)[0][0] + 1)/ (len(col_to_list[col_name])+1)
			return nor_var
		except:
			print(var)
			print(col_name)
			print(col_to_list[col_name])
			print("An exception occurred")

	# return nor_var

def unnormalize_var(var, col_name):
	if col_name in num_min_max:
		unnor_var = round(var*(num_min_max[col_name][1]- num_min_max[col_name][0]) + num_min_max[col_name][0])
	else:
		var_id = round(var*(len(col_to_list[col_name])+1) - 1)
		unnor_var = col_to_list[col_name][var_id]
	return unnor_var


def unnormalize_torch(vals, min_val, max_val):
	vals = (vals * (max_val - min_val)) + min_val
	return torch.exp(vals)

def get_qerror(preds, targets, num_data, workload_type='In-Distribution'):
	qerror = []
	preds = preds.cpu().data.numpy()
	targets = targets.cpu().data.numpy()

	rmse = np.sqrt(np.mean(np.square(preds - targets)))
	for i in range(len(targets)):
		p_card = round(preds[i] * num_data)
		t_card = round(targets[i] * num_data)
		if p_card <= 1.:
			p_card = 1.

		if (p_card > t_card):
			qerror.append(p_card / t_card)
		else:
			qerror.append(t_card / p_card)

	print("{} Workload: Rmse: {}, median Qerror: {}, 90-th Qerror: {}, max Qerror: {}".format(workload_type, rmse, np.median(qerror), np.quantile(qerror, 0.9), np.max(qerror)))

def check_errors(preds, targets, num_data=50000):
	square_loss = (preds - targets) ** 2
	square_loss_np = square_loss.cpu().data.numpy()
	targets = targets.cpu().data.numpy()

	print("loss: {}".format(torch.sqrt(torch.mean(square_loss))))

	lows =[]
	middles = []
	highs = []

	for gt, error in zip(targets, square_loss_np):
		gt = gt * num_data
		if gt < 5000:
			lows.append(error)
		elif gt < 20000:
			middles.append(error)
		else:
			highs.append(error)
	# print("lows: {}, middles: {}, highs: {}".format(np.max(lows), np.max(middles), np.max(highs)))

def get_tensor_data(queries, signs, sels):
	queries_tensor = np.vstack(queries)
	queries_tensor = torch.FloatTensor(queries_tensor)

	signs_tensor = np.vstack(signs)
	signs_tensor = torch.FloatTensor(signs_tensor)

	sels_tensor = torch.FloatTensor(sels)

	return queries_tensor, signs_tensor, sels_tensor

def find_next_var(var, col_name, direction):
	### onlu used for '=' operator
	if direction == 'up':
		if col_name in num_min_max:
			nor_var = (int(var) + 1 - num_min_max[col_name][0]) / (num_min_max[col_name][1] - num_min_max[col_name][0])
		else:
			nor_var = (np.where(col_to_list[col_name] == var)[0][0] + 2) / (len(col_to_list[col_name])+1)
		return nor_var
	else:
		if col_name in num_min_max:
			nor_var = (int(var) -1 - num_min_max[col_name][0]) / (num_min_max[col_name][1] - num_min_max[col_name][0])
		else:
			nor_var = (np.where(col_to_list[col_name] == var)[0][0]) / (len(col_to_list[col_name])+1)
		return nor_var

def get_train_test_queries(queries, cards, shift_col_id, shift='granularity'):
	in_ids = []
	ood_ids = []
	q_ranges = []
	new_qs = []
	q_sels = []
	qid = 0
	for q, card in zip(queries, cards):
		card = int(card)
		sel = card / float(table_card)
		q_range = []
		new_q = []
		for _ in cols:
			q_range.append([0, 1.])

		for col_name, op, var in zip(q[0], q[1], q[2]):
			col_id = cols.index(col_name)
			norm_var = normalize_var(var, col_name)

			if op == '=':
				if norm_var > 0:
					q_range[col_id][1] = norm_var
					q_range[col_id][0] = find_next_var(var, col_name, 'down')
				else:
					print('wait~!')
					q_range[col_id][1] = find_next_var(var, col_name, 'up')
					q_range[col_id][0] = norm_var
			elif op == '<=':
				q_range[col_id][1] = norm_var
			elif op == '>=':
				q_range[col_id][0] = norm_var

			new_q.append([col_id, op, norm_var])

		new_qs.append(new_q)

		if shift == 'card':  ## card shift
			val_center = np.mean(q_range[shift_col_id])
			val_range = q_range[shift_col_id][1] - q_range[shift_col_id][0]
			if card > 6000:
				in_ids.append(qid)
			elif card < 1000:
				ood_ids.append(qid)
		else:  ## granularity shift
			val_range = q_range[shift_col_id][1] - q_range[shift_col_id][0]
			if val_range > 0.9:
				in_ids.append(qid)
			elif val_range < 0.2:
				ood_ids.append(qid)

		q_range_concate = []
		for col_range in q_range:
			q_range_concate.extend(col_range)

		q_ranges.append(q_range_concate)
		q_sels.append(sel)
		qid += 1

	print(len(ood_ids))
	in_ids = in_ids[:20000]
	ood_ids = ood_ids[:1000]
	return q_ranges, new_qs, q_sels, in_ids, ood_ids