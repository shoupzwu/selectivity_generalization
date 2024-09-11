import argparse
import random
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from multiprocessing import Process, Manager, Event

from mscn.util import *
from mscn.data import get_train_datasets_w_cdf, make_dataset, \
	load_ood_data, make_aug_dataset_cdf_no_padding
from mscn.model import SetConv
import mscn.get_bitmap_dsb as get_bitmap

from SeConCDF.dsb_utilities import  *

rs = np.random.RandomState(42)

def unnormalize_torch(vals, min_val, max_val, is_log_scale=False):
	vals = (vals * (max_val - min_val)) + min_val
	if not is_log_scale:
		return torch.exp(vals) / np.exp(max_val)
	else:
		return vals - max_val

def normalize_torch(vals, min_val, max_val):
	vals = torch.log(vals)
	labels_norm = (vals - min_val) / (max_val - min_val)
	return labels_norm


def predict(model, data_loader, cuda):
	preds = []
	t_total = 0.

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch
		if cuda:
			samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
			sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
		samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
			targets)
		sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
			join_masks)

		t = time.time()
		outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
		t_total += time.time() - t

		for i in range(outputs.data.shape[0]):
			preds.append(outputs.data[i])

	return preds, t_total

def predict_and_get_labels(model, data_loader, cuda):
	preds = []
	labels = []

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch
		if cuda:
			samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
			sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
		samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
			targets)
		sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
			join_masks)

		outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)

		for i in range(outputs.data.shape[0]):
			preds.append(outputs.data[i])
			labels.append(targets.data[i])

	preds = torch.stack(preds)
	labels = torch.stack(labels)

	return preds, labels

def print_qerror(preds_unnorm, labels_unnorm, max_val):
	qerror_res = []

	preds_unnorm = np.squeeze(preds_unnorm)
	for i in range(len(preds_unnorm)):

		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror_res.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror_res.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

	print("num of queries {}".format(len(preds_unnorm)))
	print("Median: {}".format(np.median(qerror_res)))
	print("90th percentile: {}".format(np.percentile(qerror_res, 90)))
	print("Max: {}".format(np.max(qerror_res)))
	print("Mean: {}".format(np.mean(qerror_res)))

	max_card = np.exp(max_val)
	print("RMSE: {}".format(np.sqrt(np.mean(np.square(preds_unnorm/max_card - labels_unnorm/max_card)))))

	return qerror_res

def get_top_masks(preds, gts, min_sel, topK, cuda):
	preds_num = torch.squeeze(preds).data.numpy()
	gts_num = torch.squeeze(gts).data.numpy()

	q_error = []

	for i in range(len(preds_num)):
		pred = preds_num[i]
		gt = gts_num[i]

		if pred < min_sel:
			pred = min_sel

		if gt < min_sel:
			gt = min_sel

		if pred > gt:
			q_error.append(pred / gt)
		else:
			q_error.append(gt / pred)

	topk_list = sorted(range(len(q_error)), key=lambda x: q_error[x], reverse=True)[:topK]

	mask = np.zeros(len(q_error))
	for qid in topk_list:
		mask[qid] = 1

	if cuda:
		return torch.FloatTensor(mask).cuda()
	else:
		return torch.FloatTensor(mask)


def train_and_predict(num_queries, num_epochs, batch_size, hid_units, cuda, table_to_df, weight, shift='center'):
	# Load training and validation data
	num_materialized_samples = 1000

	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_validation, max_num_joins, max_num_predicates, train_data, test_data, candi_data, \
	ori_predicates_train, ori_samples_train, ori_tables_train, ori_joins_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, num_joins_test, num_predicates_test, table_sets_test, \
	numerical_cols, candi_query_typeids, candi_joins, candi_predicates, candi_tables, candi_samples, min_max_numpreds_per_joingraph = get_train_datasets_w_cdf(
		num_queries, num_materialized_samples, dataset='dsb', add_one=True, workload_type='in', shift=shift)

	table2vec, column2vec, op2vec, join2vec = dicts

	# Train model
	sample_feats = len(table2vec) + num_materialized_samples
	predicate_feats = len(column2vec) + len(op2vec) + 1
	join_feats = len(join2vec)

	model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	if cuda:
		torch.cuda.empty_cache()
		model = model.to('cuda:0')

	train_data_loader = DataLoader(train_data, batch_size=batch_size)
	candi_data_loader = DataLoader(candi_data, batch_size=batch_size)

	#### load out-of-distribution workload

	workload_name3 = 'dsb'
	file_name = "workloads/" + workload_name3
	joins3, predicates3, tables3, samples3, label3 = load_ood_data(file_name, num_materialized_samples, num_queries,dataset='dsb', column_min_max_vals=column_min_max_vals,
	                                                               trans_op=True, shift=shift)
	# Get feature encoding and proper normalization
	samples_test3 = encode_samples(tables3, samples3, table2vec)
	predicates_test3, joins_test3 = encode_data(predicates3, joins3, column_min_max_vals, column2vec, op2vec, join2vec)
	labels_test3, _, _ = normalize_labels(label3, min_val, max_val, add_one=True)

	label3 = label3 + 1.

	test_max_num_predicates3 = max([len(p) for p in predicates_test3])
	test_max_num_joins3 = max([len(j) for j in joins_test3])

	# Get test set predictions
	test_data3 = make_dataset(samples_test3, predicates_test3, joins_test3, labels_test3, test_max_num_joins3,
	                          test_max_num_predicates3)
	test_data_loader3 = DataLoader(test_data3, batch_size=batch_size)

	# Get bitmap list
	train_cdfs_cache = []
	num_qs = 0
	burn_in = -1

	for batch_idx, data_batch in enumerate(train_data_loader):
		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch
		train_ids = train_ids.numpy()
		batch_predicates = [ori_predicates_train[x] for x in train_ids]
		batch_tables = [ori_tables_train[x] for x in train_ids]
		batch_joins = [ori_joins_train[x] for x in train_ids]

		train_cdfs = []
		train_signs = []
		mask_primitives = []

		for filters in batch_predicates:
			q_cdfs, q_signs = query2cdfs(filters, column_min_max_vals)
			train_cdfs.append(q_cdfs)
			train_signs.append(q_signs)
			if len(q_signs) == 1:
				mask_primitives.append(0)
			else:
				mask_primitives.append(1)

		train_cdf_bitmap_list = get_bitmap.cdf_compute_bitmap(table_to_df, batch_tables, train_cdfs)

		train_cdf_bitmap_list = encode_cdf_samples(batch_tables, train_cdf_bitmap_list, table2vec,
		                                           is_ori_table=True)

		train_cdf_predicates, train_cdf_joins = encode_cdf_data(train_cdfs, batch_joins,
		                                                        column_min_max_vals,
		                                                        column2vec, op2vec, join2vec,
		                                                        is_join_same_shape=False)

		train_cdf_sample_list, train_cdf_predicate_list, train_cdf_join_list, train_cdf_sample_masks, train_cdf_predicate_masks, \
		train_cdf_join_masks, train_cdf_signs_tensors, max_num_cdfs = make_aug_dataset_cdf_no_padding(
			train_cdf_bitmap_list,
			train_cdf_predicates,
			train_cdf_joins, train_signs,
			max_num_joins, max_num_predicates)

		train_cdfs_cache.append(
			[train_cdf_sample_list, train_cdf_predicate_list, train_cdf_join_list, train_cdf_sample_masks,
			 train_cdf_predicate_masks, train_cdf_join_masks, train_cdf_signs_tensors, max_num_cdfs, mask_primitives])

		num_qs += len(batch_predicates)

	aug_bs = 2 * batch_size
	with Manager() as manager:
		shared_batch = manager.list()  # Managed dictionary to hold the current data batch
		lock = manager.Lock()
		stop_event = Event()
		updated_event = Event()
		sampler = Process(target=sample_batch,
		                  args=(aug_bs, column_min_max_vals, max_num_predicates, max_num_joins, table_to_df,
		                        shared_batch, table2vec, column2vec, op2vec, join2vec, cuda, lock, stop_event, updated_event, min_max_numpreds_per_joingraph,))
		sampler.start()
		for epoch in range(num_epochs):
			print('epoch: {}'.format(epoch))
			loss_total = 0.
			num_qs = 0
			model.train()
			for batch_idx, data_batch in enumerate(train_data_loader):

				samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch

				train_ids = train_ids.numpy()
				batch_predicates = [ori_predicates_train[x] for x in train_ids]

				num_qs += len(batch_predicates)
				curr_batch_size = samples.shape[0]

				if cuda:
					samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
					sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
				samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(
					joins), Variable(
					targets)
				sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
					join_masks)

				optimizer.zero_grad()
				outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
				loss = torch.sqrt(torch.mean(torch.square((torch.squeeze(outputs) - torch.squeeze(targets.float())))))

				#### generate cdf loss
				train_cdfs = []
				train_signs = []
				for filters in batch_predicates:
					q_cdfs, q_signs = query2cdfs(filters, column_min_max_vals)
					train_cdfs.append(q_cdfs)
					train_signs.append(q_signs)

				train_cdf_sample_tensors, train_cdf_predicate_tensors, train_cdf_join_tensors, train_cdf_sample_masks, train_cdf_predicate_masks, \
				train_cdf_join_masks, train_cdf_signs_tensors, max_num_cdfs, mask_primitives = train_cdfs_cache[batch_idx]

				batch_cdf_sample_tensors = []
				batch_cdf_sample_masks = []
				for a_query_sample_tensors_no_pad, a_query_sample_masks_no_pad in zip(train_cdf_sample_tensors,
				                                                                      train_cdf_sample_masks):
					a_query_sample_tensors = copy.copy(a_query_sample_tensors_no_pad)
					a_query_sample_masks = copy.copy(a_query_sample_masks_no_pad)

					num_to_add = max_num_cdfs - len(a_query_sample_masks)
					if num_to_add > 0:
						a_query_sample_tensors.extend([a_query_sample_tensors[0]] * num_to_add)
						a_query_sample_masks.extend([a_query_sample_masks[0]] * num_to_add)

					a_query_sample_tensors = np.vstack(a_query_sample_tensors)
					a_query_sample_masks = np.vstack(a_query_sample_masks)

					batch_cdf_sample_tensors.append(a_query_sample_tensors)
					batch_cdf_sample_masks.append(a_query_sample_masks)

				batch_cdf_sample_tensors = torch.from_numpy(np.array(batch_cdf_sample_tensors)).float()
				batch_cdf_sample_masks = torch.from_numpy(np.array(batch_cdf_sample_masks)).float()
				mask_primitives = torch.from_numpy(np.array(mask_primitives)).float()

				batch_cdf_joins_tensors = []
				batch_cdf_joins_masks = []
				for a_query_joins_tensors_no_pad, a_query_joins_masks_no_pad in zip(train_cdf_join_tensors,
				                                                                    train_cdf_join_masks):
					a_query_joins_tensors = copy.copy(a_query_joins_tensors_no_pad)
					a_query_joins_masks = copy.copy(a_query_joins_masks_no_pad)

					num_to_add = max_num_cdfs - len(a_query_joins_masks)
					if num_to_add > 0:
						a_query_joins_tensors.extend([a_query_joins_tensors[0]] * num_to_add)
						a_query_joins_masks.extend([a_query_joins_masks[0]] * num_to_add)

					a_query_joins_tensors = np.vstack(a_query_joins_tensors)
					a_query_joins_masks = np.vstack(a_query_joins_masks)

					batch_cdf_joins_tensors.append(a_query_joins_tensors)
					batch_cdf_joins_masks.append(a_query_joins_masks)

				batch_cdf_joins_tensors = torch.from_numpy(np.array(batch_cdf_joins_tensors)).float()
				batch_cdf_joins_masks = torch.from_numpy(np.array(batch_cdf_joins_masks)).float()

				batch_cdf_predicates_tensors = []
				batch_cdf_predicates_masks = []
				for a_query_predicates_tensors_no_pad, a_query_predicates_masks_no_pad in zip(
						train_cdf_predicate_tensors,
						train_cdf_predicate_masks):
					a_query_predicates_tensors = copy.copy(a_query_predicates_tensors_no_pad)
					a_query_predicates_masks = copy.copy(a_query_predicates_masks_no_pad)

					num_to_add = max_num_cdfs - len(a_query_predicates_masks)
					if num_to_add > 0:
						a_query_predicates_tensors.extend([a_query_predicates_tensors[0]] * num_to_add)
						a_query_predicates_masks.extend([a_query_predicates_masks[0]] * num_to_add)

					a_query_predicates_tensors = np.vstack(a_query_predicates_tensors)
					a_query_predicates_masks = np.vstack(a_query_predicates_masks)

					batch_cdf_predicates_tensors.append(a_query_predicates_tensors)
					batch_cdf_predicates_masks.append(a_query_predicates_masks)

				batch_cdf_predicates_tensors = torch.from_numpy(np.array(batch_cdf_predicates_tensors)).float()
				batch_cdf_predicates_masks = torch.from_numpy(np.array(batch_cdf_predicates_masks)).float()

				### finish getting inputs

				train_max_num_cdfs = batch_cdf_sample_tensors.shape[1]

				if cuda:
					batch_cdf_sample_tensors, batch_cdf_predicates_tensors, batch_cdf_joins_tensors, mask_primitives = batch_cdf_sample_tensors.cuda(), \
					                                                                                                 batch_cdf_predicates_tensors.cuda(), \
					                                                                                                 batch_cdf_joins_tensors.cuda(), mask_primitives.cuda()
					batch_cdf_sample_masks, batch_cdf_predicates_masks, batch_cdf_joins_masks, train_cdf_signs_tensors = batch_cdf_sample_masks.cuda(), \
					                                                                                                     batch_cdf_predicates_masks.cuda(), \
					                                                                                                     batch_cdf_joins_masks.cuda(), train_cdf_signs_tensors.cuda()

				train_cdf_outputs = model(
					batch_cdf_sample_tensors.view(curr_batch_size * train_max_num_cdfs, max_num_joins + 1, -1),
					batch_cdf_predicates_tensors.view(curr_batch_size * train_max_num_cdfs, max_num_predicates, -1),
					batch_cdf_joins_tensors.view(curr_batch_size * train_max_num_cdfs, max_num_joins, -1),
					batch_cdf_sample_masks.view(curr_batch_size * train_max_num_cdfs, max_num_joins + 1, -1),
					batch_cdf_predicates_masks.view(curr_batch_size * train_max_num_cdfs, max_num_predicates, -1),
					batch_cdf_joins_masks.view(curr_batch_size * train_max_num_cdfs, max_num_joins, -1))
				train_cdf_outputs = train_cdf_outputs.view(curr_batch_size, train_max_num_cdfs)
				train_cdf_outputs = unnormalize_torch(torch.squeeze(train_cdf_outputs), min_val, max_val)
				train_cdf_outputs = train_cdf_outputs - (1./ (np.exp(max_val)-1))
				train_cdf_preds = torch.sum(train_cdf_outputs * torch.squeeze(train_cdf_signs_tensors), dim=-1)

				unnormalized_targets = unnormalize_torch(torch.squeeze(targets.float()), min_val, max_val)
				unnormalized_targets = unnormalized_targets - (1. / (np.exp(max_val)-1))
				cdf_loss = torch.sqrt(torch.mean(torch.square((train_cdf_preds - unnormalized_targets))))

				if epoch > burn_in:
					updated_event.wait()

					with lock:
						aug_sample_tensors, aug_predicate_tensors, aug_join_tensors, aug_sample_masks, aug_predicate_masks, aug_join_masks, \
						aug_cdf_sample_tensors, aug_cdf_predicate_tensors, aug_cdf_join_tensors, aug_cdf_sample_masks, \
						aug_cdf_predicate_masks, aug_cdf_join_masks, aug_cdf_signs_tensors = shared_batch

					if cuda:
						aug_sample_tensors, aug_predicate_tensors, aug_join_tensors = aug_sample_tensors.cuda(), \
						                                                              aug_predicate_tensors.cuda(), \
						                                                              aug_join_tensors.cuda()
						aug_sample_masks, aug_predicate_masks, aug_join_masks = aug_sample_masks.cuda(), \
						                                                        aug_predicate_masks.cuda(), \
						                                                        aug_join_masks.cuda()

						aug_cdf_sample_tensors, aug_cdf_predicate_tensors, aug_cdf_join_tensors = aug_cdf_sample_tensors.cuda(), \
						                                                                          aug_cdf_predicate_tensors.cuda(), \
						                                                                          aug_cdf_join_tensors.cuda()
						aug_cdf_sample_masks, aug_cdf_predicate_masks, aug_cdf_join_masks, aug_cdf_signs_tensors = aug_cdf_sample_masks.cuda(), \
						                                                                                           aug_cdf_predicate_masks.cuda(), \
						                                                                                           aug_cdf_join_masks.cuda(), aug_cdf_signs_tensors.cuda()

					aug_outputs = model(aug_sample_tensors, aug_predicate_tensors, aug_join_tensors,
					                    aug_sample_masks, aug_predicate_masks, aug_join_masks)

					aug_max_num_cdfs = aug_cdf_sample_tensors.shape[1]

					aug_cdf_outputs = model(
						aug_cdf_sample_tensors.view(aug_bs * aug_max_num_cdfs, max_num_joins + 1, -1),
						aug_cdf_predicate_tensors.view(aug_bs * aug_max_num_cdfs, max_num_predicates, -1),
						aug_cdf_join_tensors.view(aug_bs * aug_max_num_cdfs, max_num_joins, -1),
						aug_cdf_sample_masks.view(aug_bs * aug_max_num_cdfs, max_num_joins + 1, -1),
						aug_cdf_predicate_masks.view(aug_bs * aug_max_num_cdfs, max_num_predicates, -1),
						aug_cdf_join_masks.view(aug_bs * aug_max_num_cdfs, max_num_joins, -1))

					aug_cdf_outputs = aug_cdf_outputs.view(aug_bs, aug_max_num_cdfs)
					aug_cdf_outputs = unnormalize_torch(torch.squeeze(aug_cdf_outputs), min_val, max_val)
					aug_cdf_outputs = aug_cdf_outputs - (1. / (np.exp(max_val)-1))
					aug_cdf_preds = torch.sum(aug_cdf_outputs * torch.squeeze(aug_cdf_signs_tensors), dim=-1)

					unnormalized_aug_outputs = unnormalize_torch(torch.squeeze(aug_outputs), min_val, max_val)
					unnormalized_aug_outputs = unnormalized_aug_outputs - (1./ (np.exp(max_val)-1))

					### force negative cdf_preds to be positive..
					adj_unnormalized_aug_outputs = torch.where(aug_cdf_preds < 0, unnormalized_aug_outputs.detach(), unnormalized_aug_outputs)

					q_error_weight_mask = get_top_masks(aug_cdf_preds.cpu().detach(), unnormalized_aug_outputs.cpu().detach(), 1. / (np.exp(max_val)-1), curr_batch_size, cuda)
					consistency_loss = torch.sqrt(torch.sum(torch.square(q_error_weight_mask*(aug_cdf_preds - adj_unnormalized_aug_outputs))) / curr_batch_size)

					#### finish generating consistency loss
					if cdf_loss > 0:
						loss = loss + weight * cdf_loss + weight * consistency_loss
					else:
						loss = loss + weight * consistency_loss
				else:
					if cdf_loss > 0:
						loss = loss + weight * cdf_loss
					else:
						loss = loss

				loss_total += loss.item()
				loss.backward()
				optimizer.step()

			model.eval()
			print("\nEpoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

			preds_candi, candi_label = predict_and_get_labels(model, candi_data_loader, cuda)
			candi_label = unnormalize_labels(candi_label, min_val, max_val, is_cuda=False)

			# Unnormalize
			preds_card_unnorm = unnormalize_labels(preds_candi, min_val, max_val, is_cuda=False)

			# Print metrics
			print("\nPerformance on " + 'In-Distribution Workload' + ":")
			print_qerror(preds_card_unnorm, candi_label, max_val)

			print("Number of ood samples: {}".format(len(labels_test3)))
			preds_test3, _ = predict(model, test_data_loader3, cuda)

			# Unnormalize
			preds_test_unnorm3 = unnormalize_labels(preds_test3, min_val, max_val, is_cuda=False)

			# Print metrics
			print("\nPerformance on " + 'OOD Workload' + ":")
			print_qerror(preds_test_unnorm3, label3, max_val)

		# To stop the child process
		stop_event.set()
		sampler.join()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--shift", help="center or granularity?", default='granularity')
	parser.add_argument("--queries", help="number of training queries (default: 30000)", type=int, default=60000)
	parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=80)
	parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=100)
	parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
	parser.add_argument("--weight", help="weight of CDF training", type=float, default=1)
	args = parser.parse_args()
	is_cuda = torch.cuda.is_available()
	table_list = ['date_dim', 'customer_demographics', 'store_sales', 'household_demographics', 'store']
	table_to_df = get_bitmap.load_tables(table_list, data_dir='./samples/dsb')

	train_and_predict(args.queries, args.epochs, args.batch, args.hid, is_cuda, table_to_df, args.weight, shift=args.shift)


if __name__ == "__main__":
	main()
