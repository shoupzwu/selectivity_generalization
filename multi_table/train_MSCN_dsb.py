import argparse
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, make_dataset, load_ood_data
from mscn.model import SetConv

rs = np.random.RandomState(42)

def unnormalize_torch(vals, min_val, max_val):
	vals = (vals * (max_val - min_val)) + min_val
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

def train_and_predict(num_queries, num_epochs, batch_size, hid_units, cuda, shift='center'):
	# Load training and validation data
	num_materialized_samples = 1000

	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_validation, max_num_joins, max_num_predicates, train_data, test_data, candi_data, \
	ori_predicates_train, ori_samples_train, ori_tables_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, num_joins_test, num_predicates_test, table_sets_test, \
	numerical_cols, candi_query_typeids, candi_joins, candi_predicates, candi_tables, candi_samples = get_train_datasets(
		num_queries, num_materialized_samples, dataset='dsb', trans_op=True, workload_type='in', shift=shift)

	table2vec, column2vec, op2vec, join2vec = dicts

	# Train model
	sample_feats = len(table2vec) + num_materialized_samples
	predicate_feats = len(column2vec) + len(op2vec) + 1
	join_feats = len(join2vec)

	model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	if cuda:
		model.cuda()

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
	labels_test3, _, _ = normalize_labels(label3, min_val, max_val)


	test_max_num_predicates3 = max([len(p) for p in predicates_test3])
	test_max_num_joins3 = max([len(j) for j in joins_test3])

	# Get test set predictions
	test_data3 = make_dataset(samples_test3, predicates_test3, joins_test3, labels_test3, test_max_num_joins3,
	                          test_max_num_predicates3)
	test_data_loader3 = DataLoader(test_data3, batch_size=batch_size)

	model.train()
	for epoch in range(num_epochs):
		loss_total = 0.

		for batch_idx, data_batch in enumerate(train_data_loader):

			samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch

			if cuda:
				samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
				sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
			samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
				targets)
			sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
				join_masks)

			optimizer.zero_grad()
			outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)

			# loss = qerror_loss(outputs, targets.float(), min_val, max_val)
			loss = torch.mean(torch.square(torch.squeeze(outputs) - torch.squeeze(targets.float())))

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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--shift", help="center or granularity?", default='granularity')
	parser.add_argument("--queries", help="number of training queries (default: 30000)", type=int, default=60000)
	parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=80)
	parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=400)
	parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
	args = parser.parse_args()
	is_cuda = torch.cuda.is_available()

	train_and_predict(args.queries, args.epochs, args.batch, args.hid, is_cuda, shift=args.shift)

if __name__ == "__main__":
	main()
