import os
import copy
import numpy as np
import torch
from model import MLP
import random
import json
from utilites import *
from SeConCDF_helpers import *

random.seed(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extract_sublist(original_list, indices):
	return [original_list[i] for i in indices]


num_dim = 10
num_samples = 50000

epoch = 1000
bs = 1000
candi_size = 50000
test_size = 1000
feature_dim = 256
q_file = 'queries/queries_50000.csv'
model = MLP(2*num_dim, feature_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

training_queries, training_intervals, training_sels, test_queries, test_intervals, \
test_sels, ood_test_queries, ood_test_intervals, ood_test_sels = get_queries(q_file)


print("training size: {}".format(len(training_intervals)))
print("test size: {}".format(len(test_intervals)))
print("ood test size: {}".format(len(ood_test_sels)))

num_batches = int(len(training_queries) / bs) + 1

for epoch_id in range(epoch):
	model.train()
	q_idxs = list(range(len(training_queries)))
	random.shuffle(q_idxs)

	accu_loss_total = 0.
	cdf_valid_loss = 0.
	mono_loss = 0.

	for batch_id in range(num_batches):
		batch_idxs = q_idxs[batch_id * bs: batch_id * bs + bs]
		batch_queries = extract_sublist(training_queries, batch_idxs)
		batch_intervals = extract_sublist(training_intervals, batch_idxs)
		batch_sels = extract_sublist(training_sels, batch_idxs)

		batch_queries_tensor = np.stack(batch_intervals, axis=0)
		batch_queries_tensor = torch.FloatTensor(batch_queries_tensor)

		batch_sels_tensor = np.stack(batch_sels, axis=0)
		batch_sels_tensor = torch.FloatTensor(batch_sels_tensor)

		optimizer.zero_grad()

		outputs = model(batch_queries_tensor)  # [bs, max_num_cdfs, 1]
		sel_predicts = torch.squeeze(outputs)

		########### for cdf pred loss
		batch_cdfs, batch_signs, max_num_cdfs = prepare_batch(batch_queries, num_dim)

		batch_cdfs_tensor = np.stack(batch_cdfs, axis=0)
		batch_cdfs_tensor = torch.FloatTensor(batch_cdfs_tensor)

		batch_signs_tensor = np.stack(batch_signs, axis=0)
		batch_signs_tensor = torch.FloatTensor(batch_signs_tensor)

		batch_cdfs_tensor = batch_cdfs_tensor.view(-1, 2 * num_dim)
		cdf_outputs = model(batch_cdfs_tensor)  # [bs*max_num_cdfs, num_dims]

		cdf_sel_predicts = torch.squeeze(cdf_outputs)
		cdf_sel_predicts = cdf_sel_predicts.view(-1, max_num_cdfs)  # [bs, max_num_cdfs]
		cdf_sel_predicts = torch.sum(cdf_sel_predicts * batch_signs_tensor, dim=-1)
		#####################

		########## get consistency pairs
		consis_sel_preds, consis_cdf_preds = get_consistency_pairs(bs, num_dim, model)

		criterion = torch.nn.MSELoss()
		original_loss = torch.sqrt(criterion(torch.log(sel_predicts), torch.log(batch_sels_tensor)))
		cdf_loss = torch.sqrt(criterion(cdf_sel_predicts, batch_sels_tensor))
		consistency_loss = torch.sqrt(criterion(consis_sel_preds, consis_cdf_preds))

		total_loss = original_loss + cdf_loss + consistency_loss

		accu_loss_total += total_loss.item()

		total_loss.backward()
		optimizer.step()

	print("epoch: {}; loss: {}".format(epoch_id, accu_loss_total / num_batches))

	model.eval()

	test_training_queries = np.stack(test_intervals, axis=0)
	test_training_queries = torch.FloatTensor(test_training_queries)

	test_training_sels = np.stack(test_sels, axis=0)
	test_training_sels = torch.FloatTensor(test_training_sels)

	outputs = model(test_training_queries)
	sel_predicts = torch.squeeze(outputs)

	get_qerror(sel_predicts, test_training_sels)

	### test OOD queries
	test_training_queries = np.stack(ood_test_intervals, axis=0)
	test_training_queries = torch.FloatTensor(test_training_queries)

	test_training_sels = np.stack(ood_test_sels, axis=0)
	test_training_sels = torch.FloatTensor(test_training_sels)

	outputs = model(test_training_queries)
	sel_predicts = torch.squeeze(outputs)

	get_qerror(sel_predicts, test_training_sels, 'Out-of-Distribution')