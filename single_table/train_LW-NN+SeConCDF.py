import argparse
from lw_nn.utils import *
from SeConCDF.lw_utils import *
import torch
import random
from model.model import MLP

def train_and_test(table_card, shift_type='granularity', weight=1):
	q_ranges, new_queries, q_sels, in_ids, ood_ids = get_train_test_queries(queries, cards, shift_col_id=0, shift=shift_type)

	feature_dim = 256
	num_dim = len(cols)

	model = MLP(2 * num_dim, feature_dim)

	optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
	torch.autograd.set_detect_anomaly(True)

	epoch = 60
	bs = 200
	training_size = int(0.9*len(in_ids))

	training_intervals_all = [q_ranges[qid] for qid in in_ids]
	training_sels_all = [q_sels[qid] for qid in in_ids]
	training_qs_all = [new_queries[qid] for qid in in_ids]

	test_intervals = training_intervals_all[training_size:]
	test_sels = training_sels_all[training_size:]

	training_queries = training_qs_all[:training_size]
	training_intervals = training_intervals_all[:training_size]
	training_sels = training_sels_all[:training_size]

	ood_test_intervals = [q_ranges[qid] for qid in ood_ids]
	ood_test_sels = [q_sels[qid] for qid in ood_ids]

	print("training size: {}".format(len(training_intervals)))
	print("test size: {}".format(len(test_intervals)))
	print("ood test size: {}".format(len(ood_test_sels)))
	num_batches = int(len(training_intervals) / bs) + 1

	for epoch_id in range(epoch):
		model.train()
		q_idxs = list(range(len(training_intervals)))
		random.shuffle(q_idxs)

		accu_loss_total = 0.

		for batch_id in range(num_batches):
			batch_idxs = q_idxs[batch_id * bs: batch_id * bs + bs]
			batch_queries = extract_sublist(training_queries, batch_idxs)
			batch_intervals = extract_sublist(training_intervals, batch_idxs)
			batch_sels = extract_sublist(training_sels, batch_idxs)
			if len(batch_idxs) == 0:
				break

			batch_queries_tensor = np.stack(batch_intervals, axis=0)
			batch_queries_tensor = torch.FloatTensor(batch_queries_tensor)

			batch_sels_tensor = np.stack(batch_sels, axis=0)
			batch_sels_tensor = torch.FloatTensor(batch_sels_tensor)

			optimizer.zero_grad()

			outputs = model(batch_queries_tensor)
			sel_predicts = torch.squeeze(outputs)
			########### for cdfs!

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

			########## get consistency pairs!
			consis_sel_preds, consis_cdf_preds = get_consistency_pairs(bs, num_dim, col_to_valid_list, model)

			###################

			criterion = torch.nn.MSELoss()
			original_loss = torch.sqrt(criterion(torch.log(sel_predicts), torch.log(batch_sels_tensor)))
			cdf_loss = torch.sqrt(criterion(cdf_sel_predicts, batch_sels_tensor))

			adj_consis_sel_preds = consis_sel_preds
			consistency_loss = torch.sqrt(criterion(adj_consis_sel_preds, consis_cdf_preds))

			total_loss = original_loss + weight * cdf_loss + weight * consistency_loss

			accu_loss_total += total_loss.item()
			total_loss.backward()
			optimizer.step()

		print("epoch: {}; loss:{}".format(epoch_id, accu_loss_total / num_batches))

		model.eval()
		test_training_queries = np.stack(test_intervals, axis=0)
		test_training_queries = torch.FloatTensor(test_training_queries)

		test_training_sels = np.stack(test_sels, axis=0)
		test_training_sels = torch.FloatTensor(test_training_sels)

		outputs = model(test_training_queries)
		sel_predicts = torch.squeeze(outputs)

		get_qerror(sel_predicts, test_training_sels, table_card, 'In-Distribution')

		test_training_queries = np.stack(ood_test_intervals, axis=0)
		test_training_queries = torch.FloatTensor(test_training_queries)

		test_training_sels = np.stack(ood_test_sels, axis=0)
		test_training_sels = torch.FloatTensor(test_training_sels)

		outputs = model(test_training_queries)
		sel_predicts = torch.squeeze(outputs)

		get_qerror(sel_predicts, test_training_sels, table_card, 'Out-of-Distribution')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--shift", help="shift type", default='granularity')
	parser.add_argument("--weight", help="weight for CDF pred loss and consistency loss", type=float, default=1)
	args = parser.parse_args()

	### input the table card here
	table_card = 48842
	train_and_test(table_card, args.shift, args.weight)

if __name__ == "__main__":
	main()
