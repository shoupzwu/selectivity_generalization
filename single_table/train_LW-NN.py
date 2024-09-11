import argparse
from lw_nn.utils import *
import torch
import random
from model.model import MLP

def train_and_test(table_card, shift_type='granularity'):
	q_ranges, new_queries, q_sels, in_ids, ood_ids = get_train_test_queries(queries, cards, shift_col_id=0, shift=shift_type)

	feature_dim = 256
	num_dim = len(cols)

	model = MLP(2*num_dim, feature_dim)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	torch.autograd.set_detect_anomaly(True)

	epoch = 60
	bs = 200
	training_size = int(0.9*len(in_ids))

	training_intervals_all = [q_ranges[qid] for qid in in_ids]
	training_sels_all = [q_sels[qid] for qid in in_ids]

	test_intervals = training_intervals_all[training_size:]
	test_sels = training_sels_all[training_size:]

	training_intervals = training_intervals_all[:training_size]
	training_sels  = training_sels_all[:training_size]

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
			batch_intervals = extract_sublist(training_intervals, batch_idxs)
			batch_sels = extract_sublist(training_sels, batch_idxs)

			if len(batch_idxs) == 0:
				break

			batch_queries_tensor = np.stack(batch_intervals, axis=0)
			batch_queries_tensor = torch.FloatTensor(batch_queries_tensor)

			batch_sels_tensor = np.stack(batch_sels, axis=0)
			batch_sels_tensor = torch.FloatTensor(batch_sels_tensor)

			optimizer.zero_grad()

			outputs = model(batch_queries_tensor)  # [bs, max_num_cdfs, 1]
			sel_predicts = torch.squeeze(outputs)

			criterion = torch.nn.MSELoss()
			original_loss = torch.sqrt(criterion(torch.log(sel_predicts), torch.log(batch_sels_tensor)))

			total_loss = original_loss

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
	args = parser.parse_args()

	### input the table card here
	table_card = 48842
	train_and_test(table_card, args.shift)

if __name__ == "__main__":
	main()
