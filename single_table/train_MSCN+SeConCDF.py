import argparse
import torch
from torch.utils.data import DataLoader
from multiprocessing import Process, Lock, Manager, Event

from mscn.utils import *
from mscn.data import get_train_datasets, make_dataset, load_ood_data, \
	make_aug_dataset_cdf_no_padding
from mscn.model import SetConv

from lw_nn.utils import *
from SeConCDF.mscn_utils import *

rs = np.random.RandomState(42)

def train_and_predict(num_queries, num_epochs, table_card, batch_size, hid_units, cuda, weight=1, shift='granularity'):
	# Load training and validation data

	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_dataset, candi_data, \
	ori_predicates_train, ori_predicates_test, num_predicates_train, num_predicates_test, numerical_cols, candi_query_typeids, candi_predicates = get_train_datasets(
		num_queries, trans_op=True, workload_type='in', shift=shift)

	column2vec, op2vec = dicts

	# Train model
	predicate_feats = len(column2vec) + len(op2vec) + 1

	model = SetConv(predicate_feats, hid_units)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	if cuda:
		model.cuda()

	train_data_loader = DataLoader(train_data, batch_size=batch_size)
	candi_data_loader = DataLoader(candi_data, batch_size=batch_size)

	### load ood queries
	workload_name3 = 'census-mscn-50000'
	file_name = "queries/" + workload_name3
	predicates3, label3 = load_ood_data(file_name, num_queries, column_min_max_vals=column_min_max_vals,
	                                    trans_op=True, workload_type='ood', shift=shift)

	# Get feature encoding and proper normalization
	predicates_test3 = encode_data(predicates3, column_min_max_vals, column2vec, op2vec)
	labels_test3, _, _ = normalize_labels(label3, min_val, max_val, add_one=True)

	test_max_num_predicates3 = max([len(p) for p in predicates_test3])

	# Get test set predictions
	test_data3 = make_dataset(predicates_test3, labels_test3, test_max_num_predicates3)
	test_data_loader3 = DataLoader(test_data3, batch_size=batch_size)

	train_cdfs_cache = []
	num_qs = 0

	for batch_idx, data_batch in enumerate(train_data_loader):
		predicates, targets, predicate_masks, train_ids = data_batch
		train_ids = train_ids.numpy()
		batch_predicates = [ori_predicates_train[x] for x in train_ids]
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

		train_cdf_predicates = encode_cdf_data(train_cdfs, column_min_max_vals, column2vec, op2vec)

		train_cdf_predicate_list, train_cdf_predicate_masks, train_cdf_signs_tensors, max_num_cdfs = make_aug_dataset_cdf_no_padding(
			train_cdf_predicates, train_signs, max_num_predicates)

		train_cdfs_cache.append(
			[train_cdf_predicate_list, train_cdf_predicate_masks, train_cdf_signs_tensors, max_num_cdfs,
			 mask_primitives])

		num_qs += len(batch_predicates)

	with Manager() as manager:
		shared_batch = manager.list()  # Managed dictionary to hold the current data batch
		lock = manager.Lock()
		stop_event = Event()
		updated_event = Event()
		print('max_num_predicates')
		print(max_num_predicates)
		sampler = Process(target=sample_batch,
		                  args=(batch_size, column_min_max_vals, max_num_predicates,
		                        shared_batch, column2vec, op2vec, cuda, lock, stop_event, updated_event,))
		sampler.start()
		for epoch in range(num_epochs):
			loss_total = 0.
			model.train()
			for batch_idx, data_batch in enumerate(train_data_loader):

				predicates, targets, predicate_masks, train_ids = data_batch

				batch_predicates = [ori_predicates_train[x] for x in train_ids]

				if cuda:
					predicates, targets = predicates.cuda(), targets.cuda()
					predicate_masks = predicate_masks.cuda()
				predicates, targets = Variable(predicates), Variable(targets)
				predicate_masks = Variable(predicate_masks)

				optimizer.zero_grad()
				outputs = model(predicates, predicate_masks)

				loss = torch.sqrt(torch.mean(torch.square(torch.squeeze(outputs) - torch.squeeze(targets.float()))))

				curr_batch_size = len(train_ids)
				#### generate cdf loss
				train_cdfs = []
				train_signs = []
				for filters in batch_predicates:
					q_cdfs, q_signs = query2cdfs(filters, column_min_max_vals)
					train_cdfs.append(q_cdfs)
					train_signs.append(q_signs)

				train_cdf_predicate_tensors, train_cdf_predicate_masks, train_cdf_signs_tensors, max_num_cdfs, mask_primitives = \
					train_cdfs_cache[batch_idx]

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

				### finish getting the inputs

				train_max_num_cdfs = batch_cdf_predicates_tensors.shape[1]

				if cuda:
					batch_cdf_predicates_tensors = batch_cdf_predicates_tensors.cuda()
					batch_cdf_predicates_masks, train_cdf_signs_tensors = batch_cdf_predicates_masks.cuda(), train_cdf_signs_tensors.cuda()

				train_cdf_outputs = model(
					batch_cdf_predicates_tensors.view(curr_batch_size * train_max_num_cdfs, max_num_predicates, -1),
					batch_cdf_predicates_masks.view(curr_batch_size * train_max_num_cdfs, max_num_predicates, -1))
				train_cdf_outputs = train_cdf_outputs.view(curr_batch_size, train_max_num_cdfs)
				train_cdf_outputs = unnormalize_torch(torch.squeeze(train_cdf_outputs), min_val, max_val, is_selectivity=True)
				train_cdf_preds = torch.sum(train_cdf_outputs * torch.squeeze(train_cdf_signs_tensors), dim=-1)
				unnormalized_targets = unnormalize_torch(torch.squeeze(targets.float()), min_val, max_val, is_selectivity=True)
				cdf_loss = torch.sqrt(torch.mean(torch.square((train_cdf_preds - unnormalized_targets))))

				#################
				updated_event.wait()

				with lock:
					aug_predicate_tensors, aug_predicate_masks, aug_cdf_predicate_tensors, aug_cdf_predicate_masks, aug_cdf_signs_tensors = shared_batch

				if cuda:
					aug_predicate_tensors = aug_predicate_tensors.cuda()
					aug_predicate_masks = aug_predicate_masks.cuda()

					aug_cdf_predicate_tensors = aug_cdf_predicate_tensors.cuda()
					aug_cdf_predicate_masks, aug_cdf_signs_tensors = aug_cdf_predicate_masks.cuda(), aug_cdf_signs_tensors.cuda()

				aug_outputs = model(aug_predicate_tensors, aug_predicate_masks)

				aug_max_num_cdfs = aug_cdf_predicate_tensors.shape[1]

				aug_cdf_outputs = model(
					aug_cdf_predicate_tensors.view(batch_size * aug_max_num_cdfs, max_num_predicates, -1),
					aug_cdf_predicate_masks.view(batch_size * aug_max_num_cdfs, max_num_predicates, -1))

				aug_cdf_outputs = aug_cdf_outputs.view(batch_size, aug_max_num_cdfs)
				aug_cdf_outputs = unnormalize_torch(torch.squeeze(aug_cdf_outputs), min_val, max_val, is_selectivity=True)
				aug_cdf_preds = torch.sum(aug_cdf_outputs * torch.squeeze(aug_cdf_signs_tensors), dim=-1)

				unnormalized_aug_outputs = unnormalize_torch(torch.squeeze(aug_outputs), min_val, max_val, is_selectivity=True)

				### force negative cdf_preds to be positive.
				adj_unnormalized_aug_outputs = torch.where(aug_cdf_preds < 0, unnormalized_aug_outputs.detach(),
				                                           unnormalized_aug_outputs)

				consistency_loss = torch.sqrt(torch.mean(
					torch.square(aug_cdf_preds - adj_unnormalized_aug_outputs)))

				loss = loss + weight * cdf_loss + weight * consistency_loss

				loss_total += loss.item()
				loss.backward()
				optimizer.step()

			print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

			preds_candi, candi_label = predict_and_get_labels(model, candi_data_loader, cuda)
			candi_label = unnormalize_labels(candi_label, min_val, max_val, is_cuda=cuda)

			# Unnormalize
			preds_card_unnorm = unnormalize_labels(preds_candi, min_val, max_val, is_cuda=cuda)

			# Print metrics
			_ = print_qerror(preds_card_unnorm, candi_label, table_card, 'In-Distribution')
			print('')

			preds_test3, _ = predict(model, test_data_loader3, cuda)

			# Unnormalize
			preds_test_unnorm3 = unnormalize_labels(preds_test3, min_val, max_val, is_cuda=cuda)

			# Print metrics
			_ = print_qerror(preds_test_unnorm3, label3, table_card, 'Out-of-Distribution')
			print('')

		sampler.terminate()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--shift", help="shift type", default='granularity')
	parser.add_argument("--queries", help="number of training queries (default: 50000)", type=int, default=50000)
	parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=70)
	parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=200)
	parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
	parser.add_argument("--weight", help="weight for CDF pred loss and consistency loss", type=float, default=0.5)
	args = parser.parse_args()
	is_cuda = torch.cuda.is_available()

	### input the table card here
	table_card = 48842
	train_and_predict(args.queries, args.epochs, table_card, args.batch, args.hid, is_cuda, args.weight, shift=args.shift)


if __name__ == "__main__":
	main()
