import argparse
import random
import torch
from mscn.utils import *

from torch.utils.data import DataLoader
from mscn.data import get_train_datasets, make_dataset, load_ood_data
from mscn.model import SetConv

def train_and_predict(num_queries, num_epochs, table_card, batch_size, hid_units, cuda, shift='granularity'):
    # Load training and validation data
    num_materialized_samples = 1000
    weight = 1

    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_dataset, candi_data, \
    ori_predicates_train, ori_predicates_test, num_predicates_train, num_predicates_test, numerical_cols, candi_query_typeids, candi_predicates = get_train_datasets(
        num_queries, trans_op=True, workload_type='in', shift=shift)

    column2vec, op2vec  = dicts

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
    predicates_test3 = encode_data(predicates3,column_min_max_vals, column2vec, op2vec)
    labels_test3, _, _ = normalize_labels(label3, min_val, max_val)

    test_max_num_predicates3 = max([len(p) for p in predicates_test3])

    # Get test set predictions
    test_data3 = make_dataset(predicates_test3, labels_test3, test_max_num_predicates3)
    test_data_loader3 = DataLoader(test_data3, batch_size=batch_size)

    model.train()

    print(len(train_data_loader))
    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):

            predicates, targets, predicate_masks, train_ids = data_batch

            if cuda:
                predicates, targets =  predicates.cuda(),targets.cuda()
                predicate_masks =  predicate_masks.cuda()
            predicates, targets =  Variable(predicates), Variable(targets)
            predicate_masks= Variable(predicate_masks)

            optimizer.zero_grad()
            outputs = model(predicates,predicate_masks)

            # loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss = torch.mean(torch.square(torch.squeeze(outputs) - torch.squeeze(targets.float())))

            loss_total += loss.item()
            loss.backward()
            optimizer.step()


        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

        preds_candi, candi_label = predict_and_get_labels(model, candi_data_loader, cuda)
        candi_label = unnormalize_labels(candi_label, min_val, max_val, is_cuda=False)

        # Unnormalize
        preds_card_unnorm = unnormalize_labels(preds_candi, min_val, max_val, is_cuda=False)

        # Print metrics
        _ = print_qerror(preds_card_unnorm, candi_label, table_card)
        print('')

        preds_test3, _ = predict(model, test_data_loader3, cuda)

        # Unnormalize
        preds_test_unnorm3 = unnormalize_labels(preds_test3, min_val, max_val, is_cuda=False)

        # Print metrics
        _ = print_qerror(preds_test_unnorm3, label3, table_card, 'Out-of-Distribution')

        print('')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", help="shift type", default='granularity')
    parser.add_argument("--queries", help="number of training queries (default: 30000)", type=int, default=50000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=60)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=200)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    args = parser.parse_args()
    is_cuda =  torch.cuda.is_available()

    ### input the table card here
    table_card = 48842

    train_and_predict(args.queries, args.epochs, table_card, args.batch, args.hid, is_cuda, shift=args.shift)

if __name__ == "__main__":
    main()
