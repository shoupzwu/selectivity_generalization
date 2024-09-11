import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define model architecture

class SetConv(nn.Module):
	def __init__(self, predicate_feats, hid_units):
		super(SetConv, self).__init__()
		self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
		self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
		self.out_mlp1 = nn.Linear(hid_units, hid_units)
		self.out_mlp2 = nn.Linear(hid_units, 1)

	def forward(self,  predicates,  predicate_mask):
		# samples has shape [batch_size x num_joins+1 x sample_feats]
		# predicates has shape [batch_size x num_predicates x predicate_feats]
		# joins has shape [batch_size x num_joins x join_feats]

		hid_predicate = F.relu(self.predicate_mlp1(predicates))
		hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
		hid_predicate = hid_predicate * predicate_mask
		hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
		predicate_norm = predicate_mask.sum(1, keepdim=False)
		hid_predicate = hid_predicate / predicate_norm

		hid = F.relu(self.out_mlp1(hid_predicate))
		out = torch.sigmoid(self.out_mlp2(hid))
		return out
