import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class MLP(nn.Module):
	def __init__(self, predicate_feats, hid_units):
		super().__init__()
		self.predicate_mlp1 = nn.Linear(predicate_feats,  hid_units)
		self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
		self.predicate_mlp3= nn.Linear(hid_units, hid_units)
		self.out_mlp1 = nn.Linear(hid_units, 1)

		self.feature_num = predicate_feats
		self.rs = np.random.RandomState(42)
		random.seed(42)


	def forward(self, predicates):
		hid_predicate = torch.relu(self.predicate_mlp1(predicates))
		hid_predicate = torch.relu(self.predicate_mlp2(hid_predicate))
		hid_predicate = torch.relu(self.predicate_mlp3(hid_predicate))
		out = torch.sigmoid(self.out_mlp1(hid_predicate))

		return out




























