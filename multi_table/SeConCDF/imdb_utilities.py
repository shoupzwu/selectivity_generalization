import random
import numpy as np
import copy

from mscn.util import *
from mscn.data import get_train_datasets, make_dataset, load_ood_data, \
	 make_aug_dataset, make_aug_dataset_cdf
import mscn.get_bitmap_imdb as get_bitmap
import csv

NUM_MATERIALIZED_SAMPLES = 1000

seed = 42
random.seed(seed)

file_name_queries = "workloads/imdb-train"
file_name_column_min_max_vals = "./data/column_min_max_vals_imdb.csv"

with open(file_name_column_min_max_vals, 'rU') as f:
	data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
	column_min_max_vals = {}
	for i, row in enumerate(data_raw):
		if i == 0:
			continue
		column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

def sample_queries(num_queries, column_min_max_vals, max_num_preds=5, min_max_numpreds_per_joingraph=None, fixed_col=None):

	candi_joins = {'cast_info ci':'t.id=ci.movie_id',
	               'movie_companies mc':'t.id=mc.movie_id',
	               'movie_info mi':'t.id=mi.movie_id',
	               'movie_info_idx mi_idx':'t.id=mi_idx.movie_id',
	                'movie_keyword mk':'t.id=mk.movie_id'}

	dim_tables = ['cast_info ci', 'movie_companies mc', 'movie_info mi', 'movie_info_idx mi_idx', 'movie_keyword mk']
	fact_table = 'title t'

	col_dict = {'title t':['t.kind_id', 't.production_year'],
	               'movie_companies mc': ['mc.company_id', 'mc.company_type_id'],
	               'cast_info ci': ['ci.role_id'],
	               'movie_info mi': ['mi.info_type_id'],
	              'movie_info_idx mi_idx': ['mi_idx.info_type_id'],
	                'movie_keyword mk': ['mk.keyword_id']}

	filters = []
	tables = []
	joins = []
	num_generated = 0

	if min_max_numpreds_per_joingraph:
		num2tables = {}
		for t_set in min_max_numpreds_per_joingraph:
			if len(t_set) not in num2tables:
				num2tables[len(t_set)] = [t_set]
			else:
				num2tables[len(t_set)].append(t_set)

	while True:
		if num_generated >= num_queries:
			break

		num_tables = random.randint(1, 6)

		if num_tables == 1:
			if min_max_numpreds_per_joingraph:
				chosen_table_set = random.choice(num2tables[1])
				chosen_table = list(chosen_table_set)[0]
				num_filters = random.randint(1,  min_max_numpreds_per_joingraph[chosen_table_set][1])
			else:
				chosen_table = random.choice(dim_tables + [fact_table])
				num_filters =  random.randint(1, len(col_dict[chosen_table]))

			if not fixed_col:
				sampled_cols = random.sample(col_dict[chosen_table], num_filters)
			else:
				if chosen_table == fact_table:
					sampled_cols = random.sample(col_dict[chosen_table], num_filters-1) + [fixed_col]
				else:
					sampled_cols = random.sample(col_dict[chosen_table], num_filters)

			q_filters = []

			for col in sampled_cols:
				op = random.choice(['<=', '=', '>='])
				val =  random.randint(column_min_max_vals[col][0], column_min_max_vals[col][1])
				q_filters.append([col, op, val])

			tables.append([chosen_table])
			joins.append([''])
			filters.append(q_filters)

		else:
			if min_max_numpreds_per_joingraph:
				sampled_tables_set = random.choice(num2tables[num_tables])
				sampled_tables = list(sampled_tables_set)

				total_num_candi_filters = 2
				q_joins = []
				total_candi_cols = copy.deepcopy(col_dict[fact_table])

				for t in sampled_tables:
					if t != fact_table:
						total_num_candi_filters += len(col_dict[t])
						total_candi_cols.extend(col_dict[t])
						q_joins.append(candi_joins[t])

				num_filters = random.randint(1, min(total_num_candi_filters, min_max_numpreds_per_joingraph[sampled_tables_set][1]))

				if not fixed_col:
					sampled_cols = random.sample(total_candi_cols, num_filters)
				else:
					sampled_cols = random.sample(total_candi_cols, num_filters-1) + [fixed_col]

				q_filters = []
				for col in sampled_cols:
					op = random.choice(['<=', '=', '>='])
					val = random.randint(column_min_max_vals[col][0], column_min_max_vals[col][1])
					q_filters.append([col, op, val])

				tables.append(sampled_tables)
				joins.append(q_joins)
				filters.append(q_filters)
			else:
				sampled_dim_tables = random.sample(dim_tables, num_tables-1)

				total_num_candi_filters = 2
				q_joins = []
				total_candi_cols = copy.deepcopy(col_dict[fact_table])

				for t in sampled_dim_tables:
					total_num_candi_filters += len(col_dict[t])
					total_candi_cols.extend(col_dict[t])
					q_joins.append(candi_joins[t])

				num_filters = random.randint(1, min(total_num_candi_filters, max_num_preds))

				if not fixed_col:
					sampled_cols = random.sample(total_candi_cols, num_filters)
				else:
					sampled_cols = random.sample(total_candi_cols, num_filters - 1) + [fixed_col]

				q_filters = []
				for col in sampled_cols:
					op = random.choice(['<=', '=', '>='])
					val =  random.randint(column_min_max_vals[col][0], column_min_max_vals[col][1])
					q_filters.append([col, op, val])


				tables.append([fact_table] + sampled_dim_tables)
				joins.append(q_joins)
				filters.append(q_filters)

		num_generated += 1

	return tables, joins, filters

def sample_queries_from_list(tables, joins, filters, sample_size=100):
	candi_ids = list(range(len(tables)))
	random.shuffle(candi_ids)

	candi_ids = candi_ids[:sample_size]

	sampled_tables = [tables[qid] for qid in candi_ids]
	sampled_joins = [joins[qid] for qid in candi_ids]
	sampled_filters = [filters[qid] for qid in candi_ids]

	return sampled_tables, sampled_joins, sampled_filters

def query2cdfs(q_filters, column_min_max_vals):
	cdfs = [[]]
	lower_counts = [0]
	signs = []

	for filter in q_filters:
		col = filter[0]
		op = filter[1]
		val = filter[2]

		col_min = int(column_min_max_vals[col][0])
		col_max = int(column_min_max_vals[col][1])

		if op == '<=':
			if int(val) == col_max:
				continue

			for cdf in cdfs:
				cdf.append([col, '<=', val])
		elif op == '>=':
			if int(val) == col_min:
				continue

			cdfs_copy = copy.deepcopy(cdfs)
			lower_counts_copy = copy.deepcopy(lower_counts)

			for cdf in cdfs:
				cdf.append([col, '<=', int(val) - 1])
			for j in range(len(lower_counts)):
				lower_counts[j] += 1

			cdfs.extend(cdfs_copy)
			lower_counts.extend(lower_counts_copy)

		else:  # the case for =
			if int(val) == col_min:
				for cdf in cdfs:
					cdf.append([col, '<=', int(val)])
			else:
				cdfs_copy = copy.deepcopy(cdfs)
				lower_counts_copy = copy.deepcopy(lower_counts)

				for cdf in cdfs:
					cdf.append([col, '<=', int(val) - 1])
				for j in range(len(lower_counts)):
					lower_counts[j] += 1

				for cdf in cdfs_copy:
					cdf.append([col, '<=', int(val)])

				cdfs.extend(cdfs_copy)
				lower_counts.extend(lower_counts_copy)

	for count in lower_counts:
		signs.append((-1) ** count)

	return cdfs, signs

def load_queries(file_name):
	tables, joins, filters = [], [], []
	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		for qid, row in enumerate(data_raw):
			q_filters = []
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))

			filter_items = row[2].split(',')

			if len(filter_items) < 3:
				q_filters.append('')
			else:
				for pid in range(int(len(filter_items)/3)):
					col = filter_items[pid*3]
					op = filter_items[pid * 3+1]
					val = filter_items[pid * 3+2]

					if op == '<':
						op = '<='
						if isinstance(val, str):
							val = str(int(val) - 1)
						else:
							val = int(val) - 1
					elif op == '>':
						op = '>='
						if isinstance(val, str):
							val = str(int(val) + 1)
						else:
							val = int(val) + 1
					q_filters.append([col, op, str(val)])

			filters.append(q_filters)

	return tables, joins, filters

def sample_batch(aug_bs, column_min_max_vals, max_num_predicates, max_num_joins, table_to_df,
                 shared_batch, table2vec, column2vec, op2vec, join2vec, cuda, lock, stop_event, updated_event, min_max_numpreds_per_joingraph):
	while not stop_event.is_set():
		# Simulate generating a new data batch
		consistent_tables, consistent_joins, consistent_filters = sample_queries(aug_bs, column_min_max_vals,max_num_predicates,
		                                                                            min_max_numpreds_per_joingraph=min_max_numpreds_per_joingraph)

		consistent_cdfs = []
		consistent_signs = []
		for filters in consistent_filters:
			q_cdfs, q_signs = query2cdfs(filters, column_min_max_vals)
			consistent_cdfs.append(q_cdfs)
			consistent_signs.append(q_signs)
		q_bitmap_list = get_bitmap.ori_compute_bitmap(table_to_df, consistent_tables, consistent_filters)
		cdf_bitmap_list = get_bitmap.cdf_compute_bitmap(table_to_df, consistent_tables, consistent_cdfs)
		consistent_cdf_tables = []
		for qid, consistent_q_tables in enumerate(consistent_tables):
			a_cdf_tables = []
			for _ in range(len(cdf_bitmap_list[qid])):
				a_cdf_tables.append(consistent_q_tables)
			consistent_cdf_tables.append(a_cdf_tables)

		q_bitmap_list = encode_samples(consistent_tables, q_bitmap_list, table2vec)
		cdf_bitmap_list = encode_cdf_samples(consistent_cdf_tables, cdf_bitmap_list, table2vec)
		predicates_consis, joins_consis = encode_data(consistent_filters, consistent_joins, column_min_max_vals,
		                                              column2vec, op2vec, join2vec)
		cdf_predicates_consis, cdf_joins_consis = encode_cdf_data(consistent_cdfs, consistent_joins,
		                                                          column_min_max_vals,
		                                                          column2vec, op2vec, join2vec,
		                                                          is_join_same_shape=False)
		aug_sample_tensors, aug_predicate_tensors, aug_join_tensors, aug_sample_masks, aug_predicate_masks, \
		aug_join_masks = make_aug_dataset(q_bitmap_list, predicates_consis, joins_consis, max_num_joins,
		                                  max_num_predicates)
		aug_cdf_sample_tensors, aug_cdf_predicate_tensors, aug_cdf_join_tensors, aug_cdf_sample_masks, aug_cdf_predicate_masks, \
		aug_cdf_join_masks, aug_cdf_signs_tensors = make_aug_dataset_cdf(cdf_bitmap_list, cdf_predicates_consis,
		                                                                 cdf_joins_consis, consistent_signs,
		                                                               max_num_joins, max_num_predicates)
		with lock:
			shared_batch[:] = []
			shared_batch.append(aug_sample_tensors.share_memory_())
			shared_batch.append(aug_predicate_tensors.share_memory_())
			shared_batch.append(aug_join_tensors.share_memory_())
			shared_batch.append(aug_sample_masks.share_memory_())
			shared_batch.append(aug_predicate_masks.share_memory_())
			shared_batch.append(aug_join_masks.share_memory_())
			shared_batch.append(aug_cdf_sample_tensors.share_memory_())
			shared_batch.append(aug_cdf_predicate_tensors.share_memory_())
			shared_batch.append(aug_cdf_join_tensors.share_memory_())
			shared_batch.append(aug_cdf_sample_masks.share_memory_())
			shared_batch.append(aug_cdf_predicate_masks.share_memory_())
			shared_batch.append(aug_cdf_join_masks.share_memory_())
			shared_batch.append(aug_cdf_signs_tensors.share_memory_())

			updated_event.set()
