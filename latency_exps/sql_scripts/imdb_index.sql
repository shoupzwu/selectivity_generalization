create index mi_movie_id_btree_index on movie_info using btree(movie_id);
create index mi_idx_movie_id_btree_index on movie_info_idx using btree(movie_id);
create index mc_movie_id_btree_index on movie_companies using btree(movie_id);
create index mk_movie_id_btree_index on movie_keyword using btree(movie_id);
create index ci_movie_id_btree_index on cast_info using btree(movie_id);
