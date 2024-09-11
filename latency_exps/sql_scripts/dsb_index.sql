create index date_dim_id_btree_index on date_dim using btree(d_date_sk);
create index customer_demographics_id_btree_index on customer_demographics using btree(cd_demo_sk);
create index household_demographics_id_btree_index on household_demographics using btree(hd_demo_sk);
create index store_id_btree_index on store using btree(s_store_sk);

create index store_sales_id_btree_index1 on store_sales using btree(ss_sold_date_sk);
create index store_sales_id_btree_index2 on store_sales using btree(ss_cdemo_sk);
create index store_sales_id_btree_index3 on store_sales using btree(ss_hdemo_sk);
create index store_sales_id_btree_index4 on store_sales using btree(ss_store_sk);


analyze store_sales;
analyze date_dim;
analyze customer_demographics;
analyze household_demographics;
analyze store;