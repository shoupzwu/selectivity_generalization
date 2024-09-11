# A Practical Theory of Generalization in Selectivity Learning
- In this paper, we first demonstrate that _**a hypothesis class is learnable if predictors are induced by signed measures**_.  More importantly, we establish, **_under mild assumptions, that predictors from this class exhibit favorable OOD generalization error bounds_**. 

- Our theoretical results  allow us to design two improvement strategies for existing query-driven models: 1) **NeuroCDF**, that models the underlying cumulative distribution functions (CDFs) instead of the ultimate query selectivity; 2) a general training framework (**SeConCDF**) to enhance OOD generalization, without compromising good in-distribution generalization with any loss function. 

- This repository includes the codes, queries and scripts for our paper: A Practical Theory of Generalization in Selectivity Learning.

## Experiments of NeuroCDF
This section contains the experiments over a synthetic dataset from a highly-correlated 10-dimensional Gaussian distribution. 

We provide the codes for **LW-NN** (MSCN shares the same results with LW-NN in this section, and we will focus more on MSCN in next section), **LW-NN+NeuroCDF** (LW-NN trained with NeuroCDF), and **LW-NN+SeConCDF** (LW-NN trained with SeConCDF). 

The motivational experiments are minimal and clearly demonstrate 1) the superiority  of NeuroCDF over LW-NN on OOD generalization; and 2) the effectiveness of CDF self-consistency training of SeConCDF in enhancing LW-NN's generalization on OOD queries. 

Below are steps to reproduce the experiments. Have fun!

1. Please enter the [synthetic_data](synthetic_data) directory.
2. To reproduce the result of LW-NN, please run
```shell
    $ python train_LW-NN.py
```
3. To reproduce the result of LW-NN+NeuroCDF, please run
```shell
    $ python train_LW-NN+NeuroCDF.py
```
4. To reproduce the result of LW-NN+SeConCDF, please run
```shell
    $ python train_LW-NN+SeConCDF.py
```
5. Notice the RMSE and Median Qerror for both types  (In-Distribution and Out-of-Distribution) of test workloads across each model and compare the results!

## Experiments of SeConCDF
This section contains the experiments related to **SeConCDF**. You'll need a GPU for this section.

We first focus on **single-table** (Census) experiments. Below are steps to reproduce the experimental results.

1. Please enter the [single_table](single_table) directory.
2. To reproduce the result of LW-NN, please run
```shell
    $ python train_LW-NN
```
3. To reproduce the result of LW-NN+SeConCDF (LW-NN trained with SeConCDF), please run
```shell
    $ python train_LW-NN+SeConCDF
``` 

4. Similarly, run `train_MSCN.py` and `train_MSCN+SeConCDF.py` to reproduce the experiments of MSCN and MSCN+SeConCDF (MSCN trained with SeConCDF)


Then, we move on to **multi-table**  experiments. Below are steps to reproduce the experimental results.
1. Please enter the [multi_table](multi_table) directory.
2. Due to the file size limit of Github, please download the zipped directory of bitmaps from [this link](https://drive.google.com/file/d/1eBd4SJg8i8h9yv-dKDj-8ffWWqOyL9Qi/view?usp=sharing) and unzip it in current directory.
3. Similarly, download the zipped directory of workloads from [this link](https://drive.google.com/file/d/1bwlr_glPDQjFbiLw3613qZlUMDuc8CKy/view?usp=sharing) and unzip it in current directory.
4. To reproduce the results on IMDb, please run
```shell
    $ python train_MSCN_imdb --shift granularity
    $ python train_MSCN+SeConCDF_imdb --shift granularity
```

where the parameter `shift` controls the OOD scenario (`granularity` for granularity shift and `center` for center move).


```shell
    $ python train_MSCN_imdb --shift center
    $ python train_MSCN+SeConCDF_imdb --shift center
```
You will observe substantial improvements (in terms of both RMSE and Q-error) with MSCN+SeConCDF compared to MSCN on OOD queries, which showcases the advantages of SeConCDF in multi-table cases.

5. Also, you can run `train_MSCN_dsb.py` and `train_MSCN+SeConCDF_dsb.py` to reproduce the experiments on DSB.


## Query Latency Performance

1. Please download and install the modified PostgreSQL from [CEB](https://github.com/learnedsystems/CEB) or [another project](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master).
2. Download the IMDb dataset from [here](http://homepages.cwi.nl/~boncz/job/imdb.tgz), and download the populated DSB dataset used in the paper from [here](https://mega.nz/file/iCI2hRhY#96_uiKFvFq0HUcoNNPRnVtMy5BbJ-1QuSry2d3l83xk).
3. Please load the data into PostgreSQL and construct indexes using scripts in [this directory](latency_exps/sql_scripts).
4. Run PostgreSQL on the workloads provided in [this directory](latency_exps/workloads), which contains both in-distribution and OOD workloads for each OOD scenario, over both datasets.

