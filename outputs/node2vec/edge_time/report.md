# Node2Vec Edge-Time Sweep Report

## Dataset
- Input file: `data/interim/edge_time.list`
- Unique directed edge rows: 8642
- Nodes: 6749
- Time range: 0.027221 to 23.086442
- Mean time: 1.113224
- Median time: 0.599724
- Long-tail summary: p95=3.826312, p99=7.172728
- Training interpretation: each row `<start> <end> <time>` is a directed weighted edge.
- Weight transform: none. Travel-time values are already positive and valid for node2vec.

## Evaluation Protocol
- Hold-out split: 6914 train edges / 1728 test edges
- Random seed: 42
- Negative samples per positive: 50
- Selection priority: AUC, then MRR, then Hit@10, then score margin

## Ranked Results

| Rank | Config | AUC | MRR | Hit@1 | Hit@5 | Hit@10 | Margin | Corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | d64_wl20_nw10_win5_p1_q1 | 0.7834 | 0.4280 | 0.3391 | 0.4948 | 0.6279 | 0.2160 | 0.0099 |
| 2 | d32_wl20_nw10_win5_p1_q1 | 0.7781 | 0.4191 | 0.3356 | 0.4826 | 0.6227 | 0.2129 | 0.0117 |
| 3 | d32_wl40_nw10_win5_p1_q1 | 0.7657 | 0.4200 | 0.3339 | 0.4797 | 0.6019 | 0.2090 | 0.0020 |
| 4 | d16_wl20_nw10_win5_p1_q1 | 0.7593 | 0.3988 | 0.3183 | 0.4514 | 0.5671 | 0.2022 | 0.0071 |
| 5 | d16_wl40_nw10_win5_p1_q1 | 0.7546 | 0.4031 | 0.3235 | 0.4566 | 0.5648 | 0.2017 | -0.0059 |
| 6 | d32_wl40_nw20_win10_p1_q1 | 0.7338 | 0.4248 | 0.3513 | 0.4826 | 0.5660 | 0.2074 | -0.0022 |
| 7 | d64_wl40_nw20_win10_p1_q1 | 0.7302 | 0.4362 | 0.3611 | 0.4988 | 0.5706 | 0.2080 | 0.0007 |
| 8 | d32_wl40_nw20_win10_p2_q0.5 | 0.7288 | 0.4351 | 0.3640 | 0.4896 | 0.5677 | 0.2074 | -0.0030 |
| 9 | d32_wl40_nw20_win10_p0.5_q2 | 0.7277 | 0.4268 | 0.3536 | 0.4832 | 0.5660 | 0.2020 | -0.0094 |

## Selected Configuration
- Config: `d64_wl20_nw10_win5_p1_q1`
- dimensions=64, walk_length=20, num_walks=10, window=5, p=1.0, q=1.0
- AUC=0.7834, MRR=0.4280, Hit@10=0.6279, score_margin=0.2160
- Positive cosine vs travel-time correlation: 0.0099
- Final delivery model: retrained on the full input file with the selected configuration.

## Output Files
- `best_model.n2v`: final model retrained on the full edge-time dataset
- `best_eval.json`: best run metrics and selection metadata
- `selected_config.json`: chosen hyperparameters
- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs
- `best_model_neighbors.json`: sample nearest neighbours from the final model
