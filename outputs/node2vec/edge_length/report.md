# Node2Vec Edge-Length Sweep Report

## Dataset
- Input file: `data/interim/edge_length.list`
- Unique directed edge rows: 8642
- Nodes: 6749
- Length range: 0.378063 to 328.475412
- Mean length: 15.7677
- Median length: 8.4858
- Long-tail summary: p95=54.0312, p99=100.3051
- Training interpretation: each row `<start> <end> <length>` is a directed weighted edge.
- Weight transform: none. Edge lengths are already positive and valid for node2vec.

## Evaluation Protocol
- Hold-out split: 6914 train edges / 1728 test edges
- Random seed: 42
- Negative samples per positive: 50
- Selection priority: AUC, then MRR, then Hit@10, then score margin

## Ranked Results

| Rank | Config | AUC | MRR | Hit@1 | Hit@5 | Hit@10 | Margin | Corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | d64_wl20_nw10_win5_p1_q1 | 0.7803 | 0.4272 | 0.3391 | 0.4896 | 0.6198 | 0.2147 | -0.0046 |
| 2 | d32_wl20_nw10_win5_p1_q1 | 0.7769 | 0.4242 | 0.3391 | 0.4878 | 0.6100 | 0.2127 | -0.0031 |
| 3 | d32_wl40_nw10_win5_p1_q1 | 0.7714 | 0.4194 | 0.3356 | 0.4821 | 0.6146 | 0.2117 | -0.0009 |
| 4 | d16_wl20_nw10_win5_p1_q1 | 0.7626 | 0.4053 | 0.3264 | 0.4572 | 0.5839 | 0.2042 | -0.0122 |
| 5 | d16_wl40_nw10_win5_p1_q1 | 0.7565 | 0.4068 | 0.3304 | 0.4433 | 0.5781 | 0.2018 | -0.0038 |
| 6 | d64_wl40_nw20_win10_p1_q1 | 0.7372 | 0.4390 | 0.3634 | 0.4994 | 0.5810 | 0.2101 | -0.0177 |
| 7 | d32_wl40_nw20_win10_p0.5_q2 | 0.7324 | 0.4347 | 0.3628 | 0.4873 | 0.5666 | 0.2053 | -0.0109 |
| 8 | d32_wl40_nw20_win10_p2_q0.5 | 0.7268 | 0.4319 | 0.3565 | 0.4954 | 0.5573 | 0.2081 | 0.0005 |
| 9 | d32_wl40_nw20_win10_p1_q1 | 0.7237 | 0.4370 | 0.3652 | 0.4907 | 0.5671 | 0.2048 | -0.0162 |

## Selected Configuration
- Config: `d64_wl20_nw10_win5_p1_q1`
- dimensions=64, walk_length=20, num_walks=10, window=5, p=1.0, q=1.0
- AUC=0.7803, MRR=0.4272, Hit@10=0.6198, score_margin=0.2147
- Positive cosine vs edge-length correlation: -0.0046
- Final delivery model: retrained on the full input file with the selected configuration.

## Output Files
- `best_model.n2v`: final model retrained on the full edge-length dataset
- `best_eval.json`: best run metrics and selection metadata
- `selected_config.json`: chosen hyperparameters
- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs
- `best_model_neighbors.json`: sample nearest neighbours from the final model
