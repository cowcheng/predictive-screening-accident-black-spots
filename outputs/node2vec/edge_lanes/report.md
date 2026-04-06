# Node2Vec Edge-Lanes Sweep Report

## Dataset
- Input file: `data/interim/edge_lanes.list`
- Unique directed edge rows: 8642
- Nodes: 6749
- Lane range: 1.0 to 6.0
- Mean lanes: 1.7717
- Lane-count distribution: 1=4272, 2=2600, 3=1296, 4=423, 5=47, 6=4
- Training interpretation: each row `<start> <end> <lanes>` is a directed weighted edge.
- Weight transform: none. Lane counts are already positive and valid for node2vec.

## Evaluation Protocol
- Hold-out split: 6914 train edges / 1728 test edges
- Random seed: 42
- Negative samples per positive: 50
- Selection priority: AUC, then MRR, then Hit@10, then score margin

## Ranked Results

| Rank | Config | AUC | MRR | Hit@1 | Hit@5 | Hit@10 | Margin | Corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | d64_wl20_nw10_win5_p1_q1 | 0.7758 | 0.4303 | 0.3455 | 0.4948 | 0.6325 | 0.2141 | -0.2150 |
| 2 | d32_wl20_nw10_win5_p1_q1 | 0.7722 | 0.4231 | 0.3414 | 0.4734 | 0.6140 | 0.2119 | -0.2046 |
| 3 | d32_wl40_nw10_win5_p1_q1 | 0.7683 | 0.4202 | 0.3391 | 0.4786 | 0.6065 | 0.2107 | -0.2216 |
| 4 | d16_wl20_nw10_win5_p1_q1 | 0.7638 | 0.4058 | 0.3275 | 0.4514 | 0.5909 | 0.2072 | -0.1947 |
| 5 | d16_wl40_nw10_win5_p1_q1 | 0.7596 | 0.3978 | 0.3148 | 0.4433 | 0.5787 | 0.2052 | -0.2093 |
| 6 | d32_wl40_nw20_win10_p1_q1 | 0.7337 | 0.4400 | 0.3657 | 0.5064 | 0.5689 | 0.2102 | -0.2201 |
| 7 | d64_wl40_nw20_win10_p1_q1 | 0.7326 | 0.4407 | 0.3652 | 0.5012 | 0.5781 | 0.2099 | -0.2199 |
| 8 | d32_wl40_nw20_win10_p2_q0.5 | 0.7296 | 0.4318 | 0.3605 | 0.4780 | 0.5602 | 0.2088 | -0.2258 |
| 9 | d32_wl40_nw20_win10_p0.5_q2 | 0.7262 | 0.4316 | 0.3617 | 0.4855 | 0.5573 | 0.2040 | -0.2273 |

## Selected Configuration
- Config: `d64_wl20_nw10_win5_p1_q1`
- dimensions=64, walk_length=20, num_walks=10, window=5, p=1.0, q=1.0
- AUC=0.7758, MRR=0.4303, Hit@10=0.6325, score_margin=0.2141
- Positive cosine vs lane-count correlation: -0.2150
- Final delivery model: retrained on the full input file with the selected configuration.

## Output Files
- `best_model.n2v`: final model retrained on the full edge-lanes dataset
- `best_eval.json`: best run metrics and selection metadata
- `selected_config.json`: chosen hyperparameters
- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs
- `best_model_neighbors.json`: sample nearest neighbours from the final model
