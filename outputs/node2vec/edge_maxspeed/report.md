# Node2Vec Edge-Maxspeed Sweep Report

## Dataset
- Input file: `data/interim/edge_maxspeed.list`
- Unique directed edge rows: 8642
- Nodes: 6749
- Speed range: 8.0 to 100.0
- Mean speed: 51.1083
- Median speed: 50.0000
- Speed distribution: 8=6, 30=36, 50=8203, 70=252, 80=87, 100=58
- Training interpretation: each row `<start> <end> <maxspeed>` is a directed weighted edge.
- Weight transform: none. Maxspeed values are already positive and valid for node2vec.

## Evaluation Protocol
- Hold-out split: 6914 train edges / 1728 test edges
- Random seed: 42
- Negative samples per positive: 50
- Selection priority: AUC, then MRR, then Hit@10, then score margin

## Ranked Results

| Rank | Config | AUC | MRR | Hit@1 | Hit@5 | Hit@10 | Margin | Corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | d64_wl20_nw10_win5_p1_q1 | 0.7846 | 0.4352 | 0.3530 | 0.4936 | 0.6354 | 0.2193 | -0.0447 |
| 2 | d32_wl20_nw10_win5_p1_q1 | 0.7781 | 0.4288 | 0.3484 | 0.4844 | 0.6181 | 0.2142 | -0.0430 |
| 3 | d32_wl40_nw10_win5_p1_q1 | 0.7743 | 0.4263 | 0.3420 | 0.4890 | 0.6221 | 0.2150 | -0.0425 |
| 4 | d16_wl20_nw10_win5_p1_q1 | 0.7723 | 0.4152 | 0.3351 | 0.4664 | 0.5897 | 0.2122 | -0.0336 |
| 5 | d16_wl40_nw10_win5_p1_q1 | 0.7641 | 0.4142 | 0.3316 | 0.4705 | 0.5966 | 0.2094 | -0.0336 |
| 6 | d32_wl40_nw20_win10_p0.5_q2 | 0.7328 | 0.4305 | 0.3565 | 0.4797 | 0.5712 | 0.2073 | -0.0948 |
| 7 | d64_wl40_nw20_win10_p1_q1 | 0.7264 | 0.4397 | 0.3669 | 0.4971 | 0.5694 | 0.2080 | -0.0999 |
| 8 | d32_wl40_nw20_win10_p1_q1 | 0.7250 | 0.4347 | 0.3657 | 0.4832 | 0.5666 | 0.2055 | -0.0976 |
| 9 | d32_wl40_nw20_win10_p2_q0.5 | 0.7247 | 0.4306 | 0.3600 | 0.4809 | 0.5561 | 0.2061 | -0.1044 |

## Selected Configuration
- Config: `d64_wl20_nw10_win5_p1_q1`
- dimensions=64, walk_length=20, num_walks=10, window=5, p=1.0, q=1.0
- AUC=0.7846, MRR=0.4352, Hit@10=0.6354, score_margin=0.2193
- Positive cosine vs maxspeed correlation: -0.0447
- Final delivery model: retrained on the full input file with the selected configuration.

## Output Files
- `best_model.n2v`: final model retrained on the full edge-maxspeed dataset
- `best_eval.json`: best run metrics and selection metadata
- `selected_config.json`: chosen hyperparameters
- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs
- `best_model_neighbors.json`: sample nearest neighbours from the final model
