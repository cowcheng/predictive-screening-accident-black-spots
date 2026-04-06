# Node2Vec Edge-Ref Sweep Report

## Dataset
- Input file: `data/interim/edge_ref.list`
- Unique directed edge rows: 8642
- Nodes: 6749
- Raw ref range: 0.0 to 6.0
- Mean raw ref: 0.157487
- Mean transformed ref weight: 0.158487
- Ref distribution: 0=8188, 1=218, 3=52, 5=117, 6=67
- Training interpretation: each row `<start> <end> <ref>` is a directed weighted edge.
- Weight transform: `abs(ref) + 0.001` so the dominant `0.0` edges remain near-zero but valid for node2vec.

## Evaluation Protocol
- Hold-out split: 6914 train edges / 1728 test edges
- Random seed: 42
- Negative samples per positive: 50
- Selection priority: AUC, then MRR, then Hit@10, then score margin

## Ranked Results

| Rank | Config | AUC | MRR | Hit@1 | Hit@5 | Hit@10 | Margin | Corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | d32_wl40_nw10_win5_p1_q1 | 0.7800 | 0.4368 | 0.3536 | 0.4919 | 0.6134 | 0.2177 | -0.0176 |
| 2 | d64_wl20_nw10_win5_p1_q1 | 0.7796 | 0.4268 | 0.3380 | 0.4948 | 0.6331 | 0.2165 | -0.0165 |
| 3 | d32_wl20_nw10_win5_p1_q1 | 0.7718 | 0.4251 | 0.3420 | 0.4774 | 0.6152 | 0.2133 | -0.0187 |
| 4 | d16_wl40_nw10_win5_p1_q1 | 0.7650 | 0.4069 | 0.3270 | 0.4566 | 0.5845 | 0.2089 | -0.0076 |
| 5 | d16_wl20_nw10_win5_p1_q1 | 0.7571 | 0.4013 | 0.3212 | 0.4525 | 0.5689 | 0.2019 | -0.0123 |
| 6 | d32_wl40_nw20_win10_p2_q0.5 | 0.7400 | 0.4340 | 0.3623 | 0.4826 | 0.5712 | 0.2132 | -0.0909 |
| 7 | d32_wl40_nw20_win10_p1_q1 | 0.7300 | 0.4382 | 0.3652 | 0.4931 | 0.5689 | 0.2095 | -0.0790 |
| 8 | d64_wl40_nw20_win10_p1_q1 | 0.7298 | 0.4424 | 0.3692 | 0.5006 | 0.5683 | 0.2094 | -0.0899 |
| 9 | d32_wl40_nw20_win10_p0.5_q2 | 0.7254 | 0.4310 | 0.3605 | 0.4774 | 0.5561 | 0.2043 | -0.0849 |

## Selected Configuration
- Config: `d32_wl40_nw10_win5_p1_q1`
- dimensions=32, walk_length=40, num_walks=10, window=5, p=1.0, q=1.0
- AUC=0.7800, MRR=0.4368, Hit@10=0.6134, score_margin=0.2177
- Positive cosine vs transformed ref correlation: -0.0176
- Final delivery model: retrained on the full input file with the selected configuration.

## Output Files
- `best_model.n2v`: final model retrained on the full edge-ref dataset
- `best_eval.json`: best run metrics and selection metadata
- `selected_config.json`: chosen hyperparameters
- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs
- `best_model_neighbors.json`: sample nearest neighbours from the final model
