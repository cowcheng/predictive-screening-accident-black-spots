# Predictive Screening of Accident Black Spots

Predictive Screening of Accident Black Spots based on Deep Neural Models of Road Networks and Facilities: A Case Study based on a District in Hong Kong.

## Repository layout

- `scripts/data_collect/`: raw data collection for roads, traffic news, and OpenStreetMap.
- `scripts/data_preprocessing/`: graph artifact generation and edge-feature enrichment.
- `models/node2vec/`: Node2Vec training, evaluation, and sweep scripts.
- `models/classifier/`: classifier model definition, training, and evaluation.
- `data/raw/`, `data/interim/`, `data/final/`: raw inputs, derived artifacts, and finalized model inputs.
- `outputs/node2vec/`, `outputs/classifier/`: saved embeddings, checkpoints, logs, and evaluation outputs.
- `papers/`: reference papers and project background material.

## Environment setup

The checked-in virtual environment uses Python 3.11. A minimal setup looks like this:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core runtime dependencies are listed in `requirements.txt`. Optional dependencies for slope generation:

```bash
pip install elevation
```

`scripts/data_preprocessing/generate_road_structure_all_node_list.py` also requires GDAL / `osgeo` for DEM-backed slope extraction. If those packages are unavailable, use `--skip-slope` or `--slope-zero-fallback`.

## End-to-end workflow

### 1. Collect raw source data

```bash
python scripts/data_collect/collect_roads_data.py
python scripts/data_collect/collect_news_data.py
python scripts/data_collect/collect_osm_data.py --address "Nathan Road" --clean-geometry
```

These commands populate `data/raw/` with road metadata, traffic news, and OSM node/edge tables.

### 2. Generate graph artifacts

```bash
python scripts/data_preprocessing/generate_road_structure_all_node_list.py --mode all --skip-slope
python scripts/data_preprocessing/generate_road_structure_all_edge_list.py --mode all
```

This step writes node-based artifacts such as `node_angle.list` and edge lists such as `edge_length.list`, `edge_maxspeed.list`, `edge_time.list`, `edge_lanes.list`, and `edge_ref.list` under `data/interim/`.

If you want DEM-based slope features, remove `--skip-slope` after installing the optional elevation dependencies.

### 3. Enrich OSM edges with accident features

```bash
python scripts/data_preprocessing/add_edges_extra_info.py --write-road-accident-output
```

This merges road/news signals with OSM edges and writes enriched files such as `data/interim/osm_edge_info_with_accident.csv`.

### 4. Train Node2Vec embeddings

Train one embedding model per edge feature. Example for `length`:

```bash
python -m models.node2vec.train \
  --edgelist data/interim/edge_length.list \
  --output outputs/node2vec/edge_length/model.n2v

python -m models.node2vec.eval \
  --model outputs/node2vec/edge_length/model.n2v \
  --mode edge-embeddings \
  --edgelist data/interim/edge_length.list \
  --edge-output outputs/node2vec/edge_length/best_model_edges.npy
```

Repeat the same pattern for `maxspeed`, `time`, `lanes`, and `ref`. For experiment automation, see the sweep scripts in `models/node2vec/sweep_edge_*.py`.

### 5. Train and evaluate the classifier

```bash
python -m models.classifier.train
python -m models.classifier.eval --checkpoint outputs/classifier/train/<run>/best.pt
```

The classifier expects a finalized edge-feature table under `data/` and the generated embedding matrices under `outputs/node2vec/edge_<feature>/`. If your artifact names or locations differ from the defaults, run each CLI with `--help` and pass explicit paths.

## Artifacts and data handling

- Use `data/raw/` for collected source tables.
- Use `data/interim/` for generated `.list` files, enriched CSVs, and temporary elevation caches.
- Use `data/final/` for stable model-ready tables consumed by the classifier.
- Use `outputs/node2vec/` and `outputs/classifier/` for trained artifacts and evaluation results.

Large CSVs, model files, archives, and papers are tracked with Git LFS through `.gitattributes`.

## Notes

- There is no committed `requirements.txt` or `pyproject.toml`; the commands above are the canonical entrypoints.
- Most scripts are written as standalone CLIs. Use `--help` on any script or module to inspect all available options.
- The default data sources and text processing are tuned for Hong Kong roads and traffic incident reporting.
