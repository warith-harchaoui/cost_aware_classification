# Cost-aware classification + fraud benchmark (IEEE-CIS)

Corporate research/engineering repo for **cost-aware classification** with **example-dependent misclassification costs**.

Included losses
- **Cross-Entropy baselines** (in the example runner)
  - `cross_entropy`
  - `cross_entropy_weighted` with sample weights $w_i = C_i[y_i, 1-y_i]$
- **SinkhornFenchelYoungLoss**: implicit Fenchel–Young loss with a Frank–Wolfe inner solver
- **SinkhornEnvelopeLoss**: entropic OT-loss with envelope (implicit-ish) gradients
- **SinkhornFullAutodiffLoss**: entropic OT-loss with full autodiff through Sinkhorn

## Installation

Install conda if you haven't already.

See [conda](https://harchaoui.org/warith/4ml) for installation instructions.

Then:
```bash
PROJECT=cost_aware_fraud
GITHUB_OWNER=warith-harchaoui

REPOURL=https://github.com/$GITHUB_OWNER/$PROJECT.git
ENV=env4$PROJECT

git clone $REPOURL
cd $PROJECT
conda update -y -n base -c defaults conda
conda create -y -n $ENV python=3.10
conda activate $ENV
conda install -y pip
pip install -r requirements.txt
```


## IEEE-CIS fraud experiment

### Download the dataset

```bash
rm -rf ieee-fraud-detection.zip ieee-fraud-detection || true
mkdir ieee-fraud-detection
wget -c http://deraison.ai/ai/ieee-fraud-detection.zip
unzip ieee-fraud-detection.zip -d ieee-fraud-detection
```


### Run one method

```bash
python -m examples.fraud_detection --loss cross_entropy --epochs 5 --run-id demo1
```


### Run all methods

```bash
python -m examples.fraud_detection --loss all --epochs 5 --run-id demo1
```


### Resume training (“continue”)

This continues for **additional epochs**:

```bash
python -m examples.fraud_detection --loss all --epochs 3 --run-id demo1 --resume
```


Artifacts are written under:

```text
fraud_output/<run-id>/<loss_name>/
  checkpoint_last.pt
  checkpoint_best.pt
  train_ema_metrics.csv
  probe_metrics.csv
  precision_recall_curve.png
  ...
```


## Documentation

- `docs/math.md` — derivations + the explicit mapping between $\varepsilon$ and POT’s `reg`
- `docs/fraud_business_and_cost_matrix.md` — business value model and the per-example cost matrix construction

## 📜 License

**Unlicense** — This is free and unencumbered software released into the public domain.  
See [UNLICENSE](https://unlicense.org) for details.
