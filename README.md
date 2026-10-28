# UIT.Viper at ViPRA 2026

Vietnamese physics problem solving with targeted data augmentation and LoRA fine-tuning of **Qwen3-4B-Instruct-2507**. The system returns a JSON object containing an explanation, an answer, and its unit.

- **Task:** text-only Vietnamese Physics Reasoning and Answering, [VLSP 2026](https://vlsp.org.vn/node/214).
- **Data:** 677 original training problems + 120 synthetic problems; 339 development questions.
- **Training:** 5 epochs, seed 42, LoRA rank 16 / alpha 32, learning rate `1e-4`, effective batch size 16. The reproduction recipe exports **epoch 4**.
- **Inference:** beam search with width 3, up to 1,600 new tokens, and a 60-second local deadline.
- **Official result:** 58.57% (41/70), second among teams with reported scores.

## Reproduce the Qwen3 system

Use a Linux CUDA environment with **two GPUs**; the original training runs used Kaggle T4 x2. Enable Internet access to download the pinned backbone and tokenizer. CUDA-enabled PyTorch 2.8+ is required; package versions are specified in `requirements.txt`.

### 1. Install

```bash
git clone --branch Nhotest --single-branch https://github.com/NgocNhoNguyen2605n/VLSP.git
cd VLSP
python -m pip install -r requirements.txt
```

On Kaggle, run these commands in a terminal, or prefix shell commands with `!` in a notebook. If an incompatible preinstalled `torchao` prevents PEFT from importing, uninstall `torchao` and restart the notebook kernel.

### 2. Prepare the data

Prepare the shared-task CSVs and the augmented training file used in the experiments, then place them as follows. Access to the original data follows the organizers' terms; the augmented training file and model weights are not bundled with this repository.

```text
data/reproduction/
├── train_original_plus_120.csv   # 797 training examples
└── dev_dataset.csv              # 339 development examples
```

Both CSVs must have the columns `id,question,cot,answer,unit`. Use the exact experiment versions; the older CSVs in `data/` are not substitutes for the augmented training set.

Expected SHA-256 fingerprints:

```text
train_original_plus_120.csv  dcea0207d41196da65c6c7a57ea57117c196e927080382b5cc5b7ccffa4f606e
dev_dataset.csv             6e6024e03b13c532fd285ebc2085e583c9f965afddd00fde2628fa77b67cce2c
```

If your files are elsewhere, update `data_root`, `train_file`, and `dev_file` in [`configs/reproduce.json`](configs/reproduce.json). Keep the CSV contents unchanged.

### 3. Check and train

```bash
python -m baseline.check --config configs/reproduce.json
python -m baseline.experiment --config configs/reproduce.json
```

This trains for five epochs, records development loss after each epoch, and exports the epoch-4 adapter. Development examples do not enter gradient updates. Use a new `--output-dir` for another run.

The reproduction config fixes the model revision and training settings. [`configs/baseline.json`](configs/baseline.json) retains the historical epoch-5 recipe used for the augmentation comparison.

### 4. Evaluate on development data

```bash
python -m baseline.evaluate --run-dir outputs/reproduce_epoch4 --dev-only
```

`--dev-only` skips test evaluation, including automatic test-file discovery. The local scorer reports answer-and-unit correctness with 5% relative tolerance, JSON validity, and deadline violations. It does not grade explanation correctness or perform general SI-prefix conversion.

```text
outputs/reproduce_epoch4/
├── adapter/                    # Selected epoch-4 adapter and tokenizer
├── run.json                    # Configuration, dataset hashes, and environment
├── selection.json              # Checkpoint selection record
├── search/epoch-*/             # Saved adapters from all five epochs
└── evaluation/
    ├── metrics.json
    └── predictions.jsonl
```

Local development scores and timing are separate from the official evaluation. The paper's original-versus-augmented comparison uses epoch 5; the backbone benchmark uses epoch 4. Seed and revision pinning support reproducibility, but do not guarantee identical results across hardware and library versions.

## Repository guide

| Path | Purpose |
| --- | --- |
| `baseline/` | Data loading, LoRA training, decoding, and local evaluation |
| `configs/` | Training, reproduction, and serving configurations |
| `data_preparation/` | Earlier data preparation utilities |
| `submission/` | Adapter preparation and shared-task API integration |
| `tests/` | Unit and integration tests |

For deployment, see [the serving guide](SUBMISSION_GUIDE_VI.md) and [the Apple Silicon guide](MAC_SERVING_VI.md). The older notebooks include exploratory and sweep workflows; the commands above specify the reproduction recipe.
