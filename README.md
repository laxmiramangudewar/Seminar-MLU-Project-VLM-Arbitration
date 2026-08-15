# Extension & Analysis: Arbitration Failure, Not Perceptual Blindness

Project for Seminar: Multimodal Language Understanding, Saarland University.
Author: Laxmiraman Gudewar

This project replicates, extends and studies the Encoding-Grounding Dissociation framework from [Nooralahzadeh et al. (2026)](https://arxiv.org/abs/2604.09364), testing whether VLM failures on counterfactual images stem from **arbitration failure** (the model encodes the visual information correctly but its linguistic prior wins at the decision stage) rather than perceptual blindness.

## Key Results

| Attribute | LLaVA-1.5-7B | Qwen2-VL-7B |
|---|---|---|
| Color (replication) | 91.5% visual-win | 82.4% |
| Size (replication) | 49.2% | 56.3% |
| MMStar Fine-grained (extension) | 70.6% | 75.8% |
| MMStar Coarse (extension) | 71.3% | 73.5% |

The core dissociation replicates: encoding strength weakly correlates with success (|ρ| < 0.11), while the final-layer logit gap is strongly predictive (ρ > 0.80).

## Repository Structure

```
├── phase2_dataset_assembly.ipynb      # Dataset assembly (Color, Size from Visual-Counterfact)
├── phase2_expansion.ipynb             # MMStar Fine-grained + Coarse expansion
├── notebook_a_data_collection.ipynb   # MAC + dissociation data collection (Colab GPU)
├── notebook_b_analysis.ipynb          # Analysis, plots, and results tables
├── metadata.csv                       # Complete dataset metadata (1,689 samples)
└── references.bib                     # Bibliography
```

## How to Replicate

### Prerequisites

- Python 3.10+
- Google Colab account (free tier is sufficient)
- HuggingFace account (for Visual-Counterfact dataset access)

### Step 1: Dataset Assembly (local Jupyter)

1. Run `phase2_dataset_assembly.ipynb` to download and process the Visual-Counterfact dataset (Color: 493, Size: 727 samples). Requires HuggingFace authentication:
   ```
   huggingface-cli login
   ```
   The dataset identifier is `mgolov/Visual-Counterfact`.

2. Run `phase2_expansion.ipynb` to generate MMStar-derived attributes. This downloads MMStar (`Lin-Chen/MMStar`), filters perception categories, and applies the recolour pipeline to generate counterfactual-standard pairs (Fine-grained: 231, Coarse: 230 samples).

3. Output: `arbitration_dataset/` folder containing all images + `metadata.csv`. Zip this folder for upload to Google Drive.

### Step 2: Data Collection (Google Colab, GPU required)

1. Upload `arbitration_dataset.zip` to Google Drive at `MyDrive/arbitration_project/`.

2. Upload `notebook_a_data_collection.ipynb` to Google Colab. Set runtime to T4 GPU.

3. **Run 1 (LLaVA):** Set `MODEL_CHOICE = "llava"` at the top of the notebook. Run all cells. Takes approximately 55 minutes on a T4 GPU. Produces `llava_results.json` on Drive.

4. **Run 2 (Qwen2-VL):** Restart runtime, change `MODEL_CHOICE = "qwen2vl"`, run all cells. Takes approximately 45 minutes. Produces `qwen2vl_results.json` on Drive.

Both models are loaded in 8-bit quantisation via BitsAndBytes. Results are checkpointed to Drive after every sample, so interrupted sessions resume automatically.

### Step 3: Analysis (local Jupyter)

1. Download `llava_results.json` and `qwen2vl_results.json` from Google Drive.

2. Place both files in the same directory as `notebook_b_analysis.ipynb`.

3. Run all cells. Produces plots and tables in `analysis_main/` (4 main attributes) and `analysis_appendix/` (all 6 including Count and Shape).

## Models

| Model | HuggingFace ID | Layers | Quantisation |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 32 | 8-bit (BitsAndBytes) |
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | 28 | 8-bit (BitsAndBytes) |

## Dataset

| Attribute | Source | N | Role |
|---|---|---|---|
| Color | Visual-Counterfact | 493 | Replication |
| Size | Visual-Counterfact | 727 | Replication |
| MMStar Fine-grained | MMStar + recolour pipeline | 231 | Extension |
| MMStar Coarse | MMStar + recolour pipeline | 230 | Extension |
| Count | Manual curation | 3 | Preliminary (appendix) |
| Shape | Manual + MMStar | 5 | Preliminary (appendix) |

## Hardware Requirements

- **Data collection (Notebook A):** GPU with 16GB+ VRAM. Tested on NVIDIA Tesla T4 (Google Colab free tier). Models loaded one per session due to memory constraints with 8-bit quantisation.
- **All other notebooks:** CPU only. No GPU required for dataset assembly or analysis.

## Citation

This project is based on:

```bibtex
@article{nooralahzadeh2026arbitration,
  title={Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts},
  author={Nooralahzadeh, Farhad and Rohanian, Omid and Zhang, Yi and F{\"u}rst, Jonathan and Stockinger, Kurt},
  journal={arXiv preprint arXiv:2604.09364},
  year={2026}
}
```
