# Extension & Analysis: Arbitration Failure, Not Perceptual Blindness

Project for Seminar: Multimodal Language Understanding, Saarland University.
Author: Laxmiraman Gudewar

This project replicates, extends and studies the Encoding-Grounding Dissociation framework from [Nooralahzadeh et al. (2026)](https://arxiv.org/abs/2604.09364), testing whether VLM failures on counterfactual images stem from **arbitration failure** (the model encodes the visual information correctly but its linguistic prior wins at the decision stage) rather than perceptual blindness.

## Key Results

| Attribute | LLaVA-1.5-7B | Qwen2-VL-7B |
|---|---|---|
| Color (replication) | **91.5%** visual-win | 82.4% |
| Size (replication) | 49.2% | 56.3% |
| MMStar Fine-grained (extension) | 70.6% | 75.8% |
| MMStar Coarse (extension) | 71.3% | 73.5% |

The core dissociation replicates: encoding strength weakly correlates with success (|rho| < 0.11), while the final-layer logit gap is strongly predictive (rho > 0.80).

## Repository Structure

```
├── 1_dataset_assembly.ipynb          # Dataset assembly: Color + Size from Visual-Counterfact
├── 2_dataset_expansion.ipynb         # MMStar Fine-grained + Coarse expansion via recolour pipeline
├── 3a_llava_data_collection.ipynb    # MAC + dissociation data collection for LLaVA-1.5-7B (Colab GPU)
├── 3b_qwen_data_collection.ipynb    # MAC + dissociation data collection for Qwen2-VL-7B (Colab GPU)
├── 4_complete_analysis.ipynb         # Full analysis, plots, and results tables
├── metadata.csv                      # Dataset metadata (1,689 samples across 6 attributes)
├── references.bib                    # Bibliography
└── README.md
```

## How to Replicate

### Prerequisites

- Python 3.10+
- Google Colab account (free tier is sufficient)
- HuggingFace account (for Visual-Counterfact dataset access)

### Step 1: Dataset Assembly (local Jupyter, no GPU)

**Run `1_dataset_assembly.ipynb`** to download and process the Visual-Counterfact dataset.
- Downloads Color (493 samples) and Size (727 samples) from HuggingFace (`mgolov/Visual-Counterfact`)
- Requires HuggingFace authentication: `huggingface-cli login`
- Produces the base `arbitration_dataset/` folder with images and `metadata.csv`

**Run `2_dataset_expansion.ipynb`** to generate MMStar-derived attributes.
- Downloads MMStar (`Lin-Chen/MMStar`) and filters perception categories
- Applies the recolour pipeline to generate counterfactual-standard pairs
- Produces Fine-grained (231 samples) and Coarse (230 samples) perception attributes
- Updates `metadata.csv` with the new entries

Output: zip the `arbitration_dataset/` folder and upload to Google Drive at `MyDrive/arbitration_project/`

### Step 2: Data Collection (Google Colab, T4 GPU required)

Upload `arbitration_dataset.zip` to Google Drive, then run each notebook on Colab with GPU runtime:

**Run `3a_llava_data_collection.ipynb`** (~55 min on T4)
- Loads LLaVA-1.5-7B in 8-bit quantisation
- Runs MAC crossover + encoding-grounding dissociation on all 1,689 samples
- Checkpoints to Drive after every sample (crash-safe)
- Saves `llava_results.json` to Drive

**Run `3b_qwen_data_collection.ipynb`** (~45 min on T4)
- Same pipeline for Qwen2-VL-7B-Instruct
- Saves `qwen2vl_results.json` to Drive

Both models run one per session due to GPU memory constraints with 8-bit quantisation.

### Step 3: Analysis (local Jupyter, no GPU)

**Run `4_complete_analysis.ipynb`**
- Place both `llava_results.json` and `qwen2vl_results.json` in the same folder
- Generates all plots and tables in two output folders:
  - `analysis_main/` — 4 main attributes (Color, Size, MMStar-FG, MMStar-Coarse)
  - `analysis_appendix/` — all 6 attributes including Count and Shape

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

- **Notebooks 3a and 3b:** GPU with 16GB+ VRAM (tested on NVIDIA Tesla T4, Google Colab free tier)
- **All other notebooks:** CPU only, no GPU needed

## Reference Paper

```bibtex
@article{nooralahzadeh2026arbitration,
  title={Arbitration Failure, Not Perceptual Blindness},
  author={Nooralahzadeh, Farhad and Rohanian, Omid and Zhang, Yi
          and F{\"u}rst, Jonathan and Stockinger, Kurt},
  journal={arXiv preprint arXiv:2604.09364},
  year={2026}
}
```
