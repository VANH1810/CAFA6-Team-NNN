# 🧬 CAFA6 - Protein Function Prediction

**INT3405E 2 - Final Project - Team NNN**

## Overview

This repository presents **Team NNN's** solution for the CAFA6 (Critical Assessment of Function Annotation) competition on Kaggle. Our team developed a multi-modal ensemble system combining deep learning (ProtBERT) and traditional machine learning (k-mer TF-IDF) to predict protein functions across three Gene Ontology subontologies: Molecular Function (MFO), Biological Process (BPO), and Cellular Component (CCO).

**Team Members:**
- **Tran Quoc Viet Anh** (23021471)
- **Duong Gia Bao** (23021475)
- **Le Duc Anh** (23021463)

**Competition Link:** [CAFA6 - Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

---

## Results

### Model Performance

| Model | Embedding | Base Model | Public Score | Notebook |
|-------|-----------|------------|--------------|----------|
| ProtBERT-MLP Baseline | ProtBERT | MLP | 0.150 | [📓](src/ProtBERT-MLP%20Baseline.ipynb) |
| ProtBERT + TaxoText | ProtBERT | MLP | 0.180 | [📓](src/Multi-Modal-TaxoText-ProtBERT.ipynb) |
| ProtBERT + Hierarchy | ProtBERT | MLP | 0.192 | [📓](src/Hierarchy-Calibrated-Improve-ProtBERT.ipynb) |
| ProtBERT (Multi-Head) | ProtBERT | MLP | 0.196 | [📓](src/Multi-Head%20ProtBERT.ipynb) |
| k-mer TF-IDF (MLP) | k-mer TF-IDF | MLP | 0.208 | [📓](src/k-mer-TF-IDF-MLP.ipynb) |
| Two-Tower Fusion | ProtBERT + k-mer TF-IDF | MLP | 0.213 | [📓](src/ProtBERT-k-mer%20TF-IDF-Fusion.ipynb) |
| Linear TF-IDF Model | k-mer TF-IDF | One-vs-Rest (SGD) | 0.221 | [📓](src/k-mer-TF-IDF-MLP.ipynb) |
| Linear TF-IDF (Legacy Blend) | k-mer TF-IDF | One-vs-Rest (SGD) | 0.255 | [📓](src/k-mer-TF-IDF-MLP.ipynb) |
| **Final System (Ensemble)** | **ProtBERT + k-mer TF-IDF** | **Ensemble - Blending** | **0.304** | [📓](src/cafa-6-blend-goa-negative-propagation.ipynb) |

### Performance Analysis

- **Baseline → Multi-modal**: +0.030 improvement by adding taxonomy and GO text embeddings
- **Multi-modal → Hierarchy**: +0.012 improvement through GO DAG constraint enforcement
- **Deep Learning → Traditional ML**: k-mer TF-IDF outperforms ProtBERT-based models
- **Single Model → Ensemble - Blending**: +0.049 improvement through model blending and GOA integration
- **Overall Improvement**: 2x performance gain from baseline (0.150 → 0.304)


## Key Technical Contributions

1. **Multi-Modal Integration**: Novel fusion of protein sequences, taxonomy, and GO text embeddings
2. **Hierarchy-Aware Learning**: Enforcement of GO DAG constraints during training and inference
3. **Hybrid Architecture**: Combination of transformer-based and traditional ML features

## Project Structure

```
CAFA6-Team-NNN/
├── eda/
│   └── eda.ipynb                                      # Exploratory Data Analysis
├── src/
│   ├── ProtBERT-MLP Baseline.ipynb                   # Baseline model
│   ├── Multi-Modal-TaxoText-ProtBERT.ipynb          # Multi-modal approach
│   ├── Hierarchy-Calibrated-Improve-ProtBERT.ipynb  # Hierarchy-aware model
│   ├── Multi-Head ProtBERT.ipynb                     # Multi-task learning
│   ├── k-mer-TF-IDF-MLP.ipynb                        # Traditional ML approach
│   ├── ProtBERT-k-mer TF-IDF-Fusion.ipynb           # Hybrid fusion model
│   └── cafa-6-blend-goa-negative-propagation.ipynb  # Ensemble & post-processing
└── README.md
```

---

## Technical Stack

### Core Libraries

**Deep Learning & NLP**
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers for ProtBERT models
- `scikit-learn` - Traditional machine learning algorithms

**Bioinformatics**
- `biopython` - Protein sequence processing and FASTA parsing
- `obonet` - Gene Ontology (GO) graph parsing and manipulation
- `networkx` - Graph algorithms for GO hierarchy processing

**Data Processing & Analysis**
- `pandas` - Data manipulation and tabular data processing
- `numpy` - Numerical computing and array operations

**Visualization**
- `matplotlib` - Static plotting
- `seaborn` - Statistical data visualization
- `plotly` - Interactive visualizations

**Utilities**
- `tqdm` - Progress bars for long-running processes
- `scipy` - Scientific computing and statistical analysis

---

## Training & Inference

### 1. Environment Setup

**Prerequisites:**
- Python 3.8 or higher
- CUDA-capable GPU (recommended for ProtBERT models)
- 16GB+ RAM

**Installation:**

```bash
# Clone the repository
git clone https://github.com/VANH1810/CAFA6-Team-NNN.git
cd CAFA6-Team-NNN

# Install required packages
pip install -r requirements.txt
```

**For Kaggle Environment:**

```python
# In Kaggle notebook, install additional packages
!pip install obonet sentence-transformers
```

### 2. Data Preparation

Download the CAFA6 dataset from Kaggle and organize as follows:

```
data/
├── Train/
│   ├── train_sequences.fasta
│   ├── train_terms.tsv
│   ├── train_taxonomy.tsv
│   └── go-basic.obo
├── Test/
│   └── testsuperset.fasta
└── IA.tsv
```

**Kaggle Setup:**
- The dataset is available at: [CAFA6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
- Add the competition dataset to your Kaggle notebook input

### 3. Exploratory Data Analysis

Before training models, explore the dataset:

```bash
# Open EDA notebook
jupyter notebook eda/eda.ipynb
```

**What you'll discover:**
- Protein sequence statistics and amino acid composition
- GO term distribution and class imbalance
- Ontology hierarchy structure
- Species distribution
- Train-test distribution comparison

### 4. Training Individual Models

Each notebook is self-contained. Run them in the suggested order:

**Step 1: Baseline Model**
```bash
jupyter notebook "src/ProtBERT-MLP Baseline.ipynb"
```
- Establishes baseline performance (F-max: 0.150)
- Trains ProtBERT + MLP classifier
- Saves predictions to `submissions/baseline_submission.tsv`

**Step 2: Multi-Modal Model**
```bash
jupyter notebook "src/Multi-Modal-TaxoText-ProtBERT.ipynb"
```
- Adds taxonomy and GO text embeddings
- Expected F-max: 0.180
- Saves to `submissions/multimodal_submission.tsv`

**Step 3: Hierarchy-Calibrated Model**
```bash
jupyter notebook "src/Hierarchy-Calibrated-Improve-ProtBERT.ipynb"
```
- Enforces GO hierarchy constraints
- Expected F-max: 0.192
- Saves to `submissions/hierarchy_submission.tsv`

**Step 4: Multi-Head Model**
```bash
jupyter notebook "src/Multi-Head ProtBERT.ipynb"
```
- Multi-task learning approach
- Expected F-max: 0.196
- Saves to `submissions/multihead_submission.tsv`

**Step 5: K-mer TF-IDF Model**
```bash
jupyter notebook "src/k-mer-TF-IDF-MLP.ipynb"
```
- Traditional ML approach (best single model!)
- Expected F-max: 0.221
- Saves to `submissions/tfidf_submission.tsv`

**Step 6: Two-Tower Fusion**
```bash
jupyter notebook "src/ProtBERT-k-mer TF-IDF-Fusion.ipynb"
```
- Hybrid deep learning + traditional ML
- Expected F-max: 0.213
- Saves to `submissions/fusion_submission.tsv`

### 5. Generating Final Submission

After training all models, create the final ensemble:

```bash
jupyter notebook "src/cafa-6-blend-goa-negative-propagation.ipynb"
```

**This notebook will:**
1. Load all individual model predictions
2. Apply weighted blending (optimized weights)
3. Integrate GOA database annotations
4. Filter negative annotations
5. Apply hierarchy-aware post-processing
6. Generate final submission file: `submissions/final_ensemble.tsv`

**Expected Performance:** F-max = 0.304

### 6. Submission to Kaggle

```bash
# The final submission file will be in:
# submissions/final_ensemble.tsv

# Format: EntryID, term, score
# Example:
# P12345  GO:0008150  0.85
# P12345  GO:0003674  0.72
```

Upload `final_ensemble.tsv` to the competition submission page.


---

## Conclusion

This project demonstrates a comprehensive approach to protein function prediction, combining:

- **Multiple modeling paradigms**: Deep learning (ProtBERT) and traditional ML (TF-IDF)
- **Multi-modal learning**: Leveraging protein sequences, taxonomy, and GO text semantics
- **Hierarchy-aware methods**: Respecting biological constraints in predictions
- **Advanced ensembling**: Weighted blending with external knowledge integration


---

## Contact & Support

For questions or issues, please contact:
- **Tran Quoc Viet Anh**: [GitHub](https://github.com/VANH1810)

