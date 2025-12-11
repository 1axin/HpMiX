# HpMiX 

![architecture](HpMiX.jpg)
HpMiX is an implementation of the 
[HpMiX: A Disease ceRNA Biomarker Prediction Framework Driven by Graph Topology-Constrained Mixup and Hypergraph Residual Enhancement] 


## Overview
HpMiX is designed for predicting disease-ceRNA associations by leveraging the Multi-Structure Hypergraph Weighted Random Walk (MHWRW), residual hypergraph convolution, Graph Topology-Constrained Mixup (GTCM) augmentation technique, cross-channel attention mechanism fusion.

## Installation

To install the required dependencies in your environment, follow these steps:



1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## File Structure
```
HpMiX/
├── Main Results/                                
│   ├── CCA/                    # circRNA-cancer association results
│   ├── LCA/                    # lncRNA-cancer association results
│   └── MCA/                    # miRNA-cancer association results
│
├── Ablation Study/             # Ablation experiments
│
├── Impact of GTCM/             # Effect of the Mixup-enhanced module
│
├── Selection of the Optimal K Value/
│
├── The Impact of Biological Network Embedding Dimensions on Performance/
│
├── CaseStudy/                  # Reproducible case study (Figure 7 & Figure 8)
│   ├── DE/                     # Differential expression analysis (Figure 7)
│   │   ├── TCGA-BLCA data
│   │   ├── BLCA Differential Expression.R
│   │   └── output/             # Heatmaps and boxplots
│   │
│   └── KEGG/                   # KEGG enrichment (Figure 8)
│       ├── KEGG.R
│       ├── c2.cp.kegg.v7.4.symbols.gmt
│       └── output/             # KEGG_barplot, KEGG_dotplot, KEGG_cnetplot
└── miRNADE/ 
        ├── miRNADE.R
        ├── 列注释信息.csv
        └── output/ # miRNA differential expression boxplots & heatmaps
└── README.md


```

## Data Preparation
The dataset should be provided in CSV format with the following structure:
- `train.csv`: Contains node connections with format `[node1, node2]`.

## Usage
### Training the Model
Run the training script using:
```bash
python train.py --epochs 100 --lr 0.01 --batch_size 32
```
Available arguments:
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)

### Important Note on TCGA Data Files

Case Study Data Availability

To ensure full reproducibility of the differential expression analysis (Figures 7 and 8), we provide the original TCGA expression matrix used in our experiments. Since the historical version of the TCGA-BLCA dataset used in this study is no longer available on the current UCSC Xena platform and the file size exceeds GitHub’s standard limit (100 MB), the data has been uploaded via GitHub Releases.

The dataset can be downloaded at:

TCGA-BLCA FPKM expression matrix (historical version)
Download from GitHub Releases

File: TCGA-BLCA.htseq_fpkm.tsv.gz

After downloading, place the file into the corresponding folder in CaseStudy/DE/ or CaseStudy/KEGG/ as required by the R scripts.


## License
This project is licensed under the MIT License. Feel free to modify and distribute as needed.

---
For any questions or contributions, feel free to reach out!
