
# The Lazy Librarian, A Recommender System Kaggle Competition

This repository contains the code and resources for "The Lazy Librarian," a recommender system project built for a Kaggle competition. The goal is to build an AI that recommends books to users based on their interaction history and book metadata.

## Project Structure

This project follows a modular structure to ensure reproducibility and scalability.

```text
lazy_librarian/
│
├── data/
│   ├── raw/                  # Original datasets (interactions.csv, items.csv)
│   ├── processed/            # Cleaned and merged data
│   └── artifacts/            # Generated features (TF-IDF matrices, Embeddings) and config files
│
├── models/                   # Saved trained models (.pkl)
│
├── notebooks/
│   ├── archive/              # Older exploratory notebooks
│   └── lab_experiments.ipynb # Main experimentation notebook
│
├── src/                      # Source code package
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and splitting logic
│   ├── models.py             # Recommender model classes (CF, Content, Hybrid)
│   ├── evaluation.py         # Metrics and evaluation logic
│   └── utils.py              # Helper functions (Tuning loops)
│
├── submissions/              # Generated submission files for Kaggle
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/halim-y/DSML_Kaggle_Competition.git
    cd DSML_Kaggle_Competition
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

Data:
Cleaned and merged data, tf-idf matrix + miniLM embedding matrix

Split:
Based on pct_rank

Model:
Content + Collaborative Filtering = Hybrid model
Inclusion of popularity and novelty features -> à comprendre mdr

Evaluation of final model:
   -> Hit Rate @ 10: 75.8417%
   -> MAP @ 10: 42.3133%
   -> Novelty: 1375.0498%
   -> Coverage: 89.1058%