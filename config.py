'''  
  - sample_csv: "/home/kmlynch/projects/active-learning-experiments/gbh_al/data/chyron_gold.csv"
  - numpy_file: "features.npy"
  - threshold: 0.5
  - model_name: openai/clip-vit-base-patch32'''

config = {
    "csv_filepath": "/home/kmlynch/projects/active-learning-experiments/gbh_al/data/chyron_gold.csv",
    "numpy_file": "features.npy",
    "threshold": 0.5,
    "model_name": "openai/clip-vit-base-patch32",
    "build": False,
    "index_filepath":"index.faiss",
  }
