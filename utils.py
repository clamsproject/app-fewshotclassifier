import pathlib

def media_path_dict(root="/llc_data/clams"):
    all_paths = pathlib.Path(root).glob("**/*.mp4")
    return {p.stem.split('.')[0]: str(p.absolute()) for p in all_paths}

def add_label_column(csv_filepath: str, label_column: str='label', default_label: str='chyron'):
    # load a csv file and check if the label column exists, if not, add it with the default label
    import pandas as pd
    df = pd.read_csv(csv_filepath)
    if label_column not in df.columns:
        df[label_column] = default_label
    df.to_csv(csv_filepath, index=False)
