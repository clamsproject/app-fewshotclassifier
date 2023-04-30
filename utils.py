import pathlib

def media_path_dict(root="/data/clams/wgbh"):
    all_paths = pathlib.Path(root).glob("**/*.mp4")
    return {p.name : str(p.absolute()) for p in all_paths}

def add_label_column(csv_filepath: str):
    # load a csv file and add a column label with the value "chyron"
    import pandas as pd
    df = pd.read_csv(csv_filepath)
    df["label"] = "chyron"
    df.to_csv(csv_filepath, index=False)
