import pathlib
from urllib.request import urlopen
from urllib.parse import quote
import os 
import json
import sys

def media_path_dict(root="/data/clams/"):
    all_paths = list(pathlib.Path(root).glob("**/*.mp4"))
    full_filename_dict = {p.name: str(p.absolute()) for p in all_paths}
    guid_dict = {p.name.split('.')[0]: str(p.absolute()) for p in all_paths}
    return {**full_filename_dict, **guid_dict}

def add_label_column(csv_filepath: str, label_column: str='label', default_label: str='chyron'):
    # load a csv file and check if the label column exists, if not, add it with the default label
    import pandas as pd
    df = pd.read_csv(csv_filepath)
    if label_column not in df.columns:
        df[label_column] = default_label
    df.to_csv(csv_filepath, index=False)

def video_path_baapb(guid):
    RESOLVER_ADDRESS = os.environ['BAAPB_RESOLVER_ADDRESS']
    url = 'http://' + RESOLVER_ADDRESS + '/searchapi'
    qstr = "&".join(map(lambda x: f"{x[0]}={quote(x[1])}", {'file': 'video', 'guid': guid}.items()))
    res = urlopen("?".join((url,qstr)))
    return json.loads(res.read().decode('utf-8'))[0]
