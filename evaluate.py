import itertools
import os
import sys
import numpy as np
import pandas as pd
import faiss
import argparse
import json
import torch
import cv2
from config import config
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionErrorRate

import csv
import build_index
from transformers import CLIPProcessor, CLIPModel

from utils import media_path_dict

model = CLIPModel.from_pretrained(config["model_name"])
processor = CLIPProcessor.from_pretrained(config["model_name"])

def get_label(frame, threshold, index, index_map):
    # process the frame with the CLIP model
    with torch.no_grad():
        image = processor.image_processor(frame, return_tensors="pt")
        image_features = model.get_image_features(image["pixel_values"]).numpy()

    # calculate cosine similarity
    faiss.normalize_L2(image_features)
    D, I = index.search(image_features, k=15)
    # iterate over the results and get the label for each
    for d, i in zip(D[0], I[0]):
        if d > threshold:
            if index_map[str(i)] == "chyron":
                return index_map[str(i)], d
    return None, None

def run_chyrondetection(video_filename, index, index_map, **kwargs):
    sample_ratio = int(kwargs.get("sampleRatio", 30))
    min_duration = int(kwargs.get("minFrameCount", 1))
    threshold = 0.1 if "threshold" not in kwargs else float(kwargs["threshold"])
    print ("running chyron detection with threshold", threshold)
    full_filepath = media_path_dict()[video_filename]
    if not os.path.exists(full_filepath):
        print(f"File {full_filepath} does not exist")
        return []
    cap = cv2.VideoCapture(full_filepath)
    counter = 0
    chyrons = []
    in_chyron = False
    start_frame = None
    start_seconds = None
    score = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter > 30 * 60 * 10:  # 1 hour
            if in_chyron:
                if counter - start_frame > min_duration:
                    chyrons.append(
                        {
                            "start_frame": start_frame,
                            "end_frame": counter,
                            "start_seconds": start_seconds/1000,
                            "end_seconds": cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                            "label": label,
                            "score": float(score),
                        }
                    )
            break
        if counter % sample_ratio == 0:
            previous_score = score
            label, score = get_label(frame, threshold, index, index_map)
            # print (label, score)
            result = label == "chyron"
            if result:  # has chyron
                if not in_chyron:
                    in_chyron = True
                    start_frame = counter
                    start_seconds = cap.get(cv2.CAP_PROP_POS_MSEC)
            else:
                if in_chyron:
                    in_chyron = False
                    if counter - start_frame > min_duration:
                        chyrons.append(
                            {
                                "start_frame": start_frame,
                                "end_frame": counter,
                                "start_seconds": start_seconds/1000,
                                "end_seconds": cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                                "label": "chyron",
                                "score": float(previous_score),
                            }
                        )
        counter += 1
    return chyrons

def evaluate(chyrons, golds):
    # for each chyron (row) in the dataset, add a segment to a pyannote timeline for the ground truth
    ground_truth = Annotation()
    for row in golds:
        ground_truth[Segment(float(row['start']), float(row['end']))] = "chyron"
    
    # for each chyron (row) in the dataset, add a segment to a pyannote timeline for the prediction
    prediction = Annotation()
    for chyron in chyrons:
        prediction[Segment(float(chyron['start_seconds']), float(chyron['end_seconds']))] = "chyron"

    # calculate the detection error rate
    detection_error_rate = DetectionErrorRate(collar=.5)
    detection_error_rate(ground_truth, prediction, detailed=True)
    print (detection_error_rate)
    print (detection_error_rate.report(display=False))
        
def build_or_load(df, index_filepath, index_map_filepath, use_cache=True):
    if os.path.exists(index_filepath) and os.path.exists(index_map_filepath) and use_cache:
        print (f"Loading index from {index_filepath}")
        index = faiss.read_index(index_filepath)
        index_map = json.load(open(index_map_filepath, "r"))
    else:
        index, index_map = build_index.build_faiss_index(df, index_filepath=index_filepath, index_map_filepath=index_map_filepath, num_negative_samples=10, default_label="chyron", save_to_disk=True)
    return index, index_map

if __name__ == "__main__":
    # argparse, number of folds, path to csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--csv", type=str, default="chyron_gold.csv")
    args = parser.parse_args()

    # load dataset using dictreader
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)

        # group from reader by filename
        dataset_dict = {
            filename: list(rows) for filename, rows in itertools.groupby(reader, key=lambda x: x['guid'])
        }

        kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        kf.get_n_splits(dataset_dict.keys())

        # for each fold, divide the dataset into training and testing sets
        for fold, (train_index, test_index) in enumerate(kf.split(list(dataset_dict.keys()))):
            if fold == 0:
                continue
            print (f"Fold {fold}")
            train_df = pd.DataFrame()
            # for each video in the training set, get the embeddings and add them to the index
            for i, filename in enumerate(dataset_dict.keys()):
                if i in train_index:
                    video_df = pd.DataFrame(dataset_dict[filename])
                    train_df = pd.concat([train_df, video_df], ignore_index=True)
            fold_index_filename = f"index_{fold}.faiss"
            fold_index_map_filename = f"index_{fold}.json"
            index, index_map = build_or_load(train_df, fold_index_filename, fold_index_map_filename, use_cache=True)
            # for each video in the testing set, get the embeddings and compare them to the embeddings in the index
            for i, filename in enumerate(dataset_dict.keys()):
                if i in test_index:
                    chyrons = run_chyrondetection(filename, index=index, index_map=index_map)
                    evaluate(chyrons, dataset_dict[filename])
    