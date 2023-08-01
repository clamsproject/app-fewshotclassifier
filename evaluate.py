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
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure


import csv
import build_index
from transformers import CLIPProcessor, CLIPModel

from utils import media_path_dict

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_label(frame, threshold, index, index_map):
    # process the frame with the CLIP model
    with torch.no_grad():
        image = processor.image_processor(frame, return_tensors="pt")
        image_features = model.get_image_features(image["pixel_values"]).numpy()

    # calculate cosine similarity
    faiss.normalize_L2(image_features)
    D, I = index.search(image_features, k=15)
    return D[0], I[0]


def run_chyrondetection(video_filename, index, index_map, fold_number=0, **kwargs):
    sample_ratio = int(kwargs.get("sampleRatio", 5))
    min_duration = int(kwargs.get("minFrameCount", 1))
    threshold = 0.9 if "threshold" not in kwargs else float(kwargs["threshold"])
    stop_seconds = 60 * 10 if "stopSeconds" not in kwargs else int(kwargs["stopSeconds"])

    print (f"{video_filename=}, {threshold=}, {sample_ratio=}, {min_duration=}, {stop_seconds=}")

    full_filepath = media_path_dict()[video_filename]
    if not os.path.exists(full_filepath):
        print(f"File {full_filepath} does not exist")
        return []

    output_file = f"results/fold_{fold_number}/{video_filename}_results.csv"
    if not os.path.exists(f"results/fold_{fold_number}"):
        os.makedirs(f"results/fold_{fold_number}")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'seconds', 'score_1', 'label_1', 'score_2', 'label_2', 'score_3', 'label_3', 'score_4', 'label_4', 'score_5', 'label_5', 'score_6', 'label_6', 'score_7', 'label_7', 'score_8', 'label_8', 'score_9', 'label_9', 'score_10', 'label_10']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        cap = cv2.VideoCapture(full_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
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
            if counter > stop_seconds * fps:
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
                # print (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                previous_score = score
                scores, labels = get_label(frame, threshold, index, index_map)
                label, score = labels[0], scores[0]
                # Write the results to the CSV file
                row_data = {
                    'filename': video_filename,
                    'seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                }
                for i, (d, idx) in enumerate(zip(scores, labels)):
                    if i >= 10:
                        break
                    row_data[f'score_{i + 1}'] = float(d)
                    row_data[f'label_{i + 1}'] = index_map[str(idx)]
                writer.writerow(row_data)
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

def evaluate(chyrons, golds, **kwargs):
    stop_seconds = 60 * 10 if "stopSeconds" not in kwargs else int(kwargs["stopSeconds"])
    # for each chyron (row) in the dataset, add a segment to a pyannote timeline for the ground truth
    ground_truth = Annotation()
    for row in golds:
        print (row)
        ground_truth[Segment(float(row['start']), float(row['end']))] = "labeled"
    
    # for each chyron (row) in the dataset, add a segment to a pyannote timeline for the prediction
    prediction = Annotation()
    for chyron in chyrons:
        print (chyron)
        prediction[Segment(float(chyron['start_seconds']), float(chyron['end_seconds']))] = "labeled"

    # uem timeline first 10 minutes only
    uem = Timeline(segments=[Segment(0, stop_seconds)])

    detection_precision_recall_fmeasure = DetectionPrecisionRecallFMeasure(collar=2)
    detection_precision_recall_fmeasure(ground_truth, prediction, uem=uem, detailed=True)
    print (detection_precision_recall_fmeasure)
    print (detection_precision_recall_fmeasure.report(display=False))
    
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
    parser.add_argument("--csv", type=str, default="transformed_data.csv")
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
            print(f"Fold {fold}")
            train_df = pd.DataFrame()
            # for each video in the training set, get the embeddings and add them to the index
            for i, filename in enumerate(dataset_dict.keys()):
                if i in train_index:
                    video_df = pd.DataFrame(dataset_dict[filename])
                    train_df = pd.concat([train_df, video_df], ignore_index=True)
            fold_index_filename = f"index_{fold}.faiss"
            fold_index_map_filename = f"slate_index_{fold}.json"
            index, index_map = build_or_load(train_df, fold_index_filename, fold_index_map_filename, use_cache=True)
            # for each video in the testing set, get the embeddings and compare them to the embeddings in the index
            for i, filename in enumerate(dataset_dict.keys()):
                if i in test_index:
                    chyrons = run_chyrondetection(filename, index=index, index_map=index_map, fold_number=fold, stopSeconds=60*10)
                    evaluate(chyrons, dataset_dict[filename])
    