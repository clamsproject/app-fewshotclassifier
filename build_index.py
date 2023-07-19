import json
import logging
import random

import cv2
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from utils import media_path_dict
import tqdm

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a function to extract features from a video segment and add them to a Faiss index
def add_video_segment_to_index(guid, start_time, end_time, label, index, label_map):
    full_filepath = media_path_dict().get(guid)
    if full_filepath is None:
        raise ValueError(f"No file path found for guid {guid}")
    # Load the video and extract a frame from the specified time segment
    cap = cv2.VideoCapture(full_filepath)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    success, frame = cap.read()
    if not success:
        raise ValueError("Unable to extract frame from video segment")

    # todo - update this to process frames in batches
    # Generate the image embedding using the CLIP model
    with torch.no_grad():
        image = processor.image_processor(frame, return_tensors="pt")
        embedding = model.get_image_features(image["pixel_values"])

    # flatten the embedding
    embedding = embedding.numpy()

    # Add the embedding to the Faiss index
    faiss.normalize_L2(embedding)
    index.add(embedding)

    # add the label to the label map, the key will be the index from the index
    label_map[str(index.ntotal - 1)] = label

def add_csv_to_index(csv_filepath):
    df = pd.read_csv(csv_filepath)
    build_faiss_index(df)

def build_faiss_index(df, index_filepath=None, index_map_filepath=None, default_label=None, num_negative_samples=100, save_to_disk=False):
    # Initialize a Faiss index with the correct dimensions
    index_dimension = 512
    index = faiss.IndexFlatIP(index_dimension)
    index_map = {}

    #get list of guids
    guids = df["guid"].unique()
    logging.info(f"Building index for {len(guids)} videos")
    for guid in guids:
        # Get the total duration of the video
        cap = cv2.VideoCapture(media_path_dict()[guid])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_sec = total_frames / fps

        # print the type of row
        # Generate the list of annotated segments
        annotated_segments = [(float(row["start"]), float(row["end"])) for _, row in df[df["guid"] == guid].iterrows()]

        # Add a buffer around the annotated segments
        buffer = 5  # in seconds
        buffered_segments = [(max(0, start - buffer), min(end + buffer, total_sec)) for start, end in annotated_segments]

        # Potential negative sample timestamps
        potential_negatives = [i for i in range(int(total_sec)) if
                            not any(start <= i <= end for start, end in buffered_segments)]

        # Randomly select timestamps for negative samples
        negative_samples = random.sample(potential_negatives, min(num_negative_samples, len(potential_negatives)))

        logging.info(f"Adding {len(negative_samples)} negative samples for video {guid}")
        # Add negative samples to index
        for sample in negative_samples:
            add_video_segment_to_index(guid, sample, sample, None, index, index_map)

    logging.info(f"Adding {len(df)} labeled examples")
    # Process each row of the DataFrame and add the embeddings to the Faiss index
    for _, row in tqdm.tqdm(df.iterrows()):
        guid = row["guid"]
        label = row.get("label", default_label)
        add_video_segment_to_index(guid, float(row["start"]), float(row["end"]), label, index, index_map)

    if save_to_disk:
        # Save the Faiss index to disk
        faiss.write_index(index, index_filepath)
        json.dump(index_map, open(index_map_filepath, "w"))
    
    return index, index_map

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_filepath", type=str, required=True)
    parser.add_argument("--index_filepath", type=str, required=True)
    parser.add_argument("--index_map_filepath", type=str, required=True)
    parser.add_argument("--num_negative_samples", type=int, default=100)
    parser.add_argument("--default_label", type=str, default="chyron")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_filepath)
    logging.info(f"csv_filepath: {args.csv_filepath}")
    logging.info(f"index_filepath: {args.index_filepath}")
    logging.info(f"index_map_filepath: {args.index_map_filepath}")
    logging.info(f"num_negative_samples: {args.num_negative_samples}")
    logging.info(f"default_label: {args.default_label}")
    
    build_faiss_index(df, args.index_filepath, args.index_map_filepath, args.default_label, args.num_negative_samples, save_to_disk=True)

