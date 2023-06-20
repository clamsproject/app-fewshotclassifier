import json
import cv2
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from utils import media_path_dict
import tqdm

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Define a function to extract features from a video segment and add them to a Faiss index
def add_video_segment_to_index(video_filepath, start_time, end_time, label, index, label_map):
    full_filepath = media_path_dict()[video_filepath]
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
    label_map[index.ntotal - 1] = label


# Define a function to process a CSV file containing video segment information
def add_csv_to_index(csv_filepath, index_filepath, index_map_filepath, include_unlabeled=True):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_filepath)

    # Initialize a Faiss index with the correct dimensions
    index_dimension = 512
    index = faiss.IndexFlatIP(index_dimension)
    index_map = {}

    # Process each row of the DataFrame and add the embeddings to the Faiss index
    for _, row in tqdm.tqdm(df.iterrows()):
        label = row.get("label", None)

        if label is None and not include_unlabeled:
            continue

        add_video_segment_to_index(row["filename"], row["start"], row["end"], label, index, index_map)

    # Save the Faiss index to disk
    faiss.write_index(index, index_filepath)
    json.dump(index_map, open(index_map_filepath + ".json", "w"))
