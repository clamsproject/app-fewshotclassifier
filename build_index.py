import bisect
import json
import logging
import pathlib

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from mmif.utils import video_document_helper as vdh
from transformers import CLIPProcessor, CLIPModel

from utils import video_path_baapb

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def build_faiss_index(df, timeunit='m', index_filepath=None, default_label=None, num_negative_samples=None, save_to_disk=False):
    # Initialize a Faiss index with the correct dimensions
    index_dimension = 512
    index = faiss.IndexFlatIP(index_dimension)
    index_map = {}

    def add_image_to_faiss(image, label):
        # Generate the image embedding using the CLIP model
        with torch.no_grad():
            image = processor.image_processor(image, return_tensors="pt",
                                              device="cuda" if torch.cuda.is_available() else "cpu")
            embedding = model.get_image_features(image["pixel_values"])
        # flatten the embedding
        embedding = embedding.numpy()

        # Add the embedding to the Faiss index
        faiss.normalize_L2(embedding)
        index.add(embedding)

        # add the label to the label map, the key will be the index from the index
        index_map[str(index.ntotal - 1)] = label
    
    def add_image_as_nolabel(framenum):
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        success, frame = cap.read()
        if not success:
            raise ValueError("Unable to extract frame from video segment")
        no_label_frames.append(frame)

    #get list of guids
    guids = df["guid"].unique()
    logging.info(f"Building index for {len(guids)} videos")
    for guid in guids:
        # Get the total duration of the video
        cap = cv2.VideoCapture(video_path_baapb(guid))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_rate = 20  # frames
        buffer = int(5 * fps)  # frames
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cur_df = df[df["guid"] == guid].sort_values(by=['start'])
        ss = list(map(lambda x:vdh.convert(x, timeunit, 'f', fps), cur_df["start"].values))
        es = list(map(lambda x:vdh.convert(x, timeunit, 'f', fps), cur_df["end"].values))
        no_label_frames = []
        cur_frame = 0
        while cur_frame < total_frames:
            sidx = bisect.bisect(ss, cur_frame)
            eidx = bisect.bisect(es, cur_frame)
            if sidx > 0 and eidx == sidx - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                success, frame = cap.read()
                if not success:
                    raise ValueError("Unable to extract frame from video segment")
                label = cur_df.iloc[sidx - 1].get("label", default_label)
                add_image_to_faiss(frame, label)
            else:
                if sidx > len(ss):
                    break
                elif sidx == 0:
                    if cur_frame + buffer < ss[0]:
                        add_image_as_nolabel(cur_frame)
                elif eidx == len(es):
                    if cur_frame - buffer > es[-1]:
                        add_image_as_nolabel(cur_frame)
                elif (cur_frame + buffer) < ss[sidx+1] and (cur_frame - buffer) > es[eidx]:
                    add_image_as_nolabel(cur_frame)

            cur_frame += sample_rate
        if num_negative_samples is None:
            num_negative_samples = len(index_map) // len(set(index_map.values()))
        for no_label_frame_idx in set(np.linspace(0, len(no_label_frames), num_negative_samples, endpoint=False, dtype=int)):
            add_image_to_faiss(no_label_frames[no_label_frame_idx], None)
                
    if save_to_disk:
        # Save the Faiss index to disk
        faiss.write_index(index, index_filepath)
        json.dump(index_map, open(pathlib.Path(index_filepath).with_suffix('.json'), "w"))
    
    return index, index_map

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # TODO (krim @ 9/29/23): add documentation for the format of the csv
    parser.add_argument("--csv_filepath", '-i', type=str, required=True)
    # TODO (krim @ 9/29/23): auto-infer this number from the # of rows in the csv
    parser.add_argument("--num_negative_samples", '-n', type=int)
    parser.add_argument("--default_label", '-l', type=str, default="chyron")
    # TODO (krim @ 9/29/23): auto-fill these names from the "label" argument
    parser.add_argument("--index_filepath", '-o', type=str, required=True)
    parser.add_argument("--timeunit", '-u', type=str, default='ms', help="timeunit in the csv file, one of ['ms', 's', 'f']")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_filepath, delimiter='\t')
    logging.info(f"csv_filepath: {args.csv_filepath}")
    logging.info(f"timeunit_in_csv: {args.timeunit}")
    logging.info(f"index_filepath: {args.index_filepath}")
    logging.info(f"num_negative_samples: {args.num_negative_samples}")
    logging.info(f"default_label: {args.default_label}")
    
    build_faiss_index(df, args.timeunit[0], args.index_filepath, args.default_label, args.num_negative_samples, save_to_disk=True)

