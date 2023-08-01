import argparse
import logging
from typing import Union

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

import json

import cv2
import torch
import faiss
import build_index
from transformers import CLIPProcessor, CLIPModel

CSV_FILEPATH = "path/to/csv/file.csv"
INDEX_FILEPATH = "index/index.faiss"
BUILD = False  # if BUILD get the build from csv file

class Fewshotclassifier(ClamsApp):

    def __init__(self):
        super().__init__()
        if BUILD:
            build_index.add_csv_to_index(CSV_FILEPATH)

        # load the index
        self.index = faiss.read_index(INDEX_FILEPATH)
        self.index_map = json.load(open(INDEX_FILEPATH + "_map" + ".json", "r"))

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def get_label(self, frames, threshold):
        # process the frame with the CLIP model
        with torch.no_grad():
            images = self.processor(images=frames, return_tensors="pt")
            image_features = self.model.get_image_features(images["pixel_values"])

        # Convert to numpy array
        image_features_np = image_features.detach().cpu().numpy()
        # calculate cosine similarity
        faiss.normalize_L2(image_features_np)
        D, I = self.index.search(image_features_np, k=1)

        self.logger.debug(self.index_map[str(I[0][0])], D[0][0])

        # get the labels from self.label_map
        labels_scores = []
        for i, score in zip(I[0], D[0]):
            if score > threshold:
                labels_scores.append((self.index_map[str(i)], score))
                self.logger.debug("=============IN TARGET=============")
            else:
                labels_scores.append((None, None))
        return labels_scores

    def run_targetdetection(self, video_doc, **kwargs):
        sample_ratio = int(kwargs.get("sampleRatio", 10))
        min_duration = int(kwargs.get("minFrameCount", 10))
        threshold = 0.9 if "threshold" not in kwargs else float(kwargs["threshold"])
        batch_size = 10
        cutoff_minutes = int(kwargs.get("cutoffMins"))

        cap = vdh.capture(video_doc)
        counter = 0
        rich_timeframes = []
        active_targets = {}  # keys are labels, values are dicts with "start_frame", "start_seconds", "target_scores"
        while True:
            if counter > 30 * 60 * cutoff_minutes:  # Stop processing after cutoff
                self.logger.debug("stopping at frmae number ", counter)
                break
            frames = []
            frames_counter = []
            for _ in range(batch_size*sample_ratio):
                ret, frame = cap.read()
                if not ret:
                    break
                if counter % sample_ratio == 0:
                    frames.append(frame)
                    frames_counter.append(counter)
                counter += 1
            if not frames:
                break
            # process batch of frames
            labels_scores = self.get_label(frames, threshold)
            self.logger.debug(f"Frames: {frames_counter[0]} - {frames_counter[batch_size-1]}")
            for (detected_label, score), frame_counter in zip(labels_scores, frames_counter):
                if detected_label is not None:  # has any label
                    if detected_label not in active_targets:
                        active_targets[detected_label] = {
                            "start_frame": frame_counter,
                            "start_seconds": cap.get(cv2.CAP_PROP_POS_MSEC),
                            "target_scores": [score],
                        }
                    else:
                        active_targets[detected_label]["target_scores"].append(score)
                else:
                    # process and reset all active_targets
                    for active_label, target_info in active_targets.items():
                        avg_score = sum(target_info["target_scores"]) / len(target_info["target_scores"])
                        if frame_counter - target_info["start_frame"] > min_duration:
                            rich_timeframes.append(
                                {
                                    "start_frame": target_info["start_frame"],
                                    "end_frame": frame_counter,
                                    "start_seconds": target_info["start_seconds"],
                                    "end_seconds": cap.get(cv2.CAP_PROP_POS_MSEC),
                                    "label": active_label,
                                    "score": float(avg_score),
                                }
                            )
                    active_targets = {}  # reset active targets

                # process any remaining active_targets at the end
            if active_targets:
                for active_label, target_info in active_targets.items():
                    avg_score = sum(target_info["target_scores"]) / len(target_info["target_scores"])
                    rich_timeframes.append({
                        "start_frame": target_info["start_frame"],
                        "end_frame": counter,
                        "start_seconds": target_info["start_seconds"],
                        "end_seconds": cap.get(cv2.CAP_PROP_POS_MSEC),
                        "label": active_label,
                        "score": float(avg_score),
                    })
        return rich_timeframes

    def _annotate(self, mmif: Union[str, dict, Mmif], **kwargs) -> Mmif:
        # load file location from mmif
        video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        config = self.get_configuration(**kwargs)
        unit = config["timeUnit"]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, config)
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0].id,
        )
        timeframe_list = self.run_targetdetection(video_doc, **kwargs)
        # add all the timeframes as annotations to the new view
        for timeframe in timeframe_list:
            # skip timeframes that are labeled as "None"
            if timeframe["label"] == "None":
                continue
            timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)

            # if the timeUnit is milliseconds, convert the start and end seconds to milliseconds
            if unit == "milliseconds":
                timeframe_annotation.add_property("start", timeframe["start_seconds"])
                timeframe_annotation.add_property("end", timeframe["end_seconds"])
            # otherwise use the frame number
            else:
                timeframe_annotation.add_property("start", timeframe["start_frame"])
                timeframe_annotation.add_property("end", timeframe["end_frame"])
            timeframe_annotation.add_property("frameType", timeframe["label"]),
            timeframe_annotation.add_property("score", timeframe["score"])
        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = Fewshotclassifier()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
