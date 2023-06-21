import argparse
from typing import Union

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

from config import config
import json

import cv2
import torch
import faiss
import build_index
from transformers import CLIPProcessor, CLIPModel


class Clip(ClamsApp):

    def __init__(self):
        super().__init__()
        index_filepath = config["index_filepath"]
        # if the value of "build" is true, get the csv filepath and call build_index
        if config["build"]:
            csv_filepath = config["csv_filepath"]
            build_index.add_csv_to_index(
                csv_filepath, index_filepath, index_filepath + "_map"
            )

        # load the index
        self.index = faiss.read_index(index_filepath)
        self.index_map = json.load(open(index_filepath + "_map" + ".json", "r"))

        self.model = CLIPModel.from_pretrained(config["model_name"])
        self.processor = CLIPProcessor.from_pretrained(config["model_name"])

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def get_label(self, frames, threshold):
        # process the frame with the CLIP model
        with torch.no_grad():
            #images = self.processor.image_processor(frames, return_tensors="pt")
            images = self.processor(images=frames, return_tensors="pt")
            image_features = self.model.get_image_features(images["pixel_values"])

        print("convert to numpy")
        # Convert to numpy array
        image_features_np = image_features.detach().cpu().numpy()
        print(image_features_np.shape)
        print("calculate similarity")
        # calculate cosine similarity
        faiss.normalize_L2(image_features_np)   # Not getting past this
        print("normalized")
        D, I = self.index.search(image_features_np, k=1)
        print("index searched")

        print(self.index_map[str(I[0][0])], D[0][0])

        # get the labels from self.label_map
        labels_scores = []
        for i, score in zip(I[0], D[0]):
            if score > threshold:
                labels_scores.append((self.index_map[str(i)], score))
            else:
                labels_scores.append((None, None))
        print("return labels_scores")
        return labels_scores

    def run_chyrondetection(self, video_filename, **kwargs):
        sample_ratio = int(kwargs.get("sampleRatio", 10))
        min_duration = int(kwargs.get("minFrameCount", 10))
        threshold = 0.5 if "threshold" not in kwargs else float(kwargs["threshold"])
        batch_size = 2
        cutoff_minutes = 1
        print("running chyrondetection")

        cap = cv2.VideoCapture(video_filename)
        counter = 0
        chyrons = []
        in_chyron = False
        start_frame = None
        start_seconds = None
        score = 0
        while True:
            if counter > 30 * 60 * cutoff_minutes:  # Stop processing after cutoff
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
            print("processing")
            labels_scores = self.get_label(frames, threshold)
            for label, score, frame_counter in zip(labels_scores, frames_counter):
                result = label == "chyron"
                if result:  # has chyron
                    if not in_chyron:
                        in_chyron = True
                        start_frame = frame_counter
                        start_seconds = cap.get(cv2.CAP_PROP_POS_MSEC)
                else:
                    if in_chyron:
                        in_chyron = False
                        if frame_counter - start_frame > min_duration:
                            chyrons.append(
                                {
                                    "start_frame": start_frame,
                                    "end_frame": frame_counter,
                                    "start_seconds": start_seconds,
                                    "end_seconds": cap.get(cv2.CAP_PROP_POS_MSEC),
                                    "label": label,
                                    "score": float(score),
                                }
                            )
        # If  last frames are in_chyron
        if in_chyron:
            chyrons.append({
                "start_frame": start_frame,
                "end_frame": counter,
                "start_seconds": start_seconds,
                "end_seconds": cap.get(cv2.CAP_PROP_POS_MSEC),
                "label": label,
                "score": float(score),
            })
        return chyrons

    def _annotate(self, mmif: Union[str, dict, Mmif], **kwargs) -> Mmif:
        # load file location from mmif
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument)
        config = self.get_configuration(**kwargs)
        unit = config["timeUnit"]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, config)
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0].id,
        )
        timeframe_list = self.run_chyrondetection(video_filename, **kwargs)
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
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = Clip()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
