import argparse
import json
import logging
from pathlib import Path as P
from typing import Union

import cv2
import faiss
import torch
from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes, Document
from mmif.utils import video_document_helper as vdh
from transformers import CLIPProcessor, CLIPModel


class Fewshotclassifier(ClamsApp):

    def __init__(self):
        super().__init__()

        # model is shared among all instances of the app
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def get_label(self, frames, threshold, finetuning, finetuning_map):
        # process the frame with the CLIP model
        with torch.no_grad():
            images = self.processor(images=frames, return_tensors="pt", device="cuda" if torch.cuda.is_available() else "cpu")
            image_features = self.model.get_image_features(images["pixel_values"])

        # Convert to numpy array
        image_features_np = image_features.detach().cpu().numpy()
        # calculate cosine similarity
        faiss.normalize_L2(image_features_np)
        D, I = finetuning.search(image_features_np, k=1)

        # get the labels from finetuning map
        labels_scores = []
        for i, score in zip(I, D):
            if score[0] > threshold:
                labels_scores.append((finetuning_map[str(i[0])], score[0]))
            else:
                labels_scores.append((None, None))
        return labels_scores

    def run_targetdetection(self, vd: Document, **conf):
        batch_size = 100

        # load the fine-tuned index
        self.logger.debug(f"frame_type_to_detect: {conf['finetunedFrameType']}")
        index_fpath = P(__file__).parent / f'{conf["finetunedFrameType"]}.faiss'
        index = faiss.read_index(str(index_fpath))
        index_map = json.load(open(index_fpath.with_suffix(".json")))
        
        cap = vdh.capture(vd)

        if conf['finetunedFrameType'] == 'credits':
            # For credits, we only need to sample the last 10 minutes
            self.logger.debug("sampling last 10 minutes")
            frames_to_test = vdh.sample_frames(
                int(vd.get_property('frameCount')) - int(vdh.convert(600, 's', 'f', vd.get_property('fps'))),
                int(vd.get_property('frameCount')), conf['sampleRatio'])
        else:
            # sample all frame numbers
            frames_to_test = vdh.sample_frames(0, 30 * 60 * 60, conf['sampleRatio'])  # 1-hour video

        # then extract numpy array, label arrays in batches while extracting
        labels_scores = []
        extracted = []
        for cur_frame in frames_to_test:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame - 1)
            ret, frame = cap.read()
            if not ret:
                break
            extracted.append(frame)
            if len(extracted) == batch_size:
                self.logger.debug(f"processing roughly {vdh.convert(cur_frame, 'f', 's', vd.get_property('fps'))} secs")
                labels_from_a_batch = self.get_label(extracted, conf['threshold'], index, index_map)
                labels_scores.extend(labels_from_a_batch)
                extracted = []
        if len(extracted) > 0:
            labels_from_last_batch = self.get_label(extracted, conf['threshold'], index, index_map)
            labels_scores.extend(labels_from_last_batch)

        rich_timeframes = []
        start_frame = 0
        # for now, the app doesn't support multi-label classification, hence the detected_label is always the same
        in_frame = False
        scores = []
        self.logger.debug(f"input length: {len(frames_to_test)} / output length: {len(labels_scores)}")
        cur_frame = 0
        for (detected_label, score), cur_frame in zip(labels_scores, frames_to_test):
            self.logger.debug(f"{detected_label}, {score}, at {cur_frame}")
            if detected_label is not None:  # has any label
                if not in_frame:
                    in_frame = True
                    start_frame = cur_frame
                    scores = [score]
                else:
                    scores.append(score)
            else:
                if in_frame:
                    in_frame = False
                    avg_score = sum(scores) / len(scores)
                    if cur_frame - start_frame > conf['minFrameCount']:
                        rich_timeframes.append((start_frame, cur_frame,
                                                conf['finetunedFrameType'],
                                                float(avg_score)))
        if in_frame and cur_frame > start_frame:
            avg_score = sum(scores) / len(scores)
            rich_timeframes.append((start_frame, cur_frame,
                                    conf['finetunedFrameType'],
                                    float(avg_score)))
        return rich_timeframes

    def _annotate(self, mmif: Union[str, dict, Mmif], **kwargs) -> Mmif:
        
        # load file location from mmif
        vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not vds:
            # no video document found
            return mmif
        vd = vds[0]
        config = self.get_configuration(**kwargs)
        unit = config["timeUnit"]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, kwargs)
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=vd.id,
        )
        for s, e, l, score in self.run_targetdetection(vd, **config):
            self.logger.debug(f"annotating TF: {s} - {e} ({l}@{score})")
            # skip timeframes that are labeled as "None"
            if l is None:
                continue
            timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", vdh.convert(s, 'f', unit, vd.get_property('fps')))
            timeframe_annotation.add_property("end", vdh.convert(e, 'f', unit, vd.get_property('fps')))
            timeframe_annotation.add_property("frameType", l),
            timeframe_annotation.add_property("score", score)
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
