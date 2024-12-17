"""
RCNN-YOLO Adapter (rcnn_wrapper.py)
============================

This module provides an adapter to make Faster R-CNN models compatible with YOLO's interface.
It wraps R-CNN models to match YOLO's prediction format and validation pipeline.

Classes
-------
ModelWrapper
    Wraps an R-CNN model to provide YOLO-compatible outputs.
    - Converts R-CNN's list[dict] predictions to YOLO's (batch, detections, 6) tensor format
    - Maps COCO-91 class indices to COCO-80 format
    - Filters background class predictions

RCNN_YOLO
    YOLO-compatible interface for R-CNN models.
    Inherits from ultralytics.YOLO to maintain API compatibility.


Notes
-----
Requires coco91to80.json file for class index mapping from COCO-91 and COCO-80 formats.
Requires coco_id_to_name.json file for class {id: name} mapping.
Inference time predictions have overhead due to conversion and filtering but overhead is negligible.
Expects a batch size of 1 only.
"""

import torch
from ultralytics import YOLO
import json

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, device='cuda:0', stride=32.0):
        super().__init__()
        self.model = model
        self.stride = torch.tensor([stride], device=device)
        # load jsons
        with open('coco_id_to_name.json') as f:
            self.names = json.load(f)
        self.names = {int(k): v for k, v in self.names.items()}
        with open('coco91to80.json') as f:
            self.class_map = json.load(f)
        self.nc = 80 # yolo requirement

    def forward(self, x, **kwargs):
        raw_output = self.model(x)
        
        # standard r-cnn output is list[dict]
        if self.training:
            return raw_output
            
        # convert dicts to yolo format (tensor)
        batch_predictions = []
        for pred in raw_output:        # loop over batch (right now batch size == 1)
            boxes = pred['boxes']      # (N, 4) [x1, y1, x2, y2]
            scores = pred['scores']    # (N,)
            labels = pred['labels']    # (N,)

            # filter background (in R-CNN, background is a class with id=0)
            # in YOLO, background is not a class
            valid_indices = labels > 0
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = labels[valid_indices]
            
            # convert from COCO91 to COCO80 and decrement by 1
            # (don't forget to cast to str for json key lookup)
            mapped_labels = torch.tensor(
                [self.class_map[str(l.item())]-1 for l in labels], 
                device=labels.device
            )

            # combine into (N, 6) format
            # [x1, y1, x2, y2, score, class] where N = num detections
            N = boxes.shape[0]
            yolo_pred = torch.zeros((N, 6), device=boxes.device)
            yolo_pred[:, :4] = boxes  # bbox coordinates
            yolo_pred[:, 4] = scores  # confidence scores
            yolo_pred[:, 5] = mapped_labels
            
            batch_predictions.append(yolo_pred)
        
        # stack into batch dimension
        if batch_predictions:
            predictions = torch.stack(batch_predictions)
        else:
            predictions = torch.zeros((1, 0, 6), device=x.device)
            
        return predictions

    # yolo requirement
    def fuse(self, **kwargs):
        # no layers to fuse here
        return self

class RCNN_YOLO(YOLO):
    def __init__(self, model, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        # wrap the model
        self.model = ModelWrapper(model)

    def predict(self, images):
        with torch.inference_mode():
            predictions = self.model(images)
        return predictions