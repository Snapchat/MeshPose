import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonDetector:
    def __init__(self, min_detection_score=0.8, max_size=640, momentum=0.6, iou_threshold=0.3):
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights,
                                                                          max_size=max_size).to(device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.min_detection_score = min_detection_score
        self.momentum = momentum
        self.iou_threshold = iou_threshold
        self.previous_bboxes = {}
        self.next_id = 0

    def assign_person_ids(self, current_bboxes):
        tracked_bboxes = {}
        if len(self.previous_bboxes) == 0:
            for current_bbox in current_bboxes:
                tracked_bboxes[self.next_id] = current_bbox
                self.next_id += 1
            return  tracked_bboxes
        # Find matched current-previous bboxes.
        previous_bboxes = np.array(list(self.previous_bboxes.values()))
        previous_ids = np.array(list(self.previous_bboxes.keys()))
        iou_scores = torchvision.ops.box_iou(torch.from_numpy(previous_bboxes),
                                             torch.from_numpy(current_bboxes)).numpy()
        row_ind, col_ind = linear_sum_assignment(1 - iou_scores)
        current_ids = np.array([-1] * len(current_bboxes))
        for i, j in zip(row_ind, col_ind):
            if iou_scores[i, j] > self.iou_threshold:
                current_ids[j] = previous_ids[i]
                tracked_bboxes[current_ids[j]] = current_bboxes[j]
        # Add unmatched current bboxes.
        unmatched_bboxes = np.where(current_ids == -1)[0]
        for j in unmatched_bboxes:
            current_ids[j] = self.next_id
            tracked_bboxes[current_ids[j]] = current_bboxes[j]
            self.next_id += 1
        # Remove unmatched previous bboxes.
        ids_to_remove = set(previous_ids) - set(current_ids)
        for id_to_remove in ids_to_remove:
            self.previous_bboxes.pop(id_to_remove)
        return tracked_bboxes

    def apply_momentum_smoothing(self, tracked_bboxes):
        smoothed_bboxes = []
        for track_id, current_bbox in tracked_bboxes.items():
            if track_id in self.previous_bboxes:
                previous_bbox = self.previous_bboxes[track_id]
                smoothed_bbox = self.momentum * previous_bbox + (1 - self.momentum) * current_bbox
                smoothed_bboxes.append(smoothed_bbox)
                self.previous_bboxes[track_id] = smoothed_bbox
            else:
                smoothed_bboxes.append(current_bbox)
                self.previous_bboxes[track_id] = current_bbox
        return smoothed_bboxes

    def __call__(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            detections = self.model(image_tensor)
        boxes = detections[0]['boxes'].cpu().numpy()
        scores = detections[0]['scores'].cpu().numpy()
        labels = detections[0]['labels'].cpu().numpy()

        bboxes = list()
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.min_detection_score and label == 1:
                bboxes.append(box)

        if self.momentum > 0:
            tracked_bboxes = self.assign_person_ids(np.array(bboxes))
            bboxes = self.apply_momentum_smoothing(tracked_bboxes)

        return bboxes
