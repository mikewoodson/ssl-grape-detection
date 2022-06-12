from PIL import ImageDraw
from torchvision import transforms as T
from torchvision.ops import box_iou
import numpy as np
import torch
import pdb

supported_gt = ['bbox', 'dot', 'density']

def draw_gt(dset, gt_type, save_path):
    if gt_type not in supported_gt:
        raise ValueError(
            f"gt_type must be one of {supported_gt}, received {gt_type}"
        )
    for img, target in dset:
        resultImg = T.ToPILImage()(img)
        fname = dset.get_fname(target['image_id'])
        if gt_type == 'bbox':
            annotations = target['boxes']
            draw_fn = draw_boxes
        else:
            annotations = target['dots']
            draw_fn = draw_keypoints
        draw_fn(resultImg, annotations, save_path/f'{gt_type}-{fname}')

def draw_boxes(img, boxes, save_path, box_colors=None):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    width = int(0.000004 * (w * h))
    for idx, box in enumerate(boxes):
        color = (255,255,255)
        if box_colors is not None:
            color = box_colors[idx]
        box = list(np.array(box))
        draw.rectangle(box, outline=color, width=width)

    parent_dir = save_path.parent
    if not parent_dir.exists():
        parent_dir.mkdir()

    img.save(save_path)

def draw_keypoints(img, keypoints, save_path):
    draw = ImageDraw.Draw(img)
    for kp in keypoints:
        kp = list(np.array(kp))
        draw.ellipse([kp[0]-5, kp[1]-5, kp[0]+5, kp[1]+5])

    parent_dir = save_path.parent
    if not parent_dir.exists():
        parent_dir.mkdir()

    img.save(save_path)

def draw_result(img, output, target, confidence, iou_thresh, save_path):
    scores_sorted, indices = torch.sort(output['scores'], descending=True)
    boxes_sorted = output['boxes'][indices]
    top_indices = torch.where(scores_sorted > confidence)
    top_boxes = boxes_sorted[top_indices]
    top_scores = scores_sorted[top_indices]
    target_boxes = target.get('boxes')
    box_colors = None
    if target_boxes is not None:
        box_colors = [(255, 0, 0)] * len(top_boxes)
        gt_matched = [False] * len(target_boxes)
        iou_matrix = box_iou(top_boxes, target['boxes'])

        for row in range(iou_matrix.shape[0]):
            iou = iou_thresh
            match_idx = -1
            for col in range(iou_matrix.shape[1]):
                if iou_matrix[row, col] >= iou and not gt_matched[col]:
                    match_idx = col
                    iou = iou_matrix[row, col]
            if match_idx != -1:
                box_colors[row] = (0, 0, 255)
                gt_matched[match_idx] = True

    draw_boxes(img, top_boxes, save_path, box_colors)
