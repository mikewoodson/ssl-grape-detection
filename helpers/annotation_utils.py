import torch
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from enum import Enum

class ConversionType(Enum):
    centerToVert = 1

def convert_bbox_format(boxes: torch.Tensor, conversionType: int) -> torch.Tensor:
    if conversionType > ConversionType.centerToVert.value:
        raise ValueError(
            f"conversionType must be less than" +
            "{ConversionType.centerToVert.value}, received {conversionType}")

    if conversionType == ConversionType.centerToVert.value:
        # convert box annotations from (Cx,Cy,W,H) to (X0,Y0,X1,Y1)
        box_centers = boxes[:, [0, 1, 0, 1]]
        box_wh = 0.5 * boxes[:, [2, 3, 2, 3]]
        box_wh[:, :2] *= -1
        convertedBoxes = box_centers + box_wh
    else:
        raise ValueError

    return convertedBoxes

def bbox_to_tensor(boxes):
    box_list = [[box.x1, box.y1, box.x2, box.y2] for box in boxes]
    boxes_tensor = torch.tensor(box_list, dtype=torch.int32)
    return boxes_tensor

def bbox_to_imgaug(boxes, img_shape):
    boxes_imgaug = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                    for box in boxes]
    boxes = BoundingBoxesOnImage(boxes_imgaug, shape=img_shape)
    return boxes

def kp_to_imgaug(keypoints, img_shape):
    kp_imaug = [Keypoint(x=kp[0], y=kp[1]) for kp in keypoints]
    kp = KeypointsOnImage(kp_imaug, shape=img_shape)
    return kp

def kp_to_tensor(keypoints):
    kp_list = [[kp.x, kp.y] for kp in keypoints]
    kp_tensor = torch.tensor(kp_list, dtype=torch.int32)
    return kp_tensor
