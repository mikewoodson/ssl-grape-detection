import pdb
import time

import torch
import torchvision
from .utils import *
from pathlib import Path
from torchvision import transforms as T
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .img_utils import draw_result

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, eval_type=None, save_results=False,
             save_dir='untrained'):
    coco_evaluator = None
    if eval_type == 'coco':
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

    model.to(device)
    dataset = data_loader.dataset
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        if eval_type == 'coco':
            coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if save_results:
           save_dir = Path(dataset.root)/Path(save_dir)
           for img, target, output in zip(images, targets, outputs):
              fname = dataset.get_fname(target['image_id'])
              resultImg = T.ToPILImage()(img)
              result_path = save_dir/fname
              draw_result(resultImg, output, target, 0.7, 0.5, result_path)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if eval_type == 'coco':
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    return coco_evaluator
