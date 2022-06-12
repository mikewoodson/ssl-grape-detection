import torchvision.models.detection as detection
import torch
from pathlib import Path

from typing import Callable, Dict, Optional, List, Union
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import *
from torchvision.models import resnet50
from torchvision.ops import misc
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, ExtraFPNBlock
from torchvision.models.detection.anchor_utils import AnchorGenerator

def _resnet_fpn_extractor(
    backbone: resnet.ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ["layer4", "layer3",
                       "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f"layer{k}": str(v)
                     for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 *
                        2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks)

def create_model(model_type, num_classes, train_layers, **kwargs):
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    if model_type == 'resnet':
        resnet = resnet50(pretrained=True, norm_layer=misc.FrozenBatchNorm2d)
        fpn = _resnet_fpn_extractor(resnet, trainable_layers=train_layers)
        model = detection.FasterRCNN(fpn, num_classes,
                                     rpn_anchor_generator=anchor_generator,
                                     **kwargs)
    elif model_type == 'fpn':
        model = detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            trainable_backbone_layers=train_layers,
            rpn_anchor_generator=anchor_generator, **kwargs)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          num_classes)
    elif model_type == 'byol':
        resnet = resnet50(norm_layer=misc.FrozenBatchNorm2d)
        del resnet.fc
        state_dict = torch.load(str(Path.home()/Path('projects', 'thesis', 'ssl-transfer', 'models', 'byol.pth')))
        resnet.load_state_dict(state_dict)
        fpn = _resnet_fpn_extractor(resnet, trainable_layers=train_layers)
        model = detection.FasterRCNN(fpn, num_classes,
                                     rpn_anchor_generator=anchor_generator,
                                     **kwargs)
    elif model_type == 'untrained':
        model = detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=False,
            trainable_backbone_layers=train_layers,
            rpn_anchor_generator=anchor_generator, **kwargs)
    else:
        raise ValueError
    return model
