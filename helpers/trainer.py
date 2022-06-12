import torch
import matplotlib.pyplot as plt
import numpy as np
import functools

from helpers.utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform

from pprint import pprint
from datetime import datetime

import copy
import math
import pdb

def is_distribution(val):
    return issubclass(type(val), Distribution)

def plot_hist(coords, labels, label_coords, dim, fig):
    if dim > 1:
        plot_hist3d(coords[0], coords[1], labels, label_coords, fig)
    else:
        ax = fig.add_subplot()
        ax.hist(coords[0])

def plot_hist3d(x, y, labels, label_coords, fig):
    if x.shape != y.shape:
        raise ValueError(f'Shape mismatch between x ({x.shape}) and y ({y.shape})')
    assert len(labels) == len(label_coords)
    max_y = np.max(y)
    yedges = np.arange(1, max_y+2)
    hist, xedges, yedges = np.histogram2d(x, y, bins=(20,yedges))
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = 0.5 * np.ones_like(zpos)
    dy = 0.1 * np.ones_like(zpos)
    dz = hist.ravel()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    for pos, label in zip(label_coords, labels):
        ax.text(*pos, label)

def concat_coords(params, dim):
    x_coords = []
    y_coords = []
    for idx, param in enumerate(params):
        param_flat = np.asarray(param.detach().cpu().ravel())
        grid = np.meshgrid(param_flat, idx+1)
        x = grid[0].ravel()
        y = grid[1].ravel()
        x_coords.append(x)
        y_coords.append(y)

    x = np.concatenate(x_coords, axis=0)
    if dim == 1:
        y = None
    else:
        y = np.concatenate(y_coords, axis=0)
    return (x, y)

def prep_label_coords(coords):
    x = coords[0]
    y = coords[1]
    label_y_pos = set(y)
    label_x_pos = [max(x)]*len(label_y_pos)
    label_z_pos = [0]*len(label_y_pos)
    label_pos = list(zip(label_x_pos, label_y_pos, label_z_pos))
    return label_pos

def plot_model_activations(activations, name=None, dim=2, fig=None):
    if fig is None:
        fig = plt.figure()

    if name:
        activation = [activations[name]]
    else:
        activation = activations.values()

    coords = concat_coords(activation, dim=dim)
    if dim > 1:
        labels = list(activations.keys())
        label_pos = prep_label_coords(coords)
    else:
        labels = None
        label_pos = None

    plot_hist(coords, labels, label_pos, dim, fig)
    return fig

def plot_model_layers(model, name=None, dim=2, fig=None):
    if fig is None:
        fig = plt.figure()

    if name:
        params = [model.get_parameter(name)]
    else:
        params = model.parameters()

    coords = concat_coords(params, dim=dim)
    if dim > 1:
        labels = [param[1] for param in model.named_parameters()]
        label_pos = prep_label_coords(coords)
    else:
        labels = None
        label_pos = None
    plot_hist(coords, labels, label_pos, dim, fig)
    return fig

class Trainer:
    def __init__(self, model, device, train_dl, optimizer,
            lr_warmup=None, lr_decay=None):
        self._model = model
        self._device = device
        self._train_dl = train_dl
        self._optimizer = optimizer
        self._activations = {}
        self._lr_warmup = lr_warmup
        self._lr_decay = lr_decay
        self._epoch = 0
        self._param_names = None
        self._debug_mode = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, d):
        supported_devices = ['cpu', 'cuda']
        if d not in supported_devices:
            raise ValueError(f"device should be one of {supported_devices}")
        self._device = d

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def lr_warmup(self):
        return self._lr_warmup

    @lr_warmup.setter
    def lr_warmup(self, lr_warmup):
        self._lr_warmup = lr_warmup
    
    @property
    def lr_decay(self):
        return self._lr_decay

    @lr_decay.setter
    def lr_decay(self, lr_decay):
        self._lr_decay = lr_decay

    def get_activation(self, name):
        def hook(model, input, output):
            self._activations[name] = output.detach().cpu()
        return hook

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, mode):
        self._plots = defaultdict(list)
        self._debug_mode = mode

    def register_trainable_params(self):
        self._param_names = [
            p[0] for p in self.model.named_parameters() if p[1].requires_grad
        ]

    def register_forward_hooks(self):
        activations_to_monitor = [
            'backbone.body.layer1',
            'backbone.body.layer2',
            'backbone.body.layer3',
            'backbone.body.layer4',
            'backbone.fpn.inner_blocks.0',
            'backbone.fpn.inner_blocks.1',
            'backbone.fpn.inner_blocks.2',
            'backbone.fpn.inner_blocks.3',
            'backbone.fpn.layer_blocks.0',
            'backbone.fpn.layer_blocks.1',
            'backbone.fpn.layer_blocks.2',
            'backbone.fpn.layer_blocks.3',
            'rpn.head.conv',
            'rpn.head.cls_logits',
            'rpn.head.bbox_pred',
            'roi_heads.box_head.fc6',
            'roi_heads.box_head.fc7',
        ]
        for name, module in self.model.named_modules():
            if name in activations_to_monitor:
                module.register_forward_hook(self.get_activation(name))

    def load_checkpoint(self, path):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Must set model and optimizer before loading"+
                               "from a checkpoint")
        lr = self.optimizer.param_groups[0]['lr']
        w_decay = self.optimizer.param_groups[0]['weight_decay']
        nms_thresh = self.model.roi_heads.nms_thresh

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self._epoch = state['epoch']

        self.model.roi_heads.nms_thresh = nms_thresh
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[0]['weight_decay'] = w_decay

    def save_checkpoint(self, path):
        parent = path.parent
        if not parent.exists():
            parent.mkdir()
        save_dict = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(save_dict, path)

    @property
    def epoch(self):
        return self._epoch

    def train(self):
        if self.debug_mode:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.register_forward_hooks()
            self.register_trainable_params()
        train_metric_logger = MetricLogger(delimiter="  ")
        train_metric_logger.add_meter("lr", SmoothedValue(window_size=1,
                                                         fmt="{value:.10f}"))
        self.model.train()
        self.model.to(self.device)

        header = f'Epoch: [{self.epoch}]'
        batch = 0
        for images, targets in train_metric_logger.log_every(self._train_dl,
                                                             10, header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in targets]

            loss_dict = self.model(images, targets)
            if self.debug_mode:
                pdb.set_trace()

            losses = sum(loss for loss in loss_dict.values())


            if not math.isfinite(losses):
                if self.debug_mode:
                    pdb.set_trace()
                raise RuntimeError(f"Loss is {losses}, stopping training")

            self.optimizer.zero_grad()
            losses.backward()
            if self.debug_mode:
                for param in self.optimizer.param_groups[0]['params']:
                    #l2NormGrad = torch.linalg.norm(param.grad.detach())
                    l2NormWeight = torch.linalg.norm(param.detach())
                    self._plots[batch].append(l2NormWeight)
                    '''
                if batch % 10 == 0:
                    names = list(self._activations.keys())
                    plot_activation = functools.partial(plot_model_activations, self._activations, dim=1)
                    plot_model = functools.partial(plot_model_layers, model=self.model, dim=1)
                    pdb.set_trace()
                    '''
            self.optimizer.step()

            if self.lr_warmup is not None:
                self.lr_warmup.step()

            train_metric_logger.update(loss=losses, **loss_dict)
            train_metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            batch += 1

        if self.debug_mode:
            #plot_model_layers(self.model, fig)
            #plot_model_activations(self._activations, fig)
            #self._plots[self.epoch].append(fig)
            pdb.set_trace()

        if self.lr_decay is not None:
            self.lr_decay.step()

        self._epoch += 1
        loss_dict = {
            loss_key : loss_val.item()
                for loss_key, loss_val in loss_dict.items()
        }
        loss_dict['loss'] = losses.item()
        return loss_dict
