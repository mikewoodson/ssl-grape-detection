import json
import torch.optim
from .trainer import Trainer
from .engine import evaluate
from ray import tune
from .model_utils import create_model
from pathlib import Path

def save_val_results(result, val_file):
    val_results = {}
    best_config = result.best_config
    best_checkpoint_dir = result.best_checkpoint
    val_results['checkpoint_dir'] = best_checkpoint_dir
    val_results['config'] = best_config
    parent = val_file.parent
    if not parent.exists():
        parent.mkdir()

    with open(val_file, 'w') as f:
        f.write(json.dumps(val_results))

def load_val_results(val_file):
    with open(val_file) as f:
        results = json.loads(f.read())
    return results

def create_trainer(model, optimizer, dataloader, device, decay, warmup):
    lr_decay = None
    lr_warmup = None
    if decay is True:
        lr_decay = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    if warmup is True:
        raise ValueError('LR warmup unsupported')

    return Trainer(model, device, dataloader, optimizer, lr_warmup, lr_decay)

def trainval(config, checkpoint, train_dl, val_dl, device,
        checkpoint_dir=None, **kwargs):
    model_kwargs = {
        'model_type' : kwargs['model_type'],
        'box_nms_thresh' : config['box_nms_thresh'],
        'num_classes' : 2,
        'image_mean' : kwargs['image_mean'],
        'image_std' : kwargs['image_std'],
        'train_layers' : kwargs['train_layers']
    }
    model = create_model(**model_kwargs)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    trainer = create_trainer(model, optimizer, train_dl, device, False, False)

    if checkpoint_dir and checkpoint is True:
        checkpoint = Path(checkpoint_dir)/'checkpoint'
        trainer.load_checkpoint(checkpoint)

    while True:
        loss_dict = trainer.train()
        coco_evaluator = evaluate(trainer.model,
                                  val_dl,
                                  device=device,
                                  eval_type='coco')
        mAP = coco_evaluator.coco_eval['bbox'].stats[0]
        epoch = trainer.epoch

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            checkpoint = Path(checkpoint_dir)/'checkpoint'
            trainer.save_checkpoint(checkpoint)
        tune.report(avg_prec=mAP, **loss_dict)

