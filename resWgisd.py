import pdb
import sys
import traceback
import argparse
import functools

import torch
import torch.multiprocessing

from pathlib import Path
from pprint import pprint

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining
from torch.utils.data import DataLoader
from helpers.engine import evaluate
from helpers.model_utils import create_model
from helpers.train_utils import trainval, create_trainer, save_val_results, load_val_results
from helpers.data_utils import get_dataset, data_to_annotations
from helpers.utils import collate_fn
from helpers.img_utils import draw_gt

checkpoint_dir = Path('checkpoints')
validation_dir = Path('validation')
project_dir = Path.cwd()
hdrive = Path.home()/'hdrive'/'ray_results'

checkpoint_names = {
    'byol' : Path('byol_weights.pth'),
    'fpn' : Path('fpn_weights.pth'),
    'resnet' : Path('resnet_weights.pth'),
    'untrained' : Path('untrained_weights.pth'),
    'byolsmoke' : Path('byol_weights_smoke.pth'),
    'fpnsmoke' : Path('fpn_weights_smoke.pth'),
    'resnetsmoke' : Path('resnet_weights_smoke.pth'),
    'untrainedsmoke' : Path('untrained_weights_smoke.pth'),
}

def main():
    parser = argparse.ArgumentParser(description=('Evaluate pretrained'
                                                  'models on Wgisd dataset.'))
    parser.add_argument('-m', '--model-type', type=str, default='fpn',
                        help=('name of the pretrained model to load and '
                              'evaluate (byol | resnet | fpn | untrained)'))
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-t', '--type', type=str, default='test',
                        choices=['test', 'tune', 'train'])
    parser.add_argument('-s', '--smoke', action='store_true')
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-D', '--dataset', type=str, default='wgisd',
                        choices=['wgisd', 'cr2'])
    parser.add_argument('--data-dir', type=str, default=project_dir/'data')
    parser.add_argument('--ray-dir', type=str, default=hdrive)
    parser.add_argument('-l', '--layers', type=int, default=5)
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda') if torch.cuda.is_available()\
        else torch.device('cpu')

    # cr2 dataset does not support 'train' or 'val' splits
    dataset = 'wgisd'
    train_dset = get_dataset(dataset=dataset, split='train')
    mean = train_dset.mean
    stddev = train_dset.stddev
    val_dset = get_dataset(dataset=dataset, split='val')

    train_data_loader = DataLoader(
        train_dset, batch_size=2, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_data_loader = DataLoader(
        val_dset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    if args.debug is True:
        pdb.set_trace()

    val_result_file = str(validation_dir / f'layer{args.layers}'/ f'val_results_{args.model_type}')
    val_result_file = Path(val_result_file + 'smoke' if args.smoke else val_result_file)

    if args.type == 'tune':
        pb_config = {
            'lr' : tune.loguniform(1e-6, 5e-2),
            'weight_decay' : tune.loguniform(1e-3, 0.5),
            'box_nms_thresh' : tune.uniform(0.4, 0.6),
        }

        reporter = CLIReporter(metric_columns=['avg_prec', 'training_iteration'])

        num_epochs = 3 if args.smoke is True else 15
        num_samples = 5 if args.smoke is True else 15
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=1 if args.smoke else 3,
            hyperparam_mutations=pb_config
        )
        trainable = functools.partial(
            trainval,
            model_type=args.model_type,
            train_dl=train_data_loader,
            val_dl=val_data_loader,
            device=device,
            image_mean=mean,
            image_std=stddev,
            train_layers=args.layers,
            checkpoint=True)
        result = tune.run(
            trainable,
            resources_per_trial={
                'cpu': 4,
                'gpu': 1 if device == torch.device('cuda') else 0,
            },
            config=pb_config,
            metric='avg_prec',
            mode='max',
            checkpoint_score_attr='avg_prec',
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            raise_on_failed_trial=False,
            stop={'training_iteration': num_epochs},
            resume=args.resume,
            name=args.name if args.resume else None,
            local_dir=args.ray_dir
        )
        save_val_results(result, val_result_file)
        print("Finished cross validation")
        pdb.set_trace()
        return
    else:
        try:
            results = load_val_results(val_result_file)
            validation_checkpoint = results['checkpoint_dir']
            best_config = results['config']
        except FileNotFoundError as e:
            print(f'file {val_result_file} not found')
            raise e
            validation_checkpoint = None
            best_config = {
                'lr' : 0.00003,
                'weight_decay' : 0.02,
                'box_nms_thresh' : 0.5
            }
        model_kwargs = {
            'model_type' : args.model_type,
            'box_nms_thresh' : best_config['box_nms_thresh'],
            'num_classes' : 2,
            'image_mean' : mean,
            'image_std' : stddev,
            'train_layers' : args.layers,
        }
        cp_name = args.model_type + 'smoke' if args.smoke else args.model_type
        cp_file = checkpoint_dir/f'layer{args.layers}'/checkpoint_names[cp_name]
        model = create_model(**model_kwargs)
        model.to(device)
        if args.type == 'train':
            best_mAP = 0
            num_epochs = 1 if args.smoke else 15
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=best_config['lr'],
                weight_decay=best_config['weight_decay']
            )
            trainer = create_trainer(
                model,
                optimizer,
                train_data_loader,
                device,
                True,
                False
            )
            trainer.debug_mode = args.debug
            if validation_checkpoint is not None:
                trainer.load_checkpoint(validation_checkpoint + 'checkpoint')

            for epoch in range(num_epochs):
                pprint(f'EPOCH {epoch}:')
                trainer.train()
                coco_evaluator = evaluate(trainer.model,
                                          val_data_loader,
                                          device=device,
                                          eval_type='coco')
                mAP = coco_evaluator.coco_eval['bbox'].stats[0]
                if mAP >= best_mAP:
                    best_mAP = mAP
                    trainer.save_checkpoint(cp_file)
        elif args.type == 'test':
            test_dset = get_dataset(dataset=args.dataset, split='test')
            annotation_type = data_to_annotations[args.dataset]
            draw_gt(test_dset, annotation_type, test_dset.root/'gt')
            test_data_loader = DataLoader(
                test_dset, batch_size=1, shuffle=True,
                num_workers=4, pin_memory=True, collate_fn=collate_fn
            )
            state = torch.load(cp_file,
                               map_location=device)
            model.load_state_dict(state['model'])
            eval_type = 'coco' if args.dataset == 'wgisd' else None
            evaluate(model,
                     test_data_loader,
                     device=device,
                     eval_type=eval_type,
                     save_results=True,
                     save_dir=str(Path(args.model_type)/f'layer{args.layers}'))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
