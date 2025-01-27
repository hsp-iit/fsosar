import argparse
from safsar import SAFSAR
from videoloader import VideoDataset
import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
import random
import importlib
from torch.optim.lr_scheduler import MultiStepLR
from utils import AverageMeter, setup, load_configs, DataArgs, OpenSetLoss, compute_accuracy, compute_auroc
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Remove useless warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--model', type=str, required=True, choices=["STRM", "SAFSAR"], help='Model name')
    parser.add_argument('--data', type=str, required=True, choices=["SSv2", "HMBD51", "UCF101", "NTURGBD120", "Diving44"], help='Data name')
    parser.add_argument('--os_loss', type=str, required=True, choices=["softmax", "posunk", "eos", "mos"], help='Open set loss')
    return parser.parse_args()


def main(rank, world_size, model_name, data_name, os_loss):
    config = load_configs(model_name, data_name)
    setup(rank, world_size, set_seeds=config["eval_only"])

    # Create directory for saving checkpoints
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{model_name}_{os_loss}_{data_name}_{timestamp}"
        checkpoint_dir = os.path.join("logs", checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Define training parameters
    lr = config["lr"]
    log_train_after_steps = config["log_train_after_steps"]
    eval_after_steps = config["eval_after_steps"]
    log_wandb = config["log_wandb"]

    # Data
    def setup_dataloader(train=True):
        videodataset = VideoDataset(DataArgs(config), preprocessing=model_name)
        videodataset.train = train
        train_sampler = DistributedSampler(videodataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(videodataset, batch_size=1, sampler=train_sampler, num_workers=config["num_workers"])
        return dataloader, videodataset
    dataloader, videodataset = setup_dataloader()
    config["classes_names"] = videodataset.class_folders
    config["n_train_classes"] = len(set(videodataset.train_split.gt_a_list))
    config["train_unique_classes"] = videodataset.train_split.get_unique_classes()

    # Model
    model = getattr(importlib.import_module(model_name.lower()), model_name)(config)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.module.set_train()
    model.train()
    if config["eval_only"]:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

    # Initialize wandb
    if log_wandb and rank==0:
        wandb.init(project="fsosar", config=config, name=f"{config['host']}_{checkpoint_path}")
        wandb.watch(model, log="all")

    # Define optimizer and scheduler depending on the model
    if model_name == "SAFSAR":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    elif model_name == "STRM":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1000000], gamma=0.1)
    else:
        raise Exception("Wrong model name")

    # Define open-set loss
    os_loss_function = OpenSetLoss(os_loss)

    # Loop variables
    train_meter = AverageMeter("train/")
    test_meter = AverageMeter("test/")
    average_meter = train_meter
    if rank == 0:
        progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
    step = 0
    total_step = 0
    training = True
    optimize_every = config["optimize_every"]
    maximum_queries = config["maximum_queries"]

    while True:
        assert dataloader.dataset.train == training
        for elem in dataloader:
            # Data preparation
            support_set = elem["support_set"].squeeze(0).cuda()
            target_set = elem["target_set"].squeeze(0)  # .cuda()
            target_labels = elem["target_labels"].squeeze(0).long()  # .cuda()
            support_labels = elem['support_labels'].squeeze(0).long().cuda()
            batch_class_list = elem['batch_class_list'].squeeze(0).long().cuda()
            # real_target_labels = elem["real_target_labels"].squeeze(0).long()
            unknown_set = elem["unknown_set"].squeeze(0)
            unknown_labels = elem["unknown_labels"].squeeze(0).long()

            # Put together known and unknown
            img_shape = target_set.shape[-3:]
            if os_loss != "None" or (not training and os_loss == "None"):  # at test time, always use unknown set
                while True:  # Ensure that there is at least one unknown class
                    all_images = torch.cat((target_set, unknown_set), 0).reshape(-1, config["seq_len"], *img_shape)
                    all_labels = torch.cat((target_labels, torch.full_like(unknown_labels, -1)), 0)
                    t = list(zip(all_images, all_labels))
                    random.shuffle(t)
                    all_images, all_labels = zip(*t)
                    # Get only first 5 elements for memory constraints
                    all_images = torch.stack(all_images[:maximum_queries])
                    all_images = all_images.reshape(-1, config["seq_len"], *img_shape)
                    all_labels = torch.stack(all_labels[:maximum_queries])
                    if (all_labels == -1).sum() > 0 and (all_labels != -1).sum() > 0: 
                        break
            else:
                all_images = target_set.reshape(-1, config["seq_len"], *img_shape)[:maximum_queries]
                all_labels = target_labels[:maximum_queries]
            all_images = all_images.cuda()
            all_labels = all_labels.cuda()

            # Forward passs
            similarity_matrix = None  # Suppress warnings
            logits = model(support_set, support_labels, all_images, batch_class_list=batch_class_list)
            if 'logits' in logits:
                similarity_matrix = logits['logits']
            elif 'similarity_matrix' in logits:
                similarity_matrix = logits['similarity_matrix']
            # TODO STRM have 2 similarity matrices, remember to use both for open set loss

            # Compute losses
            # TODO strm uses target_labels oreded by support_labels, while safsar uses the true target labels
            # known
            known_indices = all_labels != -1
            if known_indices.sum() > 0:
                unknown_indices = all_labels == -1
                true_target_labels = torch.argsort(support_labels)[all_labels]
                true_target_labels[unknown_indices] = -1
                known_losses = model.module.compute_known_losses(**logits, true_target_labels=true_target_labels,
                                                                        target_labels=all_labels,
                                                                        support_labels=support_labels,
                                                                        batch_class_list=batch_class_list)
                # unknown
                if os_loss != "None":
                    # SAFSAR WANTS true_target_labels
                    # STRM wants all_labels
                    acc_target = true_target_labels if model_name == "SAFSAR" else all_labels
                    unknown_losses = os_loss_function(similarity_matrix, acc_target)
                else:
                    unknown_losses = {}
            
            # Visual debug must be called only during evaluation
            if config["visual_debug"] and not training and rank == 0:
                model.module.visual_debug(**logits, videodataset=videodataset,
                                                  support_labels=support_labels,
                                                  target_labels=all_labels,
                                                  batch_class_list=batch_class_list,
                                                  support_set=support_set,
                                                  target_set=all_images)

            # Optimization
            if training and step % optimize_every == 0:
                known_losses.update(unknown_losses)
                all_loss = sum([v if v is not None else 0 for k, v in known_losses.items()])
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

            # Compute metrics
            if known_indices.sum() > 0:
                # NOTE: strm sorts the support classes, while safsar does not
                # So for safsar we need to use true_target_labels, while for
                # STRM we need to use plain target labels
                acc_target = true_target_labels if model_name == "SAFSAR" else all_labels
                metrics = {"fs_acc": compute_accuracy(similarity_matrix[known_indices], acc_target[known_indices]),
                        "os_auroc": compute_auroc(similarity_matrix, true_target_labels)}
            else:
                metrics = {"fs_acc": None, "os_auroc": None}

            additional_metrics = model.module.compute_additional_metrics(**logits, support_labels=support_labels,
                                                                                   target_labels=all_labels,
                                                                                   batch_class_list=batch_class_list)
            metrics.update(additional_metrics)
            average_meter.update({**known_losses, **metrics})
        
            # Training logging
            if step % log_train_after_steps == 0 and step > 0 and training:
                train_results = average_meter.average()
                train_results.update(model.module.get_debug_data())
                if rank == 0:
                    if log_wandb:
                        wandb.log(train_results)
                    else:
                        print(train_results)

            # Enable evaluation
            if training and ((step % eval_after_steps == 0 and step > 0) or config["eval_only"]):
                dist.barrier()
                if rank == 0:
                    progress_bar.close()
                    progress_bar = tqdm(total=config["n_eval_steps"], desc="Evaluation Progress")
                model.module.set_eval()
                model.eval()
                del dataloader
                del videodataset
                new_dataloader, new_videodataset = setup_dataloader(train=False)
                average_meter.average()
                average_meter = test_meter
                training = False
                torch.set_grad_enabled(False)
                step = 0
                break  # Break out of the for loop to restart with the new dataloader

            # Disable evaluation
            if not training and step == config["n_eval_steps"]:
                dist.barrier()
                model.module.set_train()
                model.train()
                del dataloader
                del videodataset
                new_dataloader, new_videodataset = setup_dataloader(train=True)
                test_results = average_meter.average()
                if rank == 0:
                    progress_bar.close()
                    progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
                    print(test_results)
                    test_results.update(model.module.get_debug_data())
                    if config["log_wandb"]:
                        wandb.log(test_results)
                    # Save the model with test accuracy as the name
                    acc_vip = test_results["test/fs_acc"]
                    auroc_vip = test_results["test/os_auroc"]
                    model_path = os.path.join(checkpoint_dir, f"STEPS_{total_step}_ACC_{acc_vip:.4f}_AUROC_{auroc_vip:.4f}.pt")
                    torch.save(model.state_dict(), model_path)
                average_meter = train_meter
                training = True
                torch.set_grad_enabled(True)
                step = 0
                break  # Break out of the for loop to restart with the new dataloader

            if rank == 0:
                progress_bar.update(1)
            step += 1
            total_step += 1

        dataloader = new_dataloader
        videodataset = new_videodataset


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    data_name = args.data
    os_loss = args.os_loss
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, model_name, data_name, os_loss), nprocs=world_size, join=True)
