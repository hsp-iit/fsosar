import argparse
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
from torch.optim.lr_scheduler import MultiStepLR
from utils import AverageMeter, setup, load_configs, DataArgs, OpenSetLoss, compute_accuracy, compute_oscr, compute_aupr
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import is_address_in_use
import copy
from models import SAFSAR, STRM, ActionCLIP, MAML
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Remove useless warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--model', type=str, required=True, choices=["STRM", "SAFSAR", "ActionCLIP", "MAML"], help='Model name')
    parser.add_argument('--data', type=str, required=True, choices=["SSv2", "HMDB51", "UCF101", "NTURGBD120", "Diving48"], help='Data name')
    parser.add_argument('--os_loss', type=str, required=True, choices=["softmax", "eos", "objectosphere", "discriminator", "gc"], help='Open set loss')
    return parser.parse_args()


def main(rank, world_size, model_name, data_name, os_loss, port):
    config = load_configs(model_name, data_name)
    config["model_name"] = model_name
    config["data_name"] = data_name
    config["os_loss"] = os_loss
    if config["ddp"]:  # When training more models on more GPU on a single machine, DDP is needed for performance
        setup(rank, world_size, set_seeds=config["eval_only"], port=port)
    # Create directory for saving checkpoints
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{config['shot']}_{model_name}_{os_loss}_{data_name}_{timestamp}"
        checkpoint_dir = os.path.join(config["log_path"], "logs", checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Define training parameters
    lr = config["lr"]
    log_train_after_steps = config["log_train_after_steps"]
    eval_after_steps = config["eval_after_steps"]
    log_wandb = config["log_wandb"]
    if model_name == "STRM":
        if data_name in ["SSv2", "NTURGBD120", "Diving48"]:
            if os_loss == "gc":
                lr = 0.0001
            else:
                lr = 0.001
            disc_weight = 10
        elif data_name in ["HMDB51", "UCF101"]:
            lr = 0.0001
            disc_weight = 10
        elif data_name == "Diving48":
            lr = 0.0001
            disc_weight = 10
    elif model_name == "SAFSAR":
        if config["shot"] == 1:
            if data_name in ["NTURGBD120"]:
                lr = 4e-7
            elif data_name in ["SSv2", "Diving48"]:
                lr = 4e-6
            elif data_name in ["UCF101"]:
                lr = 1e-8
            elif data_name in ["HMDB51"]:
                lr = 1e-7
            disc_weight = 100
        elif config["shot"] == 5:
            lr = 4e-6
            disc_weight = 1000
    elif model_name == "ActionCLIP":
        if config["shot"] == 1:
            if data_name in ["NTURGBD120"]:
                lr = 1e-6
            elif data_name in ["SSv2", "Diving48"]:
                lr = 1e-5
            elif data_name in ["UCF101"]:
                lr = 5e-7
            elif data_name in ["HMDB51"]:
                lr = 5e-7
            disc_weight = 100
        elif config["shot"] == 5:
            lr = 1e-5
            disc_weight = 1000
    elif model_name == "MAML":
        if config["shot"] == 1:
            if data_name in ["NTURGBD120"]:
                lr = 1e-4
            elif data_name in ["SSv2", "Diving48"]:
                lr = 1e-4
            elif data_name in ["UCF101"]:
                lr = 5e-5
            elif data_name in ["HMDB51"]:
                lr = 5e-5
            disc_weight = 100
        elif config["shot"] == 5:
            lr = 1e-4
            disc_weight = 1000
        
    config["eval_after_steps"] = eval_after_steps
    config["lr"] = lr

    # Data
    data_config = copy.deepcopy(config)  # Fix config, since GC may change way later and its recreated later
    def setup_dataloader(train=True):
        videodataset = VideoDataset(DataArgs(data_config), preprocessing=model_name)
        videodataset.train = train
        if config["ddp"]:
            train_sampler = DistributedSampler(videodataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(videodataset, batch_size=1, sampler=train_sampler, num_workers=data_config["num_workers"])
        else:
            dataloader = DataLoader(videodataset, batch_size=1, num_workers=data_config["num_workers"])
        return dataloader, videodataset
    dataloader, videodataset = setup_dataloader()
    config["classes_names"] = videodataset.complex_classes_descriptions
    config["n_train_classes"] = len(set(videodataset.train_split.gt_a_list))
    config["train_unique_classes"] = videodataset.train_split.get_unique_classes()

    # Model
    if os_loss == "gc":  # We add one class for the unknown class, dataset is already declared
        config["way"] += 1
    
    if model_name == "SAFSAR":
        model = SAFSAR(config, disc=os_loss=="discriminator", gc=os_loss=="gc", dp=config["dp"])
    elif model_name == "STRM":
        model = STRM(config, disc=os_loss=="discriminator", gc=os_loss=="gc", dp=config["dp"])
    elif model_name == "ActionCLIP":
        model = ActionCLIP(config, disc=os_loss=="discriminator", gc=os_loss=="gc", dp=config["dp"])
    elif model_name == "MAML":
        model = MAML(config, disc=os_loss=="discriminator", gc=os_loss=="gc", dp=config["dp"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    os_loss_function = OpenSetLoss(os_loss)

    # Set up model
    model.to(rank)
    if config["ddp"]:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model_attributes = model.module
    else:
        model_attributes = model
    model_attributes.set_train()
    model.train()
    if config["eval_only"]:
        import collections
        old_weights = torch.load(config["checkpoint_path"])
        new_weights = collections.OrderedDict()
        for k, v in old_weights.items():
            new_name = k.replace("model.module.", "module.model.")
            if "discriminator" in new_name or "mm_fusion_module" in new_name or "task_specific_learning_module" in new_name or "global_classification_layer" in new_name:
                new_name = "module." + new_name
            new_weights[new_name] = copy.deepcopy(v)
        del old_weights
        model.load_state_dict(new_weights, strict=False)

    # Initialize wandb
    if log_wandb and rank==0:
        wandb.init(project="fsosar", config=config, name=f"{config['host']}_{checkpoint_path}")
        # wandb.watch(model, log="all")

    # Define optimizer and scheduler depending on the model
    model_param = filter(lambda p: p.requires_grad, model.parameters())
    if model_name == "SAFSAR":
        optimizer = torch.optim.Adam(model_param, lr=lr)
        scheduler = None
    elif model_name == "STRM":
        optimizer = torch.optim.SGD(model_param, lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1000000], gamma=0.1)
    elif model_name == "ActionCLIP":
        optimizer = torch.optim.Adam(model_param, lr=lr)
        scheduler = None
    elif model_name == "MAML":
        optimizer = torch.optim.Adam(model_param, lr=lr)
        scheduler = None
    else:
        raise Exception("Wrong model name")

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
    torch.set_grad_enabled(not config["eval_only"])
    training = not config["eval_only"]

    while True:
        # assert dataloader.dataset.train == training
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
            if os_loss != "softmax" or (not training and os_loss == "softmax") or config["eval_only"]:  # at test time, always use unknown set
                while True:  # Ensure that there is at least one unknown class
                    all_images = torch.cat((target_set, unknown_set), 0).reshape(-1, config["seq_len"], *img_shape)
                    all_labels = torch.cat((target_labels, torch.full_like(unknown_labels, -1)), 0)
                    all_unknowns = torch.cat((torch.full_like(target_labels, 0), unknown_labels), 0)
                    t = list(zip(all_images, all_labels, all_unknowns))
                    random.shuffle(t)
                    all_images, all_labels, all_unknowns = zip(*t)
                    # Get only first 5 elements for memory constraints
                    all_images = torch.stack(all_images[:maximum_queries])
                    all_images = all_images.reshape(-1, config["seq_len"], *img_shape)
                    all_labels = torch.stack(all_labels[:maximum_queries])
                    if (all_labels == -1).sum() > 0 and (all_labels != -1).sum() > 0: 
                        break
            else:
                all_images = target_set.reshape(-1, config["seq_len"], *img_shape)[:maximum_queries]
                all_labels = target_labels[:maximum_queries]
                all_unknowns = unknown_labels[:maximum_queries]
            all_images = all_images.cuda()
            all_labels = all_labels.cuda()

            # Forward passs
            logits = model(support_set, support_labels, all_images, batch_class_list=batch_class_list)
            similarity_matrix = logits['similarity_matrix']
            # TODO STRM have 2 similarity matrices, remember to use both for open set loss

            # Compute losses
            # known
            known_indices = all_labels != -1
            if known_indices.sum() > 0:
                if os_loss == "gc":  # Since GC is computed only as known loss, we do this
                    all_labels[all_labels==-1] = config["way"]-1
                known_losses = model_attributes.compute_known_losses(**logits, true_target_labels=None,
                                                                        target_labels=all_labels,
                                                                        support_labels=support_labels,
                                                                        batch_class_list=batch_class_list)
                if os_loss == "gc":
                    if model_name == "SAFSAR":
                        rescale_function = lambda x: (x+1)/2
                    if model_name == "STRM":
                        rescale_function = torch.exp
                    if model_name == "ActionCLIP":
                        rescale_function = lambda x: (x+1)/2
                    if model_name == "MAML":
                        rescale_function = lambda x: (x+1)/2
                else:
                    rescale_function = None
                unknown_losses = os_loss_function.loss(logits, all_labels, similarity_matrix, rescale_function)
                if os_loss == "discriminator":
                    if unknown_losses["os_loss"] is not None:
                        if os_loss == "discriminator":
                            unknown_losses["os_loss"] = unknown_losses["os_loss"]*disc_weight  # this makes safsar disc work
            
            # Visual debug must be called only during evaluation
            if config["visual_debug"] and not training and rank == 0:
                model_attributes.visual_debug(**logits, videodataset=videodataset,
                                                  support_labels=support_labels,
                                                  target_labels=all_labels,
                                                  batch_class_list=batch_class_list,
                                                  support_set=support_set,
                                                  target_set=all_images,
                                                  unknown_labels=all_unknowns)

            # Optimization
            known_losses.update(unknown_losses)
            if training and not config["eval_only"]:
                all_loss = sum([v if v is not None else 0 for k, v in known_losses.items()])
                all_loss.backward()
                if step % optimize_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            # Compute metrics
            if known_indices.sum() > 0:
                if model_name == "STRM":
                    similarity_matrix = similarity_matrix + 0.1*logits["logits_post_pat"]
                # Here we define scaled similarity matrix, that are score normalized in [0, 1] depending on method
                scaled_similarity_matrix = None
                if model_name == "STRM":
                    scaled_similarity_matrix = torch.exp(similarity_matrix)
                    # acc_target = all_labels
                elif model_name in ["SAFSAR", "ActionCLIP", "MAML"]:
                    scaled_similarity_matrix = (similarity_matrix + 1)/2
                    # acc_target = true_target_labels
                if os_loss == "gc":
                    os_target = all_labels!=(config["way"]-1)
                else:
                    os_target = all_labels!=-1
                os_target = os_target.detach().cpu().numpy()
                if os_loss in ["softmax", "eos", "objectosphere"]:  # implcit methods
                    mssm = scaled_similarity_matrix.amax(dim=-1).detach().cpu().numpy()
                    mss = torch.nn.functional.softmax(similarity_matrix, dim=-1).amax(dim=-1).detach().cpu().numpy()
                    mls = similarity_matrix.amax(dim=-1).detach().cpu().numpy()
                    os_sim = similarity_matrix.detach().cpu().numpy()
                    if len(np.unique(os_target)) == 2:  # we need 2 classes to define open-set metrics, softmax train doesn't have them
                        metrics = {"fs_acc": compute_accuracy(similarity_matrix[known_indices], all_labels[known_indices]),
                                "os_auroc_mss": roc_auc_score(os_target, mss),
                                "os_auroc_mls": roc_auc_score(os_target, mls),
                                "os_auroc_mls_scaled": roc_auc_score(os_target, mssm),
                                "os_aupr_mss": compute_aupr(os_target, mss),
                                "os_aupr_mls": compute_aupr(os_target, mls),
                                "os_aupr_mls_scaled": compute_aupr(os_target, mssm),
                                "os_oscr_mss": compute_oscr(all_labels.detach().cpu().numpy(), os_sim, mss),
                                "os_oscr_mls": compute_oscr(all_labels.detach().cpu().numpy(), os_sim, mls),
                                "os_oscr_mls_scaled": compute_oscr(all_labels.detach().cpu().numpy(), os_sim, mssm),
                                "os_acc_mss": ((torch.tensor(mss).cuda()>0.5) == torch.tensor(os_target).cuda()).sum().item()/len(os_target),
                                "os_acc_mls": ((torch.tensor(mls).cuda()>0.5) == torch.tensor(os_target).cuda()).sum().item()/len(os_target),
                                "os_acc_mls_scaled": ((torch.tensor(mssm).cuda()>0.5) == torch.tensor(os_target).cuda()).sum().item()/len(os_target)}
                    else:
                        metrics = {"fs_acc": compute_accuracy(similarity_matrix[known_indices], all_labels[known_indices])}
                else:  # explicit method
                    if os_loss == "discriminator":
                        os_score = logits["disc_prob"].squeeze(1)
                    elif os_loss == "gc":
                        os_score = torch.nn.functional.softmax(similarity_matrix, dim=-1)[:, -1]
                        similarity_matrix = similarity_matrix[:, :-1]
                        os_score = 1-os_score  # this is unknown score, but we work with known score
                        all_labels[all_labels==(config["way"]-1)] = -1  # replace index of unknown class with -1

                    os_score = os_score.detach().cpu().numpy()
                    metrics = {"fs_acc": compute_accuracy(similarity_matrix[known_indices],
                                                          all_labels[known_indices]),
                              "os_auroc": roc_auc_score(os_target, os_score),
                              "os_aupr": compute_aupr(os_target, os_score),
                              "os_oscr": compute_oscr(all_labels.detach().cpu().numpy(), similarity_matrix.detach().cpu().numpy(), os_score),
                              "os_acc": ((torch.tensor(os_score).cuda()>0.5) == torch.tensor(os_target).cuda()).sum().item()/len(os_target)}
            else:
                metrics = {"fs_acc": None, "os_auroc": None}

            additional_metrics = model_attributes.compute_additional_metrics(**logits, support_labels=support_labels,
                                                                                   target_labels=all_labels,
                                                                                   batch_class_list=batch_class_list)
            metrics.update(additional_metrics)
            average_meter.update({**known_losses, **metrics})
        
            # Training logging
            if step % log_train_after_steps == 0 and step > 0 and training:
                train_results = average_meter.average()
                train_results.update(model_attributes.get_debug_data())
                if rank == 0:
                    if log_wandb:
                        wandb.log(train_results)
                    else:
                        print(train_results)

            # Enable evaluation
            if training and ((step % eval_after_steps == 0 and step > 0) or config["eval_only"]):
                if config["ddp"]:
                    dist.barrier()
                if rank == 0:
                    progress_bar.close()
                    progress_bar = tqdm(total=config["n_eval_steps"], desc="Evaluation Progress")
                model_attributes.set_eval()
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
                if config["ddp"]:
                    dist.barrier()
                model_attributes.set_train()
                model.train()
                del dataloader
                del videodataset
                new_dataloader, new_videodataset = setup_dataloader(train=True)
                test_results = average_meter.average()
                if rank == 0:
                    progress_bar.close()
                    progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
                    print(test_results)
                    test_results.update(model_attributes.get_debug_data())
                    if config["log_wandb"]:
                        wandb.log(test_results)
                    # Save the model with test accuracy as the name
                    acc_vip = test_results["test/fs_acc"]
                    if "test/os_acc" in test_results:
                        os_vip = test_results["test/os_acc"]
                        os_metric = "os_acc"
                    elif "test/os_acc_mss" in test_results:
                        os_vip = test_results["test/os_acc_mss"]
                        os_metric = "os_acc_mss"
                    model_path = os.path.join(checkpoint_dir, f"STEPS_{total_step}_ACC_{acc_vip:.4f}_{os_metric}_{os_vip:.4f}.pt")
                    torch.save(model.state_dict(), model_path)
                average_meter = train_meter
                training = True
                torch.set_grad_enabled(True)
                step = 0
                if config["exit_after_eval"]:
                    exit(0)
                else:
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
    # NOTE ddp is useful to isolate 4 models on 4 GPUs on same machine
    config = load_configs(model_name, data_name)
    if config["ddp"]:
        # To deal with possibly multiple training on one machine
        ports = [12355, 12356, 12357, 12358, 12359]
        for port in ports:
            if not is_address_in_use('localhost', port):
                print("chosen", port)
                os.environ['MASTER_PORT'] = str(port)
                break
        torch.multiprocessing.spawn(main, args=(world_size, model_name, data_name, os_loss, port), nprocs=world_size, join=True)
    else:
        main(0, world_size, model_name, data_name, os_loss, None)
