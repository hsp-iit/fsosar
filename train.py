from videoloader import VideoDataset
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
import random
from torch.optim.lr_scheduler import MultiStepLR
from utils import AverageMeter, setup, load_configs, DataArgs
import importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Remove useless warnings


data_name = "SSv2"
model_name = "STRM"

# Import right model
model_module = importlib.import_module(model_name.lower())
model_class = getattr(model_module, model_name)


def main(rank, world_size):
    setup(rank, world_size)
    config = load_configs(model_name, data_name)

    # Create directory for saving checkpoints
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{model_name}_{data_name}_{timestamp}"
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
        # Change preprocessing to custom one
        # videodataset.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # videodataset.transform["train"] = lambda x: custom_transform(x, videodataset.processor)
        # videodataset.transform["test"] = lambda x: custom_transform(x, videodataset.processor)
        # Use DistributedSampler for the dataset
        train_sampler = DistributedSampler(videodataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(videodataset, batch_size=1, sampler=train_sampler, num_workers=config["num_workers"])
        return dataloader, videodataset
    dataloader, videodataset = setup_dataloader()
    config["classes_names"] = videodataset.class_folders
    config["n_train_classes"] = len(set(videodataset.train_split.gt_a_list))
    config["train_unique_classes"] = videodataset.train_split.get_unique_classes()

    # Model
    model = model_class(config)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.module.set_train()
    model.train()
    if config["eval_only"]:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

    # Initialize wandb
    if log_wandb and rank==0:
        wandb.init(project="fsosar", config=config)
        wandb.watch(model, log="all")

    # Define optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Define scheduler
    if config["scheduler"]:
        scheduler = MultiStepLR(optimizer, milestones=[1000000], gamma=0.1)

    # Loop variables
    train_meter = AverageMeter("train/")
    test_meter = AverageMeter("test/")
    average_meter = train_meter
    if rank == 0:
        progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
    step = 0
    training = True

    while True:
        print("Going with dataset.train = ", dataloader.dataset.train)
        print("dataloader has ", len(dataloader))
        for elem in dataloader:

            # Data preparation
            support_set = elem["support_set"].squeeze(0).cuda()
            target_set = elem["target_set"].squeeze(0).cuda()
            target_labels = elem["target_labels"].squeeze(0).long().cuda()
            support_labels = elem['support_labels'].squeeze(0).long().cuda()
            batch_class_list = elem['batch_class_list'].squeeze(0).long().cuda()
            # real_target_labels = elem["real_target_labels"].squeeze(0).long()
            unknown_set = elem["unknown_set"].squeeze(0).cuda()
            unknown_labels = elem["unknown_labels"].squeeze(0).long().cuda()

            # Put together known and unknown
            if config["open_set"]:
                if model_name == "SAFSAR":
                    img_shape = (-1, config["seq_len"], config["img_size"], 3, config["img_size"])
                elif model_name == "STRM":
                    img_shape = (-1, config["seq_len"], 3, config["img_size"], config["img_size"])
                all_images = torch.cat((target_set, unknown_set), 0).reshape(img_shape)
                all_labels = torch.cat((target_labels, torch.full_like(unknown_labels, -1)), 0)
                t = list(zip(all_images, all_labels))
                random.shuffle(t)
                all_images, all_labels = zip(*t)
                # Get only first 5 elements for memory constraints
                all_images = torch.stack(all_images[:5])
                all_images = all_images.reshape(img_shape)
                all_labels = torch.stack(all_labels[:5])
            else:
                all_images = target_set
                all_labels = target_labels

            # Forward passs
            logits = model(support_set, support_labels, all_images, batch_class_list=batch_class_list,
                                                                    precomputed_context_features=None)

            # Compute known and unknown losses
            losses = model.module.compute_loss(**logits, support_labels=support_labels,
                                                  target_labels=all_labels,
                                                  batch_class_list=batch_class_list)
            
            # Optimization
            if training:
                model.module.optimize(**losses, optimizer=optimizer)
                if config["scheduler"]:
                    scheduler.step()

            # Compute metrics
            metrics = model.module.compute_metrics(**logits, support_labels=support_labels,
                                                      target_labels=all_labels,
                                                      batch_class_list=batch_class_list)

            average_meter.update({**losses, **metrics})
        
            # Training logging
            if step % log_train_after_steps == 0 and step > 0 and training:
                train_results = average_meter.average()
                train_results.update(model.module.get_debug_data())
                if log_wandb and rank==0:
                    # print(train_results)
                    wandb.log(train_results)

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
                    # print(test_results)
                    test_results.update(model.module.get_debug_data())
                    wandb.log(test_results)
                    # Save the model with test accuracy as the name
                    acc_vip = test_results["test/fs_acc"]
                    model_path = os.path.join(checkpoint_dir, f"model_{acc_vip:.4f}.pt")
                    torch.save(model.state_dict(), model_path)
                average_meter = train_meter
                training = True
                torch.set_grad_enabled(True)
                step = 0
                break  # Break out of the for loop to restart with the new dataloader

            if rank == 0:
                progress_bar.update(1)
            step += 1

        dataloader = new_dataloader
        videodataset = new_videodataset


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)