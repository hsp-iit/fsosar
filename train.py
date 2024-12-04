from safsar import SAFSAR
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
from utils import AverageMeter, setup, load_configs, DataArgs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Remove useless warnings


data_name = "SSv2"
model_name = "SAFSAR"


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
        videodataset = VideoDataset(DataArgs(config))
        videodataset.train = train
        # Change preprocessing to custom one
        # videodataset.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # videodataset.transform["train"] = lambda x: custom_transform(x, videodataset.processor)
        # videodataset.transform["test"] = lambda x: custom_transform(x, videodataset.processor)
        # Use DistributedSampler for the dataset
        train_sampler = DistributedSampler(videodataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(videodataset, batch_size=1, sampler=train_sampler, num_workers=4)
        return iter(dataloader), videodataset
    dataloader, videodataset = setup_dataloader()
    config["classes_names"] = videodataset.class_folders
    config["n_train_classes"] = len(set(videodataset.train_split.gt_a_list))
    config["train_unique_classes"] = videodataset.train_split.get_unique_classes()

    # Model
    model = SAFSAR(config)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()
    if config["eval_only"]:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

    # Initialize wandb
    if log_wandb and rank==0:
        wandb.init(project="fsosar", config=config)
        wandb.watch(model, log="all")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loop variables
    train_meter = AverageMeter("train/")
    test_meter = AverageMeter("test/")
    average_meter = train_meter
    if rank == 0:
        progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
    step = 0
    training = True

    while True:
        elem = next(dataloader)

        # Data preparation
        support_set = elem["support_set"].squeeze(0)
        target_set = elem["target_set"].squeeze(0)
        target_labels = elem["target_labels"].squeeze(0).long()
        support_labels = elem['support_labels'].squeeze(0).long()
        batch_class_list = elem['batch_class_list'].squeeze(0).long()
        # real_target_labels = elem["real_target_labels"].squeeze(0).long()

        # Forward passs
        logits = model(support_set, support_labels, target_set, batch_class_list)

        # Compute loss
        losses = model.module.compute_loss(**logits, support_labels=support_labels,
                                              target_labels=target_labels,
                                              batch_class_list=batch_class_list)
        
        # Optimization
        if training:
            model.module.optimize(**losses, optimizer=optimizer)

        # Compute metrics
        metrics = model.module.compute_metrics(**logits, support_labels=support_labels,
                                                  target_labels=target_labels,
                                                  batch_class_list=batch_class_list)

        average_meter.update({**losses, **metrics})
    
        # Training logging
        if step % log_train_after_steps == 0 and step > 0 and training:
            train_results = average_meter.average()
            if log_wandb and rank==0:
                # print(train_results)
                wandb.log(train_results)

        # Enable evaluation
        if training and ((step % eval_after_steps == 0 and step > 0) or config["eval_only"]):
            print("ENABLE EVALUATIION")
            dist.barrier()
            if rank == 0:
                progress_bar.close()
                progress_bar = tqdm(total=config["n_eval_steps"], desc="Evaluation Progress")
            model.eval()
            dataloader, dataset = setup_dataloader(train=False)
            # dataset.train = False
            average_meter.average()
            average_meter = test_meter
            training = False
            torch.set_grad_enabled(False)
            step = 0
        # Disable evaluation
        if not training and step == config["n_eval_steps"]:
            print("DISABLE EVALUATION")
            dist.barrier()
            model.train()
            dataloader, dataset = setup_dataloader(train=True)
            # dataset.train = True
            test_results = average_meter.average()
            if rank == 0:
                progress_bar.close()
                progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
                # print(test_results)
                wandb.log(test_results)
                # Save the model with test accuracy as the name
                acc_vip = test_results["test/fs_acc"]
                model_path = os.path.join(checkpoint_dir, f"model_{acc_vip:.4f}.pt")
                torch.save(model.state_dict(), model_path)
            average_meter = train_meter
            training = True
            torch.set_grad_enabled(True)
            step = 0

        if rank == 0:
            progress_bar.update(1)
        step += 1

    progress_bar.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)