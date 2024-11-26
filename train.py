from safsar import SAFSAR
from videoloader import VideoDataset, SSv2
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import cv2
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
# Add import for saving the model
import os
from datetime import datetime
import json

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_config():
    deploy_server = "iit.local" in os.getcwd()
    config_path = "configs/safsar/server_config.json" if deploy_server else "configs/safsar/local_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(rank, world_size):
    setup(rank, world_size)

    # Load configuration
    config = load_config()

    # Create directory for saving checkpoints
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("logs", timestamp)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Define training parameters
    lr = config["lr"]
    alpha = config["alpha"]
    log_train_after_steps = config["log_train_after_steps"]
    eval_after_steps = config["eval_after_steps"]
    log_wandb = config["log_wandb"]
    use_l2_loss = config["use_l2_loss"]

    # Load data
    dataset = SSv2()
    dataset.shot = config["shot"]
    dataset.seq_len = config["seq_len"]
    dataset.query_per_class = config["query_per_class"]
    dataset.path = config["path"]
    dataset.n_eval_steps = config["n_eval_steps"]
    videodataset = VideoDataset(dataset)
    # Use DistributedSampler for the dataset
    train_sampler = DistributedSampler(videodataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(videodataset, batch_size=1, sampler=train_sampler)

    # Get features of class names with BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(rank)
    bert_model.eval()
    class_names = videodataset.class_folders
    class_name_embeddings = []
    for class_name in class_names:
        plain_class_name = class_name.replace('_', ' ')
        inputs = tokenizer(plain_class_name, return_tensors="pt").to(rank)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        class_name_embeddings.append(outputs.last_hidden_state.squeeze(0))
    bert_model = None
    tokenizer = None
    n_train_classes = len(set(videodataset.train_split.gt_a_list))

    # Initialize model
    model = SAFSAR(processor_name="MCG-NJU/videomae-base-finetuned-kinetics",
                model_name="MCG-NJU/videomae-base-finetuned-kinetics",
                n_train_classes=n_train_classes,
                use_l2_loss=use_l2_loss)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    # Initialize wandb
    if log_wandb and rank==0:
        wandb.init(project="fsosar", log=config_path)
        wandb.watch(model, log="all")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def compute_accuracy(logits, labels):
        _, preds = torch.max(logits, 1)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)

    # Loop variables
    fs_train_accuracies = []
    global_query_train_accuracies = []
    global_support_train_accuracies = []
    l1_train_losses = []
    l2_train_losses = []

    train_progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
    step = 0
    for elem in dataloader:
        # Data preparation
        support_set = elem["support_set"].squeeze(0)
        target_set = elem["target_set"].squeeze(0)
        target_labels = elem["target_labels"].squeeze(0).long()
        support_labels = elem['support_labels'].squeeze(0).long()
        batch_class_list = elem['batch_class_list'].squeeze(0).long()
        real_target_labels = elem["real_target_labels"].squeeze(0).long()

        # Visualize support NOTE for debug use
        # support_set_flat_labels = [videodataset.class_folders[int(x.item())] for x in batch_class_list[support_labels]]
        # support_set_flat = support_set.reshape(dataset.way, dataset.shot, dataset.seq_len, 3, 224, 224)
        # counter = 0
        # for k in support_set_flat:
        #     for n in k:
        #         print(support_set_flat_labels[counter])
        #         for i in n:
        #             cv2.imshow("image", i.permute(1, 2, 0).numpy())
        #             cv2.waitKey(0)
        #         counter += 1
        # Visualize queries
        # target_set_flat_labels = [videodataset.class_folders[int(x.item())] for x in real_target_labels]
        # target_set_flat = target_set.reshape(dataset.way, dataset.query_per_class, dataset.seq_len, 3, 224, 224)
        # counter = 0
        # for k in target_set_flat:
        #     print(target_set_flat_labels[counter])
        #     for n in k:
        #         for i in n:
        #             cv2.imshow("image", i.permute(1, 2, 0).numpy())
        #             cv2.waitKey(0)
        #     counter += 1

        # Forward passs
        logits, support_global_scores, query_global_scores = model(support_set, support_labels, target_set, target_labels, class_name_embeddings, batch_class_list, dataset)
        
        # Compute L1 loss
        # NO TARGET LABELS!
        # ordering doesn't matter, we need to check where target labels is equal to support labels
        true_target_labels = torch.argsort(support_labels)[target_labels].to(rank)
        l1_loss = torch.nn.functional.cross_entropy(logits, true_target_labels)

        # Compute L2 loss
        if use_l2_loss:
            unique_classes = videodataset.train_split.get_unique_classes()
            global_support_labels = torch.tensor([unique_classes.index(x) for x in batch_class_list[support_labels]]).to(rank)
            global_query_labels = torch.tensor([unique_classes.index(x) for x in real_target_labels]).to(rank)
            # Make support label one hot to apply them correctly
            l2_loss_support = torch.nn.functional.cross_entropy(support_global_scores.reshape(-1, len(unique_classes)),
                                                                global_support_labels.repeat(dataset.way))
            l2_loss_query = torch.nn.functional.cross_entropy(query_global_scores, global_query_labels)
            l2_loss = l2_loss_support + l2_loss_query
        else:
            l2_loss = torch.FloatTensor([0])

        # Optimization
        optimizer.zero_grad()
        if use_l2_loss:
            (l1_loss + alpha*l2_loss).backward()
        else:
            l1_loss.backward()
        optimizer.step()
    
        # Logging
        fs_train_accuracy = compute_accuracy(logits, true_target_labels.long().to(rank))
        fs_train_accuracies.append(fs_train_accuracy)
        if use_l2_loss:
            global_query_train_accuracy = compute_accuracy(query_global_scores, global_query_labels)
            global_support_train_accuracy = compute_accuracy(support_global_scores.reshape(-1, n_train_classes), global_support_labels.repeat(dataset.way))
        else:
            global_query_train_accuracy = 0
            global_support_train_accuracy = 0
        global_query_train_accuracies.append(global_query_train_accuracy)
        global_support_train_accuracies.append(global_support_train_accuracy)
        l1_train_losses.append(l1_loss.item())
        l2_train_losses.append(l2_loss.item())
        if step % log_train_after_steps == 0 and step > 0:
            avg_l1_train_loss = sum(l1_train_losses) / len(l1_train_losses)
            avg_l2_train_loss = sum(l2_train_losses) / len(l2_train_losses)
            avg_fs_train_accuracy = sum(fs_train_accuracies) / len(fs_train_accuracies)
            print(f"Avg L1 Train Loss: {avg_l1_train_loss}, Avg L2 Train Loss: {avg_l2_train_loss*alpha}, Avg FS Train Accuracy: {avg_fs_train_accuracy}, Avg Global Query Train Accuracy: {sum(global_query_train_accuracies) / len(global_query_train_accuracies)}, Avg Global Support Train Accuracy: {sum(global_support_train_accuracies) / len(global_support_train_accuracies)}")
            if log_wandb and rank==0:
                wandb.log({"l1_train_loss": avg_l1_train_loss,
                        "l2_train_loss": avg_l2_train_loss*alpha,
                        "fs_train_accuracy": avg_fs_train_accuracy,
                        "global_query_train_accuracy": sum(global_query_train_accuracies) / len(global_query_train_accuracies),
                        "global_support_train_accuracy": sum(global_support_train_accuracies) / len(global_support_train_accuracies)})
            avg_fs_train_accuracy = []
            global_query_train_accuracies = []
            global_support_train_accuracies = []

        # Evaluation
        if step % eval_after_steps == 0 and step > 0:
            dist.barrier()
            train_progress_bar.close()
            model.eval()
            with torch.no_grad():
                dataloader.dataset.train = False
                test_accuracies = []
                test_losses = []
                eval_progress_bar = tqdm(total=len(dataloader), desc="Evaluation Progress")
                for elem in range(dataset.n_eval_steps):
                    test_support_set = elem["support_set"].squeeze(0)
                    test_target_set = elem["target_set"].squeeze(0)
                    test_target_labels = elem["target_labels"].squeeze(0).long()
                    test_support_labels = elem['support_labels'].long().squeeze(0).long()
                    test_batch_class_list = elem['batch_class_list'].squeeze(0).long()
                    test_logits, _, _ = model(test_support_set, test_support_labels, test_target_set, test_target_labels, class_name_embeddings, test_batch_class_list, dataset)

                    true_target_labels = torch.argsort(test_support_labels)[test_target_labels].to(rank)

                    test_loss = torch.nn.functional.cross_entropy(test_logits, true_target_labels.long().to(rank))
                    test_accuracy = compute_accuracy(test_logits, true_target_labels.long().to(rank))
                    test_losses.append(test_loss.item())
                    test_accuracies.append(test_accuracy)
                    eval_progress_bar.update(1)
                eval_progress_bar.close()
                if log_wandb and rank==0:
                    wandb.log({"test_loss": sum(test_losses) / len(test_losses), "test_accuracy": sum(test_accuracies) / len(test_accuracies)})
                avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
                print(f"Avg Test Loss: {sum(test_losses) / len(test_losses)}, Avg Test Accuracy: {avg_test_accuracy}")
                # Save the model with test accuracy as the name
                if rank == 0:
                    model_path = os.path.join(checkpoint_dir, f"model_{avg_test_accuracy:.4f}.pt")
                    torch.save(model.state_dict(), model_path)
            model.train()
            dataloader.dataset.train = True
            train_progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")
            dist.barrier()

        train_progress_bar.update(1)
        step += 1

    train_progress_bar.close()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)