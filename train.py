from safsar import SAFSAR
from videoloader import VideoDataset, SSv2
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import cv2
import os

# Define training parameters depenging of the server
deploy_server = "iit.local" in os.getcwd()

lr = 1e-5
alpha = 0.3
log_train_after_steps = 3
eval_after_steps = 30000 if deploy_server else 3
step = 0
log_wandb = deploy_server

# Load data
dataset = SSv2()
dataset.shot = 1  # One shot for GPU
dataset.seq_len = 8  # 8 Frames for comparison
dataset.query_per_class = 1
if deploy_server:
    dataset.path = "/home/sberti_datasets/SSv2/images_in_class_folders"
else:
    dataset.n_eval_steps=3
videodataset = VideoDataset(dataset)
dataloader = DataLoader(videodataset, batch_size=1, shuffle=True)

# Get features of class names with BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.cuda()
bert_model.eval()
class_names = videodataset.class_folders
class_name_embeddings = []
for class_name in class_names:
    plain_class_name = class_name.replace('_', ' ')
    inputs = tokenizer(plain_class_name, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    class_name_embeddings.append(outputs.last_hidden_state.squeeze(0))
bert_model = None
tokenizer = None
n_train_classes = len(set(videodataset.train_split.gt_a_list))

# Initialize model
model = SAFSAR(processor_name="MCG-NJU/videomae-base-finetuned-kinetics",
               model_name="MCG-NJU/videomae-base-finetuned-kinetics",
               n_train_classes=n_train_classes)
model.cuda()
model.train()

# Initialize wandb
if log_wandb:
    wandb.init(project="fsosar")
    wandb.watch(model)

# Define optimizer
optimizer = torch.optim.Adam(list(model.mm_fusion_module.parameters()) + list(model.task_specific_learning_module.parameters()), 
                             lr=lr)

def compute_accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

# Loop variables
train_accuracies = []
l1_train_losses = []
l2_train_losses = []

train_progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")

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
    l1_loss = torch.nn.functional.cross_entropy(logits, target_labels.long().cuda())

    # Compute L2 loss
    unique_classes = videodataset.train_split.get_unique_classes()
    global_support_labels = torch.tensor([unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
    global_query_labels = torch.tensor([unique_classes.index(x) for x in real_target_labels]).cuda()
    # Make support label one hot to apply them correctly
    l2_loss_support = torch.nn.functional.cross_entropy(support_global_scores.reshape(-1, len(unique_classes)),
                                                        global_support_labels.repeat(dataset.way))
    l2_loss_query = torch.nn.functional.cross_entropy(query_global_scores, global_query_labels)
    l2_loss = l2_loss_support + l2_loss_query

    # Optimization
    optimizer.zero_grad()
    (l1_loss + alpha*l2_loss).backward()
    optimizer.step()
   
    # Logging
    train_accuracy = compute_accuracy(logits, target_labels.long().cuda())
    l1_train_losses.append(l1_loss.item())
    l2_train_losses.append(l2_loss.item())
    train_accuracies.append(train_accuracy)
    if step % log_train_after_steps == 0 and step > 0:
        avg_l1_train_loss = sum(l1_train_losses) / len(l1_train_losses)
        avg_l2_train_loss = sum(l2_train_losses) / len(l2_train_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        print(f"Avg L1 Train Loss: {avg_l1_train_loss}, Avg L2 Train Loss: {avg_l2_train_loss*alpha}, Avg Train Accuracy: {avg_train_accuracy}")
        if log_wandb:
            wandb.log({"l1_train_loss": avg_l1_train_loss,
                       "l2_train_loss": avg_l2_train_loss*alpha,
                       "train_accuracy": avg_train_accuracy})
        train_losses = []
        train_accuracies = []

    # Evaluation
    if step % eval_after_steps == 0 and step > 0:
        train_progress_bar.close()
        model.eval()
        with torch.no_grad():
            dataloader.dataset.train = False
            test_accuracies = []
            test_losses = []
            eval_progress_bar = tqdm(total=len(dataloader), desc="Evaluation Progress")
            for elem in dataloader:
                test_support_set = elem["support_set"].squeeze(0)
                test_target_set = elem["target_set"].squeeze(0)
                test_target_labels = elem["target_labels"].squeeze(0)
                test_support_labels = elem['support_labels'].long().squeeze(0)
                test_batch_class_list = elem['batch_class_list'].squeeze(0)
                test_logits, _, _ = model(test_support_set, test_support_labels, test_target_set, test_target_labels, class_name_embeddings, test_batch_class_list, dataset)
                test_loss = torch.nn.functional.cross_entropy(test_logits, test_target_labels.long().cuda())
                test_accuracy = compute_accuracy(test_logits, test_target_labels.long().cuda())
                test_losses.append(test_loss.item())
                test_accuracies.append(test_accuracy)
                eval_progress_bar.update(1)
            eval_progress_bar.close()
            if log_wandb:
                wandb.log({"test_loss": sum(test_losses) / len(test_losses), "test_accuracy": sum(test_accuracies) / len(test_accuracies)})
            print(f"Avg Test Loss: {sum(test_losses) / len(test_losses)}, Avg Test Accuracy: {sum(test_accuracies) / len(test_accuracies)}")
        model.train()
        dataloader.dataset.train = True
        train_progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")

    train_progress_bar.update(1)
    step += 1

train_progress_bar.close()