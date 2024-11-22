from fsosar import FSOSAR
from videoloader import VideoDataset, SSv2
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define training parameters
log_train_after_steps = 10
eval_after_steps = 10
step = 0

# Initialize wandb
# wandb.init(project="fsosar")

# Load data
dataset = SSv2()
dataset.shot = 1  # NOTE: remove on the server!
dataset.seq_len = 8  # NOTE: !
dataset.query_per_class = 1 # NOTE: change on the server!
videodataset = VideoDataset(dataset)
dataloader = DataLoader(videodataset, batch_size=1, shuffle=True)

# Initialize model
model = FSOSAR(processor_name="MCG-NJU/videomae-base-finetuned-kinetics",
               model_name="MCG-NJU/videomae-base-finetuned-kinetics",
               bert_name='bert-base-uncased')
model.cuda()
model.train()

# Get features of class names with BERT
class_names = videodataset.class_folders
class_name_embeddings = []
for class_name in class_names:
    plain_class_name = class_name.replace('_', ' ')
    inputs = model.tokenizer(plain_class_name, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.bert_model(**inputs)
    class_name_embeddings.append(outputs.last_hidden_state.squeeze(0))

# Define optimizer
optimizer = torch.optim.Adam(list(model.mm_fusion_module.parameters()) + list(model.task_specific_learning_module.parameters()), lr=1e-4)

def compute_accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

# Loop variables
train_accuracies = []
train_losses = []

train_progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")

for elem in dataloader:
    support_set = elem["support_set"].squeeze(0)
    target_set = elem["target_set"].squeeze(0)
    target_labels = elem["target_labels"].squeeze(0)
    support_labels = elem['support_labels'].long().squeeze(0)
    batch_class_list = elem['batch_class_list'].squeeze(0)

    logits = model(support_set, support_labels, target_set, target_labels, class_name_embeddings, batch_class_list, dataset)
    l1_loss = torch.nn.functional.cross_entropy(logits, target_labels.long().cuda())

    optimizer.zero_grad()
    l1_loss.backward()
    optimizer.step()
   
    train_accuracy = compute_accuracy(logits, target_labels.long().cuda())
    train_losses.append(l1_loss.item())
    train_accuracies.append(train_accuracy)

    if step % log_train_after_steps == 0 and step > 0:
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        print(f"Avg Train Loss: {avg_train_loss}, Avg Train Accuracy: {avg_train_accuracy}")
        # wandb.log({"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy})
        train_losses = []
        train_accuracies = []

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
                test_logits = model(test_support_set, test_support_labels, test_target_set, test_target_labels, class_name_embeddings, test_batch_class_list, dataset)
                test_loss = torch.nn.functional.cross_entropy(test_logits, test_target_labels.long().cuda())
                test_accuracy = compute_accuracy(test_logits, test_target_labels.long().cuda())
                test_losses.append(test_loss.item())
                test_accuracies.append(test_accuracy)
                eval_progress_bar.update(1)
            eval_progress_bar.close()
            # wandb.log({"test_loss": sum(test_losses) / len(test_losses), "test_accuracy": sum(test_accuracies) / len(test_accuracies)})
            print(f"Avg Test Loss: {sum(test_losses) / len(test_losses)}, Avg Test Accuracy: {sum(test_accuracies) / len(test_accuracies)}")
        model.train()
        dataloader.dataset.train = True
        train_progress_bar = tqdm(total=eval_after_steps, desc="Training Progress")

    train_progress_bar.update(1)
    step += 1