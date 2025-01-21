import torch
import json
import os
import torch.distributed as dist
import random
import numpy as np
from sklearn.metrics import roc_auc_score


class OpenSetLoss(torch.nn.Module):
    def __init__(self, os_loss):
        super(OpenSetLoss, self).__init__()
        self.os_losses = {"PEELER": self.peeler,
                          "RfdNET": self.rfdnet,
                          "None": self.none}
        self.os_loss = self.os_losses[os_loss]

    def none(self, logits, targets):
        return {"known_loss": torch.FloatTensor([0]).cuda(), 
                "unknown_loss": torch.FloatTensor([0]).cuda()}

    def rfdnet(self, logits, targets):
        raise Exception("To implement")

    def peeler(self, logits, targets):
        if len(logits.shape) > 2:
            logits = logits.squeeze(0)

        known_indices = targets != -1
        if known_indices.sum() > 0:
            known_logits = torch.gather(logits[known_indices], 1, targets[known_indices].unsqueeze(1)).squeeze(1)  # 20
            known_loss = torch.full_like(known_logits, fill_value=torch.exp(torch.tensor(1))) - torch.exp(known_logits)
            known_loss = known_loss.mean()  # TODO mean or sum?
        else:
            known_loss = None

        unknown_indices = targets == -1
        if unknown_indices.sum() > 0:
            unknown_logits = logits[unknown_indices].reshape(-1)
            pos_unknown_logits = unknown_logits[unknown_logits > 0]
            pos_unknown_logits = pos_unknown_logits.mean()
            unknown_loss = -1 + torch.exp(pos_unknown_logits)
        else:
            unknown_loss = None
        
        return {"known_loss": known_loss, "unknown_loss": unknown_loss}

    def forward(self, logits, targets):
        return self.os_loss(logits, targets)


class AverageMeter:
    def __init__(self, prefix=""):
        self.values = {}
        self.prefix = prefix

    def update(self, input_dict):
        for key, value in input_dict.items():
            if key not in self.values:
                self.values[key] = []
            if value == None:
                continue
            if type(value) == torch.Tensor:
                value = value.item()
            self.values[key].append(value)

    def average(self):
        for key in self.values.values():
            if len(key) == 0:
                key.append(0)
        averaged_values = {f"{self.prefix}{key}": sum(values) / len(values) for key, values in self.values.items()}
        self.values.clear()
        return averaged_values

    def get_average(self):
        for key in self.values.values():
            if len(key) == 0:
                key.append(0)
        averaged_values = {f"{self.prefix}{key}": sum(values) / len(values) for key, values in self.values.items()}
        return averaged_values


def compute_accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def setup(rank, world_size, set_seeds):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # Set seed for pytorch for reproducibility
    if set_seeds:
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)
        torch.cuda.manual_seed_all(rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        # Set seed for random module
        random.seed(rank)
        np.random.seed(rank)


def load_configs(model_name, data_name):
    local_or_server = "server" if "iit.local" in os.getcwd() else "local"
    model_config_path = f"configs/{model_name}.json"
    data_config_path = f"configs/{data_name}.json"
    train_config_path = f"configs/{local_or_server}_config.json"
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    with open(train_config_path, 'r') as f:
        train_config = json.load(f)
    model_config.update(data_config)
    model_config.update(train_config)
    return model_config


class DataArgs:
    def __init__(self, config):
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.path = config["path"]
        self.shot = config["shot"]
        self.n_eval_steps = config["n_eval_steps"]
        self.traintestlist = "splits/ssv2_OTAM"
        self.img_size = config["img_size"]
        self.way = config["way"]
        self.split = config["split"]
        self.debug_loader = config["debug_loader"]

def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def compute_auroc(similarity_matrix, true_target_labels):
    # OPEN SET PART: AUROC
    target_os_matrix = (torch.zeros_like(similarity_matrix).cuda()+1)/2
    for i, elem in enumerate(true_target_labels):
        if elem != -1:
            target_os_matrix[i, elem] = 1
    open_set_scores = similarity_matrix.amax(dim=1).detach().cpu().numpy()
    open_set_targets = target_os_matrix.amax(dim=1).detach().cpu().numpy().astype(int)
    # AUROC is defined only if there are both positive and negative samples
    if open_set_targets.sum() > 0 and open_set_targets.sum() < len(open_set_targets):
        os_auroc = roc_auc_score(open_set_targets, open_set_scores)
    else:
        os_auroc = None
    return os_auroc
