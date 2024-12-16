import torch
import json
import os
import torch.distributed as dist

class OpenSetLoss(torch.nn.Module):
    def __init__(self):
        super(OpenSetLoss, self).__init__()

    def forward(self, logits, targets):
        if len(logits.shape) > 2:
            logits = logits.squeeze(0)

        known_indices = targets != -1
        if known_indices.sum() > 0:
            known_logits = torch.gather(logits[known_indices], 1, targets[known_indices].unsqueeze(1)).squeeze(1)  # 20
            known_loss = torch.exp(torch.tensor(1)) - torch.exp(known_logits)
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
        
        return known_loss, unknown_loss

def compute_accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

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
        averaged_values = {f"{self.prefix}{key}": sum(values) / len(values) for key, values in self.values.items()}
        self.values.clear()
        return averaged_values


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def load_configs(model_name, data_name):
    local_or_server = "server" if "iit.local" in os.getcwd() else "local"
    model_config_path = f"configs/{model_name}/{local_or_server}_config.json"
    data_config_path = f"configs/{data_name}/{local_or_server}_config.json"
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    model_config.update(data_config)
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
