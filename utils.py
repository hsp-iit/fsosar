import torch
import json
import os
import torch.distributed as dist

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
