import torch
import json
import os
import torch.distributed as dist
import random
import numpy as np
import socket
from sklearn.metrics import roc_auc_score, precision_recall_curve

class OpenSetLoss(torch.nn.Module):
    def __init__(self, os_loss):
        super(OpenSetLoss, self).__init__()
        # self.os_function = {"softmax": self.softmax,
        #                   "eos": self.eos,
        #                   "objectosphere": self.objectosphere,
        #                   "discriminator": self.discriminator}
        # self.os_function = self.os_function[os_loss]
        self.os_loss = {"softmax": self.softmax_loss,
                          "eos": self.eos_loss,
                          "objectosphere": self.objectosphere_loss,
                          "discriminator": self.discriminator_loss,
                          "gc": self.gc_loss}
        self.os_loss = self.os_loss[os_loss]

    def gc_loss(self, logits, targets, similarity_matrix):
        return {"os_loss": None}

    # def softmax(self, logits, all_prototypes):
    #     return None

    def softmax_loss(self, logits, targets, similarity_matrix):
        return {"os_loss": None}

    # def discriminator(self, logits, all_prototypes):
    #     preds = logits.max(dim=-1)[1]
    #     preds_features = all_prototypes[torch.arange(logits.shape[0]), preds, ...]
    #     os_scores = self.discriminator_model(preds_features)
    #     return os_scores

    def discriminator_loss(self, logits, targets, similarity_matrix):
        os_scores = logits["disc_prob"]
        pred = similarity_matrix.max(dim=-1)[1]
        correct = pred == targets
        # correct = torch.ones_like(pred).bool() # TODO remove debug, this is to test the discriminator
        if correct.sum() > 0:
            unknown_indices = targets == -1
            # get the first correc.sum() unknkown indices
            unknown_indices = unknown_indices.nonzero().squeeze(1)[:correct.sum()]
            correct_known_indices = (pred == targets).nonzero().squeeze(1)
            all_indices = torch.cat((unknown_indices, correct_known_indices))

            os_labels = (targets[all_indices] != -1).float()
            os_scores = os_scores[all_indices].squeeze(1)
            # os_labels = torch.ones_like(os_labels)  # TODO REMOVE DEBUG
            os_loss = torch.nn.functional.binary_cross_entropy(os_scores, os_labels)
            print(os_scores)
            print(os_labels)
            os_loss = os_loss * 1000
        else:
            os_loss = None
        return {"os_loss": os_loss}


    # def eos(self, logits, all_prototypes):
    #     return logits

    def eos_loss(self, logits, targets, similarity_matrix):
        """
        for known queries, it uses cross-entropy loss
        so here we define only the case for unknown queries
        Intuitively, it pushes unknown logits to have the same values
        """
        if len(similarity_matrix.shape) > 2:
            similarity_matrix = similarity_matrix.squeeze(0)

        unknown_indices = targets == -1
        if unknown_indices.sum() > 0:
            similarity_matrix = similarity_matrix[unknown_indices]
            probs = torch.nn.functional.softmax(similarity_matrix, dim=-1)
            probs = probs + 1e-6  # avoid log(0)
            unknown_loss = -torch.log(probs).mean()
        else:
            unknown_loss = None

        return {"known_loss": torch.FloatTensor([0]).cuda(), "unknown_loss": unknown_loss}

    # def objectosphere(self, logits, all_prototypes):
    #     return logits

    def objectosphere_loss(self, logits, targets, similarity_matrix):
        """
        how to determine alpha and epsilon?
        we push unknown logits to 32 since exp(-16) = e-07
        we push known logits to 0 since exp(-0) = 1
        we set alpha to 0.0001 because this value balances closed-set loss magnitude
        """
        epsilon = 16
        alpha = 0.01

        eos_loss = self.eos_loss(logits, targets, similarity_matrix)["unknown_loss"]
        # # sphere
        unknown_indices = targets == -1
        if unknown_indices.sum() > 0:  # if > -32, push norm of diff of unknown feature to -32
            unk_norms = similarity_matrix[unknown_indices]
            unk_norms = unk_norms.mean()
            unknown_sphere_loss = alpha*torch.maximum(unk_norms-epsilon, torch.tensor(0))
        else:
            unknown_sphere_loss = None

        known_indices = targets != -1
        if known_indices.sum() > 0:
            known_norms = similarity_matrix[known_indices]
            known_norms = known_norms.mean()
            known_sphere_loss = alpha*(-known_norms)
        else:
            known_sphere_loss = None

        # all_norms = logits["all_norms"]
        # if len(all_norms.shape) > 2:
        #     all_norms = all_norms.squeeze(0)

        # # # sphere
        # unknown_indices = targets == -1
        # if unknown_indices.sum() > 0:  # push norm of feature to 0
        #     unk_norms = all_norms[unknown_indices]
        #     unknown_sphere_loss = alpha*unk_norms.mean()
        # else:
        #     unknown_sphere_loss = None

        # known_indices = targets != -1
        # if known_indices.sum() > 0:  # push norm of feature to epsilon
        #     known_norms = all_norms[known_indices]
        #     known_sphere_norm = known_norms.mean()
        #     known_sphere_loss = alpha*torch.maximum(epsilon-known_sphere_norm, torch.tensor(0))
        # else:
        #     known_sphere_loss = None

        return {"known_loss": torch.FloatTensor([0]).cuda(), "unknown_loss": eos_loss, "unknown_sphere_loss": unknown_sphere_loss, "known_sphere_loss": known_sphere_loss}

    def forward(self, logits, all_prototypes):
        return self.os_function(logits, all_prototypes)

    def loss(self, logits, targets, similarity_matrix):
        return self.os_loss(logits, targets, similarity_matrix)


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

def is_address_in_use(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((ip, port))
        except socket.error as e:
            if e.errno == socket.errno.EADDRINUSE:
                return True
            else:
                raise
    return False


def setup(rank, world_size, set_seeds):
    os.environ['MASTER_ADDR'] = 'localhost'
    # To deal with multiple training on one machine
    ports = [12355, 12356, 12357, 12358, 12359]
    for port in ports:
        if not is_address_in_use('localhost', port):
            os.environ['MASTER_PORT'] = str(port)
            break
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
    cwd = os.getcwd()
    if "steb6" in cwd:
        local_or_server = "local"
        datasets_path = "/home/steb6/datasets"
        host = "local"
    elif "iit.local" in cwd:
        local_or_server = "server"
        datasets_path = "/home/sberti_datasets"
        host = "gnode04"
    elif "sberti" in cwd:
        local_or_server = "server"
        datasets_path = "/work/sberti"
        host = "franklin"
        
    model_config_path = f"configs/{model_name}.json"
    data_config_path = f"configs/{data_name}.json"
    train_config_path = f"configs/{local_or_server}_config.json"
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    data_config["path"] = f"{datasets_path}/{data_config['path']}"
    with open(train_config_path, 'r') as f:
        train_config = json.load(f)
    model_config.update(data_config)
    model_config.update(train_config)
    model_config["host"] = host
    return model_config


class DataArgs:
    def __init__(self, config):
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.path = config["path"]
        self.shot = config["shot"]
        self.n_eval_steps = config["n_eval_steps"]
        self.traintestlist = config["traintestlist"]
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
    """
    similarity_matrix must be defined between 0 and 1
    """
    target_os_matrix = torch.zeros_like(similarity_matrix).cuda()
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


def compute_aupr(targets, logits):
    precision, recall, _ = precision_recall_curve(targets, logits)
    
    # Reverse the arrays to ensure recall is in ascending order
    precision = np.flip(precision)
    recall = np.flip(recall)
    
    # Compute the area under the PR curve using the trapezoidal rule
    aupr = np.trapz(precision, recall)
    
    return aupr


def compute_oscr(targets, logits, os_score):
    targets = np.array(targets)
    logits = np.array(logits)
    os_score = np.array(os_score)
    
    predicted = np.argmax(logits, axis=1)
    
    known_mask = (targets != -1)
    unknown_mask = (targets == -1)
    
    known_total = np.sum(known_mask)
    unknown_total = np.sum(unknown_mask)
    
    if known_total == 0 or unknown_total == 0:
        return 0.0
    
    correct_known_mask = (predicted == targets) & known_mask
    correct_known_os = os_score[correct_known_mask]
    os_score_unknown = os_score[unknown_mask]
    
    cko_sorted = np.sort(correct_known_os)
    osu_sorted = np.sort(os_score_unknown)
    
    thresholds = np.concatenate([correct_known_os, os_score_unknown, [1.0, 0.0]])
    thresholds = np.unique(thresholds)
    
    max_osu = osu_sorted[-1] if unknown_total > 0 else 0.0
    tau_prime = max_osu + 1e-9
    thresholds = np.concatenate([thresholds, [tau_prime]])
    thresholds = np.unique(thresholds)
    thresholds.sort()
    thresholds = thresholds[::-1]
    
    prev_ccr = 0.0
    prev_crr = 0.0
    area = 0.0
    
    for tau in thresholds:
        tp = len(correct_known_os) - np.searchsorted(cko_sorted, tau, side='left')
        ccr = tp / known_total
        
        tn = np.searchsorted(osu_sorted, tau, side='left')
        crr = tn / unknown_total
        
        area += (crr - prev_crr) * (prev_ccr + ccr) / 2.0
        prev_ccr, prev_crr = ccr, crr
    
    return max(0.0, min(area, 1.0))
