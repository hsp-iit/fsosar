import torch
import json
import os
import torch.distributed as dist
import random
import numpy as np
import socket
from sklearn.metrics import roc_auc_score, precision_recall_curve

# a simple MLP for binary classification with 2 layers
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.dim_reduction = torch.nn.Linear(input_dim, 128)
        self.fc1 = torch.nn.Linear(128*28, 256)
        self.fc2 = torch.nn.Linear(256, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.relu(self.dim_reduction(x))
        x = x.reshape(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class OpenSetLoss(torch.nn.Module):
    def __init__(self, os_loss, model_dimension=None):
        super(OpenSetLoss, self).__init__()
        self.os_function = {"softmax": self.softmax,
                          "eos": self.eos,
                          "objectosphere": self.objectosphere,
                          "discriminator": self.discriminator}
        self.os_function = self.os_function[os_loss]
        self.os_loss = {"softmax": self.softmax_loss,
                          "eos": self.eos_loss,
                          "objectosphere": self.objectosphere_loss,
                          "discriminator": self.discriminator_loss}
        self.os_loss = self.os_loss[os_loss]
        self.model_dimension = model_dimension
        if os_loss == "discriminator":
            self.discriminator_model = MLP(model_dimension, model_dimension*2, 1).cuda()

    def softmax(self, logits, all_prototypes):
        return None

    def softmax_loss(self, logits, targets, similarity_matrix):
        return {"os_loss": None}

    def discriminator(self, logits, all_prototypes):
        preds = logits.max(dim=-1)[1]
        preds_features = all_prototypes[torch.arange(logits.shape[0]), preds, ...]
        os_scores = self.discriminator_model(preds_features)
        return os_scores

    def discriminator_loss(self, logits, targets, similarity_matrix):
        os_scores = logits["os_score"]
        pred = similarity_matrix.max(dim=-1)[1]
        correct = pred == targets
        if correct.sum() > 0:
            unknown_indices = targets == -1
            # get the first correc.sum() unknkown indices
            unknown_indices = unknown_indices.nonzero().squeeze(1)[:correct.sum()]
            correct_known_indices = (pred == targets).nonzero().squeeze(1)
            all_indices = torch.cat((unknown_indices, correct_known_indices))

            os_labels = (targets[all_indices] != -1).float()
            os_scores = os_scores[all_indices].squeeze(1)
            os_loss = torch.nn.functional.binary_cross_entropy(os_scores, os_labels)
            # print(os_scores)
            # print(os_labels)
            os_loss = os_loss * 10
        else:
            os_loss = None
        return {"os_loss": os_loss}


    def eos(self, logits, all_prototypes):
        return logits

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

    def objectosphere(self, logits, all_prototypes):
        return logits

    def objectosphere_loss(self, logits, targets, similarity_matrix):
        """
        how to determine alpha and epsilon?
        """
        epsilon = 32
        alpha = 0.0001

        eos_loss = self.eos_loss(logits, targets, similarity_matrix)["unknown_loss"]
        all_norms = logits["all_norms"]
        if len(all_norms.shape) > 2:
            all_norms = all_norms.squeeze(0)

        # # sphere
        unknown_indices = targets == -1
        if unknown_indices.sum() > 0:  # push norm of feature to 0
            unk_norms = all_norms[unknown_indices]
            unknown_sphere_loss = alpha*unk_norms.mean()
        else:
            unknown_sphere_loss = None

        known_indices = targets != -1
        if known_indices.sum() > 0:  # push norm of feature to epsilon
            known_norms = all_norms[known_indices]
            known_sphere_norm = known_norms.mean()
            known_sphere_loss = alpha*torch.maximum(epsilon-known_sphere_norm, torch.tensor(0))
        else:
            known_sphere_loss = None

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

def compute_oscr(targets, logits):
    """
    Computes the Open Set Classification Rate (OSCR) from logits and targets.
    
    Args:
        logits (torch.Tensor): Tensor of shape (n_queries, n_classes) containing logits for each query and class.
        targets (torch.Tensor): Vector of size (n_queries), where each element is the true label (0 to n_classes-1 for known classes) or -1 for unknown labels.
    
    Returns:
        float: OSCR score.
    """
    # Get the predicted class for each query (highest logit value)
    _, predicted_classes = torch.max(logits, dim=1)
    
    # Initialize counters for TP, FP, FN, TN
    tp, fp, fn, tn = 0, 0, 0, 0
    
    # Loop through all queries and compare predictions to targets
    for i in range(len(targets)):
        true_label = targets[i]
        predicted_label = predicted_classes[i]
        
        # Case 1: Known class (0 to n_classes-1)
        if true_label != -1:
            if predicted_label == true_label:
                tp += 1  # True Positive
            else:
                fn += 1  # False Negative
        
        # Case 2: Unknown class (-1)
        else:
            if predicted_label == true_label:
                tn += 1  # True Negative (correctly rejected)
            else:
                fp += 1  # False Positive (incorrectly accepted as known)

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Calculate OSCR (Open Set Classification Rate)
    oscr = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0
    
    return oscr


def compute_aupr(targets, logits):
    """
    Compute the Area Under the Precision-Recall Curve (AUPR).
    
    Parameters:
    - targets (array-like): True labels (binary: 0 or 1).
    - logits (array-like): Predicted probabilities for the positive class (usually output from a model).
    
    Returns:
    - float: The computed AUPR score.
    """
    # Compute precision, recall, and thresholds
    precision, recall, _ = precision_recall_curve(targets, logits)
    
    # Compute the area under the Precision-Recall curve (AUPR) using the trapezoidal rule
    aupr = np.trapz(precision, recall)
    
    return aupr
