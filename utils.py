import torch
import json
import os
import torch.distributed as dist
import random
import numpy as np
import socket
from sklearn.metrics import roc_auc_score, precision_recall_curve
import torch.nn as nn


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)


class OpenSetLoss(torch.nn.Module):
    def __init__(self, os_loss):
        super(OpenSetLoss, self).__init__()
        self.os_loss = {"softmax": self.softmax_loss,
                          "eos": self.eos_loss,
                          "discriminator": self.discriminator_loss,
                          "gc": self.gc_loss}
        self.os_loss = self.os_loss[os_loss]

    def gc_loss(self, logits, targets, similarity_matrix):
        return {"os_loss": None}

    def softmax_loss(self, logits, targets, similarity_matrix):
        return {"os_loss": None}

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
            os_scores = os_scores[all_indices]
            # os_labels = torch.ones_like(os_labels)  # TODO REMOVE DEBUG
            os_scores = os_scores.squeeze(-1)  # sometimes it remain an extra dimension
            os_loss = torch.nn.functional.binary_cross_entropy(os_scores, os_labels)
        else:
            os_loss = None
        return {"os_loss": os_loss}

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


def setup(rank, world_size, set_seeds, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

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
    if "/home/sberti" in cwd:
        local_or_server = "local"
        datasets_path = "/home/sberti"
        host = "local"
        log_path = "/home/sberti/logs"
    elif "iit.local" in cwd:
        local_or_server = "server"
        datasets_path = "/home/sberti_datasets"
        host = "gnode04"
        log_path = "."
    elif "/fastwork/sberti" in cwd:
        local_or_server = "server"
        datasets_path = "/fastwork/sberti"
        host = "franklin"
        log_path = "/fastwork/sberti"
        
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
    model_config["log_path"] = log_path
    return model_config


class BinaryClassificationModelSAFSAR(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModelSAFSAR, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim*2, input_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(input_dim, 64)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Shape is 40, 28, 1152
        x = self.act1(self.fc1(x))  # Shape is 40X28, 512
        x = self.act2(self.fc2(x))  # Shape is 40X28, 128
        x = self.act3(self.fc3(x))  # Shape is 40X28, 64
        x = self.sigmoid(self.fc4(x))
        return x


class BinaryClassificationModelSTRM(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModelSTRM, self).__init__()
        # Feature extraction layers (per frame)
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        
        # Aggregation and classification layers
        self.fc4 = nn.Linear(128, 64)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(64 * 28, 1)  # 28 elements * 64 features
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, num_frames, features) = (40, 28, 1152)
        batch_size, num_frames, feat_dim = x.shape
        
        # Process each frame independently
        x = x.reshape(-1, feat_dim)  # Flatten to (40*28, 1152) - use reshape instead of view for non-contiguous tensors
        
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.drop3(self.act3(self.bn3(self.fc3(x))))
        
        # Additional processing per frame
        x = self.act4(self.fc4(x))  # (40*28, 64)
        
        # Reshape and aggregate
        x = x.view(batch_size, num_frames * 64)  # (40, 28*64)
        x = self.fc5(x)
        return self.sigmoid(x)
    

def initialize_garbage_prototype(garbage_prototype, support_features):
    """Initialize garbage prototype based on real feature statistics"""
    with torch.no_grad():
        # Compute statistics from real support features
        feature_mean = support_features.mean()  # Mean across batch and time
        feature_std = support_features.std()    # Std across batch and time

        # Initialize garbage prototype with random values using computed statistics
        # Create a tensor of the same shape as garbage_prototype filled with random values
        garbage_prototype.data = torch.normal(
            mean=feature_mean.item(),
            std=feature_std.item(),
            size=garbage_prototype.shape,
            device=garbage_prototype.device
        )
    return garbage_prototype


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

    points = []

    for thr in np.unique(os_score):
        unknown_indices = targets == -1
        fp = (os_score[unknown_indices] >= thr).sum()
        fpr = fp / (fp+unknown_indices.sum())

        known_indices = targets != -1
        correctly_classified = np.argmax(logits, axis=-1)[known_indices] == targets[known_indices]
        ccr = (correctly_classified & (os_score[known_indices] >= thr)).sum() / known_indices.sum()

        points.append((fpr, ccr))
    
    points = sorted(points, key=lambda x: x[0])
    points = [(0, 0)] + points + [(1, 1)]
    oscr = 0.0

    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        oscr += (x2-x1) * (y1+y2) / 2

    return oscr


if __name__ == "__main__":
    targets = [0, 1, 2, -1, -1, -1]
    logits = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 0],
              [1, 0, 0],
              [1, 0, 0]]
    os_score = [0.7, 0.9, 0.8, 0.2, 0.3, 0.4]
    print(compute_oscr(targets, logits, os_score))

    targets = [0, 1, 2, -1, -1, -1]
    logits = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 0],
              [1, 0, 0],
              [1, 0, 0]]
    os_score = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    print(compute_oscr(targets, logits, os_score))


def save_tsne_features(features, episode_class_names, model_name, dataset_name, os_loss_name):
    """
    Generic function to save features for t-SNE analysis
    
    Args:
        features: torch.Tensor - Features to save [num_classes, feature_dim]
        episode_class_names: list - Class names for the current episode (already indexed by batch_class_list)
        model_name: str - Name of the model (e.g., 'SAFSAR', 'STRM')
        dataset_name: str - Name of the dataset (e.g., 'HMDB51', 'UCF101')
        os_loss_name: str - Name of the open set loss (e.g., 'softmax', 'discriminator')
        
    Returns:
        str: Path to the saved file
    """
    import pickle
    import os
    
    # Create features directory with structured path
    features_dir = f'data_analysis/saved_features/{model_name}/{dataset_name}/{os_loss_name}'
    os.makedirs(features_dir, exist_ok=True)
    filename = f'{features_dir}/class_features.pkl'
    
    # Load existing data or create new dictionary
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            class_feature_dict = pickle.load(f)
    else:
        class_feature_dict = {}
    
    # Add features organized by class name
    for i in range(len(features)):
        class_name = episode_class_names[i]
        
        # Initialize class entry if not exists
        if class_name not in class_feature_dict:
            class_feature_dict[class_name] = {
                'features': [],
                'feature_dim': features.shape[-1],
                'model_type': model_name
            }
        
        # Append new feature
        class_feature_dict[class_name]['features'].append(
            features[i].detach().cpu().numpy()
        )
    
    # Save updated dictionary
    with open(filename, 'wb') as f:
        pickle.dump(class_feature_dict, f)
    
    # Print summary
    total_features = sum(len(data['features']) for data in class_feature_dict.values())
    print(f"Features appended to {filename} - Total: {total_features} features across {len(class_feature_dict)} classes")
    
    return filename


def save_confidence_scores(similarity_matrix, target_labels, model_name, dataset_name, os_loss_name, logits=None):
    """
    Save confidence scores for known and unknown queries incrementally for later histogram analysis
    
    Args:
        similarity_matrix: Predictions similarity matrix
        target_labels: Labels for query samples (-1 for unknown)
        model_name: Name of the model (e.g., 'SAFSAR', 'STRM', 'D2ST')
        dataset_name: Name of the dataset (e.g., 'HMDB51', 'UCF101')
        os_loss_name: Name of the open set loss (e.g., 'softmax', 'discriminator')
        logits: Model logits (for discriminator-based confidence)
    """
    import pickle
    import os
    import numpy as np
    import torch
    
    # Create confidence scores directory with structured path
    scores_dir = f'data_analysis/confidence_scores/{model_name}/{dataset_name}/{os_loss_name}'
    os.makedirs(scores_dir, exist_ok=True)
    filename = f'{scores_dir}/confidence_scores.pkl'
    
    # Load existing data or create new dictionary
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            scores_data = pickle.load(f)
    else:
        scores_data = {
            'known_scores': [],
            'unknown_scores': [],
            'model_type': model_name,
            'dataset': dataset_name,
            'os_loss': os_loss_name,
            'episode_count': 0
        }
    
    # Compute confidence scores based on the open set loss method
    if os_loss_name in ["softmax", "eos"]:
        # For implicit methods, use max similarity as confidence
        if model_name == "STRM":
            confidence_scores = torch.exp(similarity_matrix).max(dim=-1)[0].detach().cpu().numpy()
        elif model_name in ["SAFSAR", "D2ST"]:
            confidence_scores = ((similarity_matrix + 1) / 2).max(dim=-1)[0].detach().cpu().numpy()
        elif model_name in ["TRX", "OTAM"]:
            confidence_scores = torch.nn.functional.softmax(similarity_matrix, dim=-1).max(dim=-1)[0].detach().cpu().numpy()
    elif os_loss_name == "discriminator" and logits is not None:
        # For explicit discriminator method
        disc_prob = logits["disc_prob"]
        if disc_prob is not None:
            confidence_scores = disc_prob.squeeze(-1).detach().cpu().numpy()
        else:
            # Fallback to max similarity if disc_prob not available
            confidence_scores = similarity_matrix.max(dim=-1)[0].detach().cpu().numpy()
    elif os_loss_name == "gc":
        # For GC method
        gc_probs = torch.nn.functional.softmax(similarity_matrix, dim=-1)
        confidence_scores = 1 - gc_probs[:, -1].detach().cpu().numpy()  # 1 - unknown_prob = known_prob
    else:
        raise ValueError(f"Unsupported os_loss_name: {os_loss_name}")
    
    # Convert target labels to numpy
    target_labels_np = target_labels.detach().cpu().numpy()
    
    # Separate known and unknown scores
    known_mask = target_labels_np != -1
    unknown_mask = target_labels_np == -1
    
    # Append known scores
    if known_mask.sum() > 0:
        known_confidences = confidence_scores[known_mask].tolist()
        scores_data['known_scores'].extend(known_confidences)
    
    # Append unknown scores
    if unknown_mask.sum() > 0:
        unknown_confidences = confidence_scores[unknown_mask].tolist()
        scores_data['unknown_scores'].extend(unknown_confidences)
    
    # Increment episode count
    scores_data['episode_count'] += 1
    
    # Save updated scores data
    with open(filename, 'wb') as f:
        pickle.dump(scores_data, f)
    
    # Print summary
    total_known = len(scores_data['known_scores'])
    total_unknown = len(scores_data['unknown_scores'])
    print(f"Confidence scores updated at {filename} - Known: {total_known}, Unknown: {total_unknown} from {scores_data['episode_count']} episodes")
    
    return filename


def save_confusion_matrix(similarity_matrix, support_labels, target_labels, batch_class_list, 
                         classes_names, model_name, dataset_name, os_loss_name, logits=None, 
                         unknown_labels=None, os_threshold=0.5, add_unknown_class=True):
    """
    Save confusion matrix data incrementally for later visualization
    Includes unknown queries that are incorrectly classified as known classes
    
    Args:
        similarity_matrix: Predictions similarity matrix
        support_labels: Labels for support samples
        target_labels: Labels for query samples (-1 for unknown)
        batch_class_list: Class indices for current batch
        classes_names: List of all class names
        model_name: Name of the model (e.g., 'SAFSAR', 'STRM')
        dataset_name: Name of the dataset (e.g., 'HMDB51', 'UCF101')
        os_loss_name: Name of the open set loss (e.g., 'softmax', 'discriminator')
        logits: Model logits (for discriminator-based os_prob)
        unknown_labels: True class labels for unknown samples (for proper confusion matrix entries)
        os_threshold: Threshold for considering unknown as wrongly classified (default: 0.5)
        add_unknown_class: If True (default), add "unknown" class to confusion matrix and handle unknown predictions properly.
                          If False, keeps current behavior (only add misclassified unknowns to their true class)
    """
    import pickle
    import os
    import numpy as np
    import torch
    
    # Create confusion matrix directory with structured path
    cm_dir = f'data_analysis/confusion_matrices/{model_name}/{dataset_name}/{os_loss_name}'
    os.makedirs(cm_dir, exist_ok=True)
    filename = f'{cm_dir}/confusion_data.pkl'
    
    # Load existing data or create new dictionary
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cm_data = pickle.load(f)
    else:
        # Set up matrix size and class names based on add_unknown_class flag
        if add_unknown_class:
            # Add "unknown" class to the list
            matrix_classes = classes_names + ["unknown"]
            matrix_size = len(matrix_classes)
        else:
            # Only use known classes in confusion matrix
            matrix_classes = classes_names
            matrix_size = len(classes_names)
            
        cm_data = {
            'confusion_matrix': np.zeros((matrix_size, matrix_size), dtype=int),
            'class_names': matrix_classes,
            'model_type': model_name,
            'dataset': dataset_name,
            'os_loss': os_loss_name,
            'episode_count': 0,
            'os_threshold': os_threshold,
            'add_unknown_class': add_unknown_class,
            'unknown_correctly_rejected': 0,
            'unknown_misclassified_count': 0
        }
    
    # Get predictions from similarity matrix
    predictions = similarity_matrix.argmax(dim=-1).detach().cpu().numpy()
    target_labels_np = target_labels.detach().cpu().numpy()
    batch_class_list_np = batch_class_list.detach().cpu().numpy()
    
    # Compute os_prob based on the open set loss method
    if os_loss_name in ["softmax", "eos"]:
        # For implicit methods, use max similarity as os_prob
        if model_name == "STRM":
            os_prob = torch.exp(similarity_matrix).max(dim=-1)[0].detach().cpu().numpy()
        elif model_name in ["SAFSAR", "D2ST"]:
            os_prob = ((similarity_matrix + 1) / 2).max(dim=-1)[0].detach().cpu().numpy()
        elif model_name in ["TRX", "OTAM"]:
            os_prob = torch.nn.functional.softmax(similarity_matrix, dim=-1).max(dim=-1)[0].detach().cpu().numpy()
    elif os_loss_name == "discriminator" and logits is not None:
        # For explicit discriminator method
        os_prob = logits.get("disc_prob", None)
        if os_prob is not None:
            os_prob = os_prob.squeeze(-1).detach().cpu().numpy()
    elif os_loss_name == "gc":
        # For GC method
        gc_probs = torch.nn.functional.softmax(similarity_matrix, dim=-1)
        os_prob = 1 - gc_probs[:, -1].detach().cpu().numpy()  # 1 - unknown_prob = known_prob
    
    # Process known queries (target_labels != -1)
    known_mask = target_labels_np != -1
    if known_mask.sum() > 0:
        known_predictions = predictions[known_mask]
        known_targets = target_labels_np[known_mask]
        known_os_probs = os_prob[known_mask]
        
        if add_unknown_class:
            # When add_unknown_class=True, check if known queries are predicted as unknown
            unknown_class_idx = len(classes_names)  # Index of "unknown" class
            
            for pred_idx, true_idx, os_prob_val in zip(known_predictions, known_targets, known_os_probs):
                true_class_idx = int(batch_class_list_np[true_idx])
                
                # Check if model predicts this known query as unknown (os_prob <= threshold)
                if os_prob_val <= os_threshold:
                    # Known query predicted as unknown -> [true_known_class, unknown]
                    cm_data['confusion_matrix'][true_class_idx, unknown_class_idx] += 1
                else:
                    # Known query predicted as known class -> normal confusion matrix entry
                    pred_class_idx = int(batch_class_list_np[pred_idx])
                    cm_data['confusion_matrix'][true_class_idx, pred_class_idx] += 1
        else:
            # Original behavior: only use similarity matrix predictions (no unknown prediction handling)
            for pred_idx, true_idx in zip(known_predictions, known_targets):
                pred_class_idx = int(batch_class_list_np[pred_idx])
                true_class_idx = int(batch_class_list_np[true_idx])
                cm_data['confusion_matrix'][true_class_idx, pred_class_idx] += 1
    
    # Process unknown queries (target_labels == -1)
    unknown_mask = target_labels_np == -1
    
    if unknown_mask.sum() > 0:
        unknown_predictions = predictions[unknown_mask]
        unknown_os_probs = os_prob[unknown_mask]
        
        # Get the true class labels for unknown samples
        if unknown_labels is not None:
            unknown_labels_np = unknown_labels.detach().cpu().numpy()
            unknown_true_labels = unknown_labels_np[unknown_mask]
        else:
            unknown_true_labels = None
        
        if add_unknown_class:
            # When add_unknown_class=True, treat unknowns as a separate class
            unknown_class_idx = len(classes_names)  # Index of "unknown" class
            
            for i, pred_idx in enumerate(unknown_predictions):
                pred_class_idx = int(batch_class_list_np[pred_idx])
                
                # Check if model predicts unknown (os_prob <= threshold)
                if unknown_os_probs[i] <= os_threshold:
                    # Model correctly predicts unknown -> diagonal entry for unknown class
                    cm_data['confusion_matrix'][unknown_class_idx, unknown_class_idx] += 1
                    cm_data['unknown_correctly_rejected'] += 1
                else:
                    # Model predicts known class when true class is unknown -> off-diagonal entry
                    cm_data['confusion_matrix'][unknown_class_idx, pred_class_idx] += 1
                    cm_data['unknown_misclassified_count'] += 1
        else:
            # Original behavior: only add misclassified unknowns to their true class
            for i, pred_idx in enumerate(unknown_predictions):
                pred_class_idx = int(batch_class_list_np[pred_idx])
                
                # Check if unknown is wrongly classified as known (os_prob > threshold)
                if unknown_os_probs[i] > os_threshold:
                    # Only add to confusion matrix if we have the true class labels for unknown samples
                    if unknown_true_labels is not None:
                        # Use the actual true class of the unknown sample
                        true_unknown_class_idx = int(unknown_true_labels[i])
                        # Add this as a misclassification in the confusion matrix
                        # [true_class, predicted_class] - true class is the unknown's actual class
                        cm_data['confusion_matrix'][true_unknown_class_idx, pred_class_idx] += 1
                        cm_data['unknown_misclassified_count'] += 1
                    else:
                        # Just count unknown misclassified without adding to matrix
                        cm_data['unknown_misclassified_count'] += 1
                else:
                    # Unknown correctly rejected - just count it but don't add to matrix
                    cm_data['unknown_correctly_rejected'] += 1
    
    # Increment episode count
    cm_data['episode_count'] += 1
    
    # Save updated confusion matrix data
    with open(filename, 'wb') as f:
        pickle.dump(cm_data, f)
    
    # Print summary
    total_predictions = cm_data['confusion_matrix'].sum()
    print(f"Confusion matrix updated at {filename} - Total predictions: {total_predictions} from {cm_data['episode_count']} episodes")
    
    return filename


def visual_debug(similarity_matrix, support_set, target_set, support_labels, 
                target_labels, batch_class_list, unknown_labels, videodataset, 
                logits, debug_samples_counter, config):
    """
    Simplified visual debug function for both SAFSAR and STRM models
    Creates GIF visualizations for debugging purposes
    """
    import imageio
    import os
    import numpy as np
    import torch
    
    # Get configuration parameters (must exist)
    way = config['way']
    shot = config['shot']
    query_per_class = config['query_per_class']
    seq_len = config['seq_len']
    disc_prob = logits.get('disc_prob', None)
    
    # Create structured path like save_tsne_features
    model_name = config['model_name']
    dataset_name = config['data_name']  
    os_loss_name = config['os_loss']
    debug_dir = f'data_analysis/visual_debug/{model_name}/{dataset_name}/{os_loss_name}/{debug_samples_counter}'
    os.makedirs(debug_dir, exist_ok=True)

    # Generate GIF visualizations
    if support_set is not None and target_set is not None:
        # Save support set gif
        support_set = support_set.reshape(-1, seq_len, 224, 3, 224)
        support_set = support_set[torch.argsort(support_labels)]
        support_set = support_set.reshape(way, shot, seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        concatenated_frames = []
        support_classes = [videodataset.class_folders[int(batch_class_list[i])] for i in range(way)]

        for i in range(way):
            for j in range(shot):
                frames = []
                for k in support_set[i, j]:
                    frame_rgb = k.cpu().numpy()
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    frames.append(frame_rgb)
                concatenated_frames.append(frames)

        concatenated_frames = np.stack([np.stack(x) for x in concatenated_frames])
        concatenated_frames = concatenated_frames.reshape(way, shot, seq_len, 224, 224, 3)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        imageio.mimsave(f'{debug_dir}/ss.gif', concatenated_frames, duration=250, loop=0)
        
        with open(f'{debug_dir}/ss.txt', 'w') as f:
            for item in support_classes:
                f.write("%s\n" % item)

        # Analyze predictions for query visualization
        if similarity_matrix is not None:
            good_closed = similarity_matrix.argmax(dim=-1) == target_labels
            if disc_prob is None:
                accept_score = similarity_matrix.max(dim=-1).values.detach().cpu().numpy()
            else:
                accept_score = disc_prob.detach().cpu().numpy()
            
            true_open = target_labels != -1
            pred_open = accept_score > 0.5

            # Save queries gif
            target_set = target_set.reshape(way, query_per_class, seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
            query_labels = []
            
            for i in range(way):
                for j in range(query_per_class):
                    if target_labels[i*query_per_class+j] == -1:
                        query_label = videodataset.class_folders[int(unknown_labels[i*query_per_class+j])]
                    else:
                        query_label = videodataset.class_folders[int(batch_class_list[target_labels[i*query_per_class+j]])]
                    query_labels.append(query_label)
                    
                    concatenated_frame = []
                    for k in target_set[i, j]:
                        frame_rgb = k.cpu().numpy()
                        frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                        concatenated_frame.append(frame_rgb)
                    
                    # Determine classification result
                    cur = i*query_per_class+j
                    res = ""
                    if true_open[cur] and pred_open[cur] and good_closed[cur]:
                        res = "TP"
                    elif not true_open[cur] and not pred_open[cur]:
                        res = "TN"
                    elif true_open[cur] and not pred_open[cur]:
                        res = "FN"
                    elif not true_open[cur] and pred_open[cur]:
                        res = "FP"
                    
                    # Fix formatting issue - ensure score is a scalar
                    imageio.mimsave(f'{debug_dir}/{res}_{accept_score[cur].item():.4f}_{query_label}.gif', 
                                   concatenated_frame, duration=250, loop=0)