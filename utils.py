import torch
import json
import os
import torch.distributed as dist
import random
import numpy as np
import socket
from sklearn.metrics import roc_auc_score, precision_recall_curve
import torch.nn as nn

class OpenSetLoss(torch.nn.Module):
    def __init__(self, os_loss):
        super(OpenSetLoss, self).__init__()
        self.os_loss = {"softmax": self.softmax_loss,
                          "eos": self.eos_loss,
                          "objectosphere": self.objectosphere_loss,
                          "discriminator": self.discriminator_loss,
                          "gc": self.gc_loss}
        self.os_loss = self.os_loss[os_loss]

    def gc_loss(self, logits, targets, similarity_matrix, rescale_function=None):
        return {"os_loss": None}

    def softmax_loss(self, logits, targets, similarity_matrix, rescale_function=None):
        return {"os_loss": None}

    def discriminator_loss(self, logits, targets, similarity_matrix, rescale_function=None):
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
        else:
            os_loss = None
        return {"os_loss": os_loss}

    def eos_loss(self, logits, targets, similarity_matrix, rescale_function=None):
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

    def objectosphere_loss(self, logits, targets, similarity_matrix, rescale_function):
        """
        how to determine alpha and epsilon?
        we push unknown logits to -16 since exp(-16) = 0.0000001
        we push known logits to 0 since exp(-0) = 1
        we set alpha to 0.01 because this value balances closed-set loss magnitude
        """
        epsilon = 1
        alpha = 0.01

        eos_loss = self.eos_loss(logits, targets, similarity_matrix)["unknown_loss"]
        scaled_similarity_matrix = rescale_function(similarity_matrix)
        # # sphere
        unknown_indices = targets == -1
        if unknown_indices.sum() > 0:  # if > -32, push norm of diff of unknown feature to -32
            unk_norms = scaled_similarity_matrix[unknown_indices]
            unk_norms = unk_norms.mean()
            unknown_sphere_loss = alpha*torch.maximum(unk_norms-epsilon, torch.tensor(0))
        else:
            unknown_sphere_loss = None

        known_indices = targets != -1
        if known_indices.sum() > 0:
            known_norms = scaled_similarity_matrix[known_indices]
            known_norms = known_norms.mean()
            known_sphere_loss = alpha*(-known_norms)
        else:
            known_sphere_loss = None

        return {"known_loss": torch.FloatTensor([0]).cuda(), "unknown_loss": eos_loss, "unknown_sphere_loss": unknown_sphere_loss, "known_sphere_loss": known_sphere_loss}

    def loss(self, logits, targets, similarity_matrix, rescale_function):
        return self.os_loss(logits, targets, similarity_matrix, rescale_function)


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
        x = x.view(-1, feat_dim)  # Flatten to (40*28, 1152)
        
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.drop3(self.act3(self.bn3(self.fc3(x))))
        
        # Additional processing per frame
        x = self.act4(self.fc4(x))  # (40*28, 64)
        
        # Reshape and aggregate
        x = x.view(batch_size, num_frames * 64)  # (40, 28*64)
        x = self.fc5(x)
        return self.sigmoid(x)


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
    features_dir = f'saved_features/{model_name}/{dataset_name}/{os_loss_name}'
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
    
    os.makedirs(f'visual_debug/{debug_samples_counter}', exist_ok=True)

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
        imageio.mimsave(f'visual_debug/{debug_samples_counter}/ss.gif', concatenated_frames, duration=250, loop=0)
        
        with open(f'visual_debug/{debug_samples_counter}/ss.txt', 'w') as f:
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
                    
                    imageio.mimsave(f'visual_debug/{debug_samples_counter}/{res}_{accept_score[cur]:.4f}_{query_label}.gif', 
                                   concatenated_frame, duration=250, loop=0)