import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import combinations
from torch.autograd import Variable
import torchvision.models as models
from utils import compute_accuracy, BinaryClassificationModelSTRM, split_first_dim_linear
from einops import rearrange
import copy

NUM_SAMPLES = 1

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()
        all_prototypes = torch.zeros(n_queries, self.args.way, 28, 1152).cuda() # 20 x 5 x 1152

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
            all_prototypes[:,c_idx] = (mh_queries_vs - query_prototype) # 20 x 5 x 1152
        
        return all_distances_tensor, all_prototypes

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
class TRX(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.
    Adapted for FSOSAR framework.
    """
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(TRX, self).__init__()

        self.train()
        
        # Convert config dict to args-like object for compatibility with original TRX
        class ArgsObject:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        self.args = ArgsObject(config)
        self.seq_len = self.args.seq_len
        
        # Set backbone method from config
        backbone_name = config.get("backbone", "resnet50")
        if backbone_name == "resnet18":
            self.args.method = "resnet18"
        elif backbone_name == "resnet34":
            self.args.method = "resnet34"
        elif backbone_name == "resnet50":
            self.args.method = "resnet50"
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Set up backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -1
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        # Freeze backbone if specified
        if config.get("freeze_backbone", True):
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Set TRX specific parameters
        if backbone_name == "resnet50":
            self.args.trans_linear_in_dim = 2048
        else:
            self.args.trans_linear_in_dim = 512
            
        self.args.trans_linear_out_dim = config.get("trans_linear_out_dim", 1152)
        self.args.temp_set = config.get("temp_set", [2, 3])
        self.args.trans_dropout = config.get("trans_dropout", 0.1)

        self.transformers = nn.ModuleList([TemporalCrossTransformer(self.args, s) for s in self.args.temp_set]) 

        # Open-set components
        self.disc = disc
        self.gc = gc
        if disc:
            feature_dim = self.args.trans_linear_out_dim
            self.discriminator = BinaryClassificationModelSTRM(feature_dim).cuda()
        if gc:
            self.garbage_prototype = nn.Parameter(torch.zeros(1, self.args.seq_len, self.args.trans_linear_in_dim), requires_grad=True).cuda()
            self.garbage_initialized = False
            
        # Framework compatibility
        self.debug_samples_counter = 0
        self.classes_names = config["classes_names"]
        self.save_features = config.get("save_features", False)
        self.model_name = config.get("model_name", "TRX")
        self.dataset_name = config.get("data_name", "unknown_dataset")
        self.os_loss_name = config.get("os_loss", "unknown_loss")

    def set_train(self):
        """Set model to training mode"""
        self.train()

    def set_eval(self):
        """Set model to evaluation mode"""
        self.eval()

    def initialize_garbage_prototype(self, support_features):
        """Initialize garbage prototype based on real feature statistics"""
        with torch.no_grad():
            # Compute statistics from real support features
            feature_mean = support_features.mean(dim=(0, 1))  # Mean across batch and time
            feature_std = support_features.std(dim=(0, 1))    # Std across batch and time
            
            # Initialize garbage prototype with same statistics
            self.garbage_prototype.data = torch.normal(
                mean=feature_mean.unsqueeze(0).expand(1, self.seq_len, -1),
                std=feature_std.unsqueeze(0).expand(1, self.seq_len, -1)
            )

    def forward(self, support_set, support_labels, target_set, batch_class_list=None, precomputed_context_features=None):
        """
        Forward pass following original TRX implementation
        """
        # Reshape inputs to match expected format [batch*seq_len, C, H, W]
        n_support_total = support_set.shape[0]
        n_target_total = target_set.shape[0]
        
        context_images = support_set.reshape(-1, 3, 224, 224)
        target_images = target_set.reshape(-1, 3, 224, 224)
        context_labels = support_labels

        # Extract features using backbone
        context_features = self.resnet(context_images).squeeze()
        target_features = self.resnet(target_images).squeeze()

        dim = int(context_features.shape[1])

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        # Add garbage class support if needed
        if self.gc and not self.garbage_initialized:
            self.initialize_garbage_prototype(context_features)
            self.garbage_initialized = True

        if self.gc:
            context_labels = torch.cat([context_labels, torch.tensor([self.args.way-1], device=context_labels.device)], dim=0)
            context_features = torch.cat([context_features, self.garbage_prototype], dim=0)

        # Apply transformers following original implementation
        all_logits, all_prototypes = [t(context_features, context_labels, target_features) for t in self.transformers][0]
        sample_logits = all_logits 
        # sample_logits = torch.mean(sample_logits, dim=[-1])

        # Use split_first_dim_linear as in original implementation
        similarity_matrix = split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]])
        
        # Remove extra dimension from split_first_dim_linear for FSOSAR compatibility
        if similarity_matrix.dim() == 3 and similarity_matrix.shape[0] == 1:
            similarity_matrix = similarity_matrix.squeeze(0)

        # Save features for analysis
        if self.save_features and batch_class_list is not None:
            from utils import save_tsne_features
            # Compute class prototypes for saving
            support_class_features = []
            for c in support_labels.unique():
                class_mask = support_labels == c
                class_features = context_features[class_mask].mean(dim=0).mean(dim=0)  # Average over samples and time
                support_class_features.append(class_features)
            support_class_features = torch.stack(support_class_features)
            
            episode_class_names = [self.classes_names[int(idx)] for idx in batch_class_list]
            save_tsne_features(
                features=support_class_features,
                episode_class_names=episode_class_names,
                model_name=self.model_name,
                dataset_name=self.dataset_name,
                os_loss_name=self.os_loss_name
            )

        # Discriminator for open-set detection
        disc_prob = None
        if self.disc:
            if self.disc:
                predictions = torch.argmax(similarity_matrix, dim=-1)
                best_diffs = all_prototypes[torch.arange(all_prototypes.shape[0]), predictions]
                disc_prob = self.discriminator(best_diffs)
                disc_prob = disc_prob.squeeze(-1)
            else:
                disc_prob = None
        
        return {
            "similarity_matrix": similarity_matrix,
            "disc_prob": disc_prob
        }

    def compute_known_losses(self, **kwargs):
        """
        Compute known class losses
        """
        similarity_matrix = kwargs["similarity_matrix"]
        target_labels = kwargs["target_labels"]
        
        # Handle the case where similarity_matrix has extra dimensions from split_first_dim_linear
        if similarity_matrix.dim() == 3 and similarity_matrix.shape[0] == 1:
            # Remove the first dimension if it's 1 (from NUM_SAMPLES=1)
            similarity_matrix = similarity_matrix.squeeze(0)
        
        # Cross-entropy loss on similarity matrix
        known_indices = target_labels != -1
        if known_indices.sum() > 0:
            ce_loss = F.cross_entropy(similarity_matrix[known_indices], target_labels[known_indices])
        else:
            ce_loss = torch.tensor(0.0, device=similarity_matrix.device)
        
        return {"known_loss": ce_loss}

    def compute_additional_metrics(self, **kwargs):
        """
        Compute additional metrics for logging
        """
        return {}

    def get_debug_data(self):
        """
        Get debug data for logging
        """
        return {}

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        """
        if hasattr(self.args, 'num_gpus') and self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)