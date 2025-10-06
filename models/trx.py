import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import combinations
from torch.autograd import Variable
import torchvision.models as models
from utils import compute_accuracy, BinaryClassificationModelSTRM
from einops import rearrange
import copy


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    """
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
    return torch.reshape(class_mask_indices, (-1,))


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
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
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args["seq_len"] * 1.5)
        self.pe = PositionalEncoding(self.args["trans_linear_in_dim"], self.args["trans_dropout"], max_len=max_len)

        self.k_linear = nn.Linear(self.args["trans_linear_in_dim"] * temporal_set_size, self.args["trans_linear_out_dim"])
        self.v_linear = nn.Linear(self.args["trans_linear_in_dim"] * temporal_set_size, self.args["trans_linear_out_dim"])

        self.norm_k = nn.LayerNorm(self.args["trans_linear_out_dim"])
        self.norm_v = nn.LayerNorm(self.args["trans_linear_out_dim"])
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args["seq_len"])]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
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
        all_distances_tensor = torch.zeros(n_queries, self.args["way"], device=queries.device)

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args["trans_linear_out_dim"])

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
        
        return all_distances_tensor


class TRX(nn.Module):
    """
    TRX model adapted for the FSOSAR framework
    """
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(TRX, self).__init__()
        
        # Set up backbone
        self.backbone_name = config.get("backbone", "resnet50")
        if self.backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)  
        elif self.backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=True)
        elif self.backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
            
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Freeze backbone if specified
        if config.get("freeze_backbone", True):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Framework attributes
        self.way = config["way"]
        self.shot = config["shot"] 
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]
        
        # TRX specific parameters
        config["trans_linear_in_dim"] = 2048 if self.backbone_name == "resnet50" else 512
        config["trans_linear_out_dim"] = config.get("trans_linear_out_dim", 1152)
        config["temp_set"] = config.get("temp_set", [2, 3])
        config["trans_dropout"] = config.get("trans_dropout", 0.1)
        
        # Build transformers
        self.transformers = nn.ModuleList([TemporalCrossTransformer(config, s) for s in config["temp_set"]])
        
        # Open-set components
        self.disc = disc
        self.gc = gc
        if disc:
            feature_dim = config["trans_linear_in_dim"]
            self.discriminator = BinaryClassificationModelSTRM(feature_dim).cuda()
        if gc:
            self.garbage_prototype = nn.Parameter(torch.randn((1, config["trans_linear_out_dim"]))).cuda()
            
        # Framework compatibility
        self.debug_samples_counter = 0
        self.classes_names = config["classes_names"]
        self.save_features = config.get("save_features", False)
        self.model_name = config.get("model_name", "TRX")
        self.dataset_name = config.get("data_name", "unknown_dataset")
        self.os_loss_name = config.get("os_loss", "unknown_loss")

    def get_feats(self, support_images, target_images):
        """
        Extract features using backbone CNN
        """
        # Combine support and target for batch processing
        all_images = torch.cat([support_images, target_images], dim=0)
        all_features = self.backbone(all_images).squeeze()
        
        # Split back into support and target
        n_support = support_images.shape[0]
        support_features = all_features[:n_support]
        target_features = all_features[n_support:]
        
        dim = int(support_features.shape[1])
        
        # Reshape to [batch, seq_len, features]
        support_features = support_features.reshape(-1, self.seq_len, dim)
        target_features = target_features.reshape(-1, self.seq_len, dim)
        
        return support_features, target_features

    def set_train(self):
        """Set model to training mode"""
        self.train()

    def set_eval(self):
        """Set model to evaluation mode"""
        self.eval()

    def forward(self, support_set, support_labels, target_set, batch_class_list=None, precomputed_context_features=None):
        """
        Forward pass adapted for FSOSAR framework
        """
        # Reshape inputs to match expected format [batch*seq_len, C, H, W]
        n_support_total = support_set.shape[0]
        n_target_total = target_set.shape[0]
        
        support_images = support_set.reshape(-1, 3, 224, 224)
        target_images = target_set.reshape(-1, 3, 224, 224)
        
        # Extract features
        support_features, target_features = self.get_feats(support_images, target_images)
        
        # Apply transformers
        all_logits = [t(support_features, support_labels, target_features) for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        similarity_matrix = torch.mean(all_logits, dim=[-1])
        
        # Save features for analysis
        if self.save_features and batch_class_list is not None:
            from utils import save_tsne_features
            # Compute class prototypes for saving
            support_class_features = []
            for c in support_labels.unique():
                class_mask = support_labels == c
                class_features = support_features[class_mask].mean(dim=0).mean(dim=0)  # Average over samples and time
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
        
        # Add garbage class for GC
        if self.gc:
            n_queries = similarity_matrix.shape[0]
            garbage_scores = torch.zeros(n_queries, 1, device=similarity_matrix.device)
            similarity_matrix = torch.cat([similarity_matrix, garbage_scores], dim=1)
        
        # Discriminator for open-set detection
        disc_prob = None
        if self.disc:
            # Use query features for discriminator
            query_features_flat = target_features.mean(dim=1)  # Average over time
            predictions = torch.argmax(similarity_matrix, dim=-1)
            
            # Get prototype features for best matching class
            support_prototypes = []
            for c in support_labels.unique():
                class_mask = support_labels == c
                prototype = support_features[class_mask].mean(dim=0).mean(dim=0)
                support_prototypes.append(prototype)
            support_prototypes = torch.stack(support_prototypes)
            
            # Compute differences for discriminator
            best_prototypes = support_prototypes[predictions]
            differences = query_features_flat - best_prototypes
            disc_prob = self.discriminator(differences.unsqueeze(1).expand(-1, self.seq_len, -1))
        
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