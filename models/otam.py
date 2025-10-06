import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import compute_accuracy, BinaryClassificationModelSTRM
from einops import rearrange
import copy


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    """
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
    return torch.reshape(class_mask_indices, (-1,))


def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    """
    dists = F.pad(dists, (1,1), 'constant', 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 

    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


class OTAM(nn.Module):
    """
    OTAM model adapted for the FSOSAR framework
    """
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(OTAM, self).__init__()
        
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
        
        # Framework attributes
        self.way = config["way"]
        self.shot = config["shot"] 
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]
        
        # OTAM specific parameters
        self.feature_dim = 2048 if self.backbone_name == "resnet50" else 512
        self.lbda = config.get("otam_lambda", 0.1)
        
        # Open-set components
        self.disc = disc
        self.gc = gc
        if disc:
            self.discriminator = BinaryClassificationModelSTRM(self.feature_dim).cuda()
        if gc:
            self.garbage_prototype = nn.Parameter(torch.randn((1, self.feature_dim))).cuda()
            
        # Framework compatibility
        self.debug_samples_counter = 0
        self.classes_names = config["classes_names"]
        self.save_features = config.get("save_features", False)
        self.model_name = config.get("model_name", "OTAM")
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
        
        unique_labels = torch.unique(support_labels)
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        # Flatten features for frame-wise comparison
        support_features_flat = rearrange(support_features, 'b s d -> (b s) d')
        target_features_flat = rearrange(target_features, 'b s d -> (b s) d')

        # Compute frame similarities
        frame_sim = cos_sim(target_features_flat, support_features_flat)
        frame_dists = 1 - frame_sim
        
        # Reshape to [n_queries, n_support, query_seq_len, support_seq_len]
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)

        # Calculate query -> support and support -> query cumulative distances
        cum_dists = OTAM_cum_dist(dists, self.lbda) + OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'), self.lbda)

        # Compute class distances
        class_dists = []
        for c in unique_labels:
            class_indices = extract_class_indices(support_labels, c)
            class_dist = torch.mean(torch.index_select(cum_dists, 1, class_indices), dim=1)
            class_dists.append(class_dist)
        
        class_dists = torch.stack(class_dists)
        similarity_matrix = rearrange(class_dists, 'c q -> q c')
        similarity_matrix = -similarity_matrix  # Convert distances to similarities
        
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