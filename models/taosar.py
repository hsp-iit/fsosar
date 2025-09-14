import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy
from utils import compute_accuracy, BinaryClassificationModelSAFSAR
import wandb
import numpy as np
import torchvision.models as models
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import math

class TemporalConsistencyModule(nn.Module):
    """Enhanced temporal consistency analysis with window-based analysis"""
    def __init__(self, feature_dim, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Multi-scale consistency analysis
        self.local_consistency = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim // 2),
            nn.GELU(),
            nn.Conv1d(feature_dim // 2, feature_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim // 4),
            nn.GELU()
        )
        
        # Enhanced consistency scoring network
        self.consistency_net = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal attention for consistency weighting
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, frame_features):
        """
        Args:
            frame_features: (batch_size, seq_len, feature_dim)
        Returns:
            consistency_score: (batch_size,) - higher means more consistent/certain
        """
        batch_size, seq_len, feature_dim = frame_features.shape
        
        if seq_len > 1:
            # Ensure frame_features has consistent dtype with attention weights
            frame_features = frame_features.to(self.temporal_attention.in_proj_weight.dtype)
            
            # Apply temporal attention
            attended_features, _ = self.temporal_attention(
                frame_features, frame_features, frame_features
            )
            
            # Local consistency analysis using convolution
            features_transposed = attended_features.transpose(1, 2)  # [B, D, T]
            # Ensure dtype consistency for Conv1d
            features_transposed = features_transposed.to(self.local_consistency[0].weight.dtype)
            local_features = self.local_consistency(features_transposed)  # [B, D/4, T]
            local_features = local_features.transpose(1, 2)  # [B, T, D/4]
            
            # Global temporal variance
            temporal_var = attended_features.var(dim=1)  # Variance across time
            
            # Combine local and global features
            local_pooled = local_features.mean(dim=1)  # [B, D/4]
            combined_features = torch.cat([temporal_var, local_pooled], dim=1)  # [B, D + D/4]
            
            # Ensure combined_features has the same dtype as the network weights
            combined_features = combined_features.to(self.consistency_net[0].weight.dtype)
            consistency_score = self.consistency_net(combined_features).squeeze(1)
            
            # Higher consistency for lower variance (invert the score)
            consistency_score = 1.0 - consistency_score
        else:
            consistency_score = torch.ones(batch_size, device=frame_features.device, dtype=frame_features.dtype)
        
        return consistency_score

class CrossFrameUncertaintyNet(nn.Module):
    """Enhanced uncertainty estimation with attention mechanism"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Temporal attention for uncertainty estimation
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced uncertainty estimation network
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Frame-level uncertainty for aggregation
        self.frame_uncertainty = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frame_features):
        """
        Args:
            frame_features: (batch_size, seq_len, feature_dim)
        Returns:
            uncertainty_score: (batch_size,) - higher means more uncertain
        """
        batch_size, seq_len, feature_dim = frame_features.shape
        
        # Ensure frame_features has consistent dtype with attention weights
        frame_features = frame_features.to(self.temporal_attention.in_proj_weight.dtype)
        
        # Apply temporal attention to capture frame relationships
        attended_features, attention_weights = self.temporal_attention(
            frame_features, frame_features, frame_features
        )
        
        # Compute frame-level uncertainties
        frame_uncertainties = self.frame_uncertainty(attended_features).squeeze(-1)  # [B, T]
        
        # Weighted pooling based on attention and uncertainty
        attention_mean = attention_weights.mean(dim=1)  # [B, T]
        combined_weights = attention_mean * (1.0 + frame_uncertainties)  # Higher uncertainty gets more weight
        weighted_features = (attended_features * combined_weights.unsqueeze(-1)).sum(dim=1)
        
        # Ensure features have the same dtype as the network weights
        weighted_features = weighted_features.to(self.uncertainty_estimator[0].weight.dtype)
        uncertainty = self.uncertainty_estimator(weighted_features).squeeze(1)
        
        return uncertainty

class MotionAwareRejectionHead(nn.Module):
    """Enhanced rejection mechanism with sophisticated temporal analysis"""
    def __init__(self, feature_dim, motion_threshold=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.motion_threshold = motion_threshold
        
        # Temporal gradient computation for motion analysis
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim // 2,
            kernel_size=3,
            padding=1
        )
        
        # Motion pattern analysis
        self.motion_analyzer = nn.Sequential(
            nn.Linear(feature_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Enhanced rejection network with more sophisticated fusion
        self.rejection_net = nn.Sequential(
            nn.Linear(5, 64),  # motion_intensity + motion_complexity + consistency + uncertainty + temporal_stability
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frame_features, consistency_score, uncertainty_score):
        """
        Args:
            frame_features: (batch_size, seq_len, feature_dim)
            consistency_score: (batch_size,)
            uncertainty_score: (batch_size,)
        Returns:
            rejection_score: (batch_size,) - higher means more likely to reject (unknown)
        """
        batch_size, seq_len, feature_dim = frame_features.shape
        
        if seq_len > 1:
            # Ensure frame_features has consistent dtype
            frame_features = frame_features.to(self.temporal_conv.weight.dtype)
            
            # Compute temporal gradients for motion analysis
            frame_features_transposed = frame_features.transpose(1, 2)  # [B, D, T]
            temporal_gradients = self.temporal_conv(frame_features_transposed)  # [B, D/2, T]
            temporal_gradients = temporal_gradients.transpose(1, 2)  # [B, T, D/2]
            
            # Motion intensity: average magnitude of temporal changes
            motion_intensity = temporal_gradients.abs().mean(dim=[1, 2])  # [B]
            
            # Motion complexity: variance in temporal gradients
            motion_complexity = temporal_gradients.var(dim=1).mean(dim=1)  # [B]
            
            # Motion pattern analysis
            motion_features = temporal_gradients.mean(dim=1)  # [B, D/2]
            motion_pattern_score = self.motion_analyzer(motion_features).squeeze(1)  # [B]
            
            # Temporal stability: consistency of frame-to-frame changes
            frame_diffs = torch.diff(frame_features, dim=1)  # [B, T-1, D]
            temporal_stability = 1.0 - frame_diffs.norm(dim=2).var(dim=1)  # [B] (higher = more stable)
        else:
            # Single frame: no motion information available
            motion_intensity = torch.zeros(batch_size, device=frame_features.device, dtype=frame_features.dtype)
            motion_complexity = torch.zeros(batch_size, device=frame_features.device, dtype=frame_features.dtype)
            motion_pattern_score = torch.zeros(batch_size, device=frame_features.device, dtype=frame_features.dtype)
            temporal_stability = torch.ones(batch_size, device=frame_features.device, dtype=frame_features.dtype)
        
        # Combine all rejection cues with enhanced feature set
        rejection_input = torch.stack([
            motion_intensity,
            motion_complexity,
            consistency_score,
            uncertainty_score,
            temporal_stability
        ], dim=1)
        
        # Ensure rejection_input has the same dtype as the network weights
        rejection_input = rejection_input.to(self.rejection_net[0].weight.dtype)
        rejection_score = self.rejection_net(rejection_input).squeeze(1)
        
        return rejection_score

class TemporalPrototypeTracker(nn.Module):
    """Enhanced prototype tracker with cross-attention and temporal dynamics"""
    def __init__(self, feature_dim, num_prototypes):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        
        # Cross-attention for support-query interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced prototype update network
        self.prototype_updater = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        
        # Enhanced temporal importance weighting
        self.temporal_weighting = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Prototype refinement network
        self.prototype_refiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, support_features, support_labels, query_features=None):
        """
        Args:
            support_features: (batch_size, feature_dim) - temporal-aggregated support features
            support_labels: (batch_size,)
            query_features: (batch_size, feature_dim) - optional query features for cross-attention
        Returns:
            prototypes: (num_classes, feature_dim)
        """
        prototypes = []
        unique_labels = support_labels.unique()
        
        # Apply cross-attention if query features are available
        if query_features is not None:
            # Ensure consistent dtypes for cross-attention
            support_features = support_features.to(self.cross_attention.in_proj_weight.dtype)
            query_features = query_features.to(self.cross_attention.in_proj_weight.dtype)
            
            support_features_attended, _ = self.cross_attention(
                support_features.unsqueeze(0),  # Add batch dim for attention
                query_features.unsqueeze(0),
                query_features.unsqueeze(0)
            )
            support_features = support_features_attended.squeeze(0)
        
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            
            # Ensure class_features has the same dtype as the network weights
            class_features = class_features.to(self.temporal_weighting[0].weight.dtype)
            
            # Compute enhanced temporal importance weights
            temporal_weights = self.temporal_weighting(class_features)
            weighted_features = class_features * temporal_weights
            
            # Create initial prototype
            if len(class_features) > 1:
                # Use weighted average
                prototype = weighted_features.sum(dim=0) / temporal_weights.sum()
            else:
                prototype = class_features.squeeze(0)
            
            # Refine prototype through additional network
            prototype = self.prototype_refiner(prototype)
            prototypes.append(prototype)
        
        return torch.stack(prototypes, dim=0)

class TAOSAR(nn.Module):
    """Temporal-Aware Open-Set Action Recognition"""
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(TAOSAR, self).__init__()
        
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]
        
        # Feature extractor backbone - use more powerful ResNet50 for better performance
        self.backbone_type = config.get("backbone", "resnet50")  # Changed from resnet18
        self.feature_extractor = self._build_feature_extractor()
        
        # Distribute feature extractor for 5-shot training
        if dp:
            self.feature_extractor = torch.nn.DataParallel(self.feature_extractor)
            
        # Get feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()
        
        # Enhanced multi-scale temporal processing for better temporal modeling
        self.temporal_scales = [2, 4, 8]  # Keep max 8 frames as requested
        self.scale_processors = nn.ModuleList([
            self._build_transformer(
                self.feature_dim,
                config.get("num_layers_temporal", 3),  # Keep 3 layers
                config.get("num_heads", 8),  # Keep 8 heads
                config.get("intermediate_size", 2048)  # Keep reasonable size
            ) for _ in self.temporal_scales
        ])
        
        # Enhanced scale fusion network for multi-scale features
        if len(self.temporal_scales) > 1:
            self.scale_fusion = nn.Sequential(
                nn.Linear(self.feature_dim * len(self.temporal_scales), self.feature_dim * 2),
                nn.LayerNorm(self.feature_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
        else:
            # No fusion needed for single scale
            self.scale_fusion = nn.Identity()
        
        # Enhanced TAOSAR-specific modules with increased complexity
        self.temporal_consistency = TemporalConsistencyModule(self.feature_dim, window_size=3)
        self.uncertainty_estimator = CrossFrameUncertaintyNet(self.feature_dim, num_heads=8)
        self.motion_rejection = MotionAwareRejectionHead(self.feature_dim)
        self.prototype_tracker = TemporalPrototypeTracker(self.feature_dim, self.way)
        
        # Classification layers
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
        # Global classification layer for L2 loss
        self.use_l2_loss = config["use_l2_loss"]
        if self.use_l2_loss:
            self.global_classification_layer = nn.Linear(self.feature_dim, config["n_train_classes"])
            
        self.alpha = config["alpha"]
        self.debug_samples_counter = 0
        
        # Open set components
        if gc:
            self.garbage_prototype = nn.Parameter(torch.randn((1, self.feature_dim))).cuda()
        self.discriminator = BinaryClassificationModelSAFSAR(self.feature_dim).cuda()
        self.gc = gc
        self.disc = disc

    def set_train(self):
        self.use_l2_loss = True

    def set_eval(self):
        self.use_l2_loss = False

    def _build_feature_extractor(self):
        """Build 2D CNN feature extractor that processes frames"""
        if self.backbone_type == "resnet18":
            resnet = models.resnet18(pretrained=True)
            return nn.Sequential(*list(resnet.children())[:-2])
        elif self.backbone_type == "resnet50":
            resnet = models.resnet50(pretrained=True)
            return nn.Sequential(*list(resnet.children())[:-2])
        else:
            raise NotImplementedError(f"Backbone {self.backbone_type} not implemented")

    def _get_feature_dim(self):
        """Get the feature dimension from the backbone"""
        if self.backbone_type == "resnet18":
            return 512
        elif self.backbone_type == "resnet50":
            return 2048
        else:
            return 512

    def _build_transformer(self, hidden_size, num_layers, num_heads, intermediate_size):
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=intermediate_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        return TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    def extract_features(self, videos):
        """Extract temporal features from videos with memory-efficient processing"""
        batch_size, seq_len, channels, height, width = videos.shape
        
        # Process frames in reasonable chunks
        frame_features_list = []
        chunk_size = min(24, batch_size * seq_len)  # Process 24 frames at a time (reasonable increase)
        
        # Reshape to process all frames together
        frames = videos.view(-1, channels, height, width)
        
        # Process frames in chunks to avoid memory overflow
        for i in range(0, frames.size(0), chunk_size):
            chunk = frames[i:i+chunk_size]
            with torch.amp.autocast('cuda'):  # Use mixed precision
                chunk_features = self.feature_extractor(chunk)
                # Global average pooling to get fixed-size features
                chunk_features = F.adaptive_avg_pool2d(chunk_features, (1, 1))
                chunk_features = chunk_features.squeeze(-1).squeeze(-1)
            frame_features_list.append(chunk_features)
        
        # Concatenate all chunks
        frame_features = torch.cat(frame_features_list, dim=0)
        
        # Reshape back to video format
        frame_features = frame_features.view(batch_size, seq_len, self.feature_dim)
        
        # Simplified temporal processing for memory efficiency
        scale_features = []
        for i, (scale, processor) in enumerate(zip(self.temporal_scales, self.scale_processors)):
            # Subsample frames for this scale or use all frames if single scale
            if scale < seq_len:
                indices = torch.linspace(0, seq_len-1, scale).long()
                scale_frames = frame_features[:, indices, :]
            elif scale > seq_len:
                # Repeat frames if scale is larger
                repeat_factor = scale // seq_len + 1
                scale_frames = frame_features.repeat(1, repeat_factor, 1)[:, :scale, :]
            else:
                scale_frames = frame_features
            
            # Apply temporal transformer with gradient checkpointing
            with torch.amp.autocast('cuda'):
                processed_scale = processor(scale_frames)
                # Global temporal pooling
                scale_feature = processed_scale.mean(dim=1)
            scale_features.append(scale_feature)
        
        # Fuse features (or just return single scale if only one)
        if len(scale_features) > 1:
            fused_features = torch.cat(scale_features, dim=1)
            video_features = self.scale_fusion(fused_features)
        else:
            video_features = scale_features[0]
        
        return video_features, frame_features

    def forward(self, support_set, support_labels, target_set, batch_class_list=None, precomputed_context_features=None):
        
        # Extract features from support and query sets with memory optimization
        support_set = support_set.reshape(-1, self.seq_len, 3, 224, 224)
        
        # Process support set
        with torch.amp.autocast('cuda'):
            support_features, support_frame_features = self.extract_features(support_set)
        
        # Process target set
        if len(target_set.shape) == 4:
            n_queries = int(target_set.shape[0] / self.seq_len)
        elif len(target_set.shape) == 5:
            n_queries = target_set.shape[0]
        target_set = target_set.reshape(n_queries, self.seq_len, 3, 224, 224)
        
        with torch.amp.autocast('cuda'):
            query_features, query_frame_features = self.extract_features(target_set)
        
        # Get enhanced temporal-aware prototypes with cross-attention
        prototypes = self.prototype_tracker(support_features, support_labels, query_features)
        
        # Compute temporal consistency and uncertainty for queries (with reduced complexity)
        consistency_scores = self.temporal_consistency(query_frame_features)
        uncertainty_scores = self.uncertainty_estimator(query_frame_features)
        
        # Compute similarities using temperature-scaled cosine similarity
        similarity_matrix = []
        for prototype in prototypes:
            # Ensure both tensors have the same dtype
            query_features_dtype = query_features.to(prototype.dtype)
            similarities = self.cosine_similarity(
                query_features_dtype, 
                prototype.unsqueeze(0).expand(query_features_dtype.size(0), -1)
            )
            similarity_matrix.append(similarities * self.temperature)
        
        similarity_matrix = torch.stack(similarity_matrix, dim=1)
        
        # Add garbage class if using open-set
        if self.gc:
            query_features_dtype = query_features.to(self.garbage_prototype.dtype)
            garbage_similarity = self.cosine_similarity(
                query_features_dtype, 
                self.garbage_prototype.expand(query_features_dtype.size(0), -1)
            ) * self.temperature
            similarity_matrix = torch.cat([similarity_matrix, garbage_similarity.unsqueeze(1)], dim=-1)
        
        # Motion-aware rejection for discriminator (simplified)
        if self.disc:
            rejection_scores = self.motion_rejection(
                query_frame_features, consistency_scores, uncertainty_scores
            )
            disc_prob = rejection_scores.unsqueeze(1)
        else:
            disc_prob = None

        # Compute global logits for L2 loss
        if self.use_l2_loss:
            # Ensure features have the same dtype as the global classification layer weights
            support_features_casted = support_features.to(self.global_classification_layer.weight.dtype)
            query_features_casted = query_features.to(self.global_classification_layer.weight.dtype)
            support_global_logits = self.global_classification_layer(support_features_casted)
            query_global_logits = self.global_classification_layer(query_features_casted)
        else:
            support_global_logits = 0.
            query_global_logits = 0.

        return {
            "similarity_matrix": similarity_matrix,
            "support_global_logits": support_global_logits,
            "query_global_logits": query_global_logits,
            "disc_prob": disc_prob,
            "consistency_scores": consistency_scores,
            "uncertainty_scores": uncertainty_scores
        }

    def compute_known_losses(self, similarity_matrix, support_global_logits, query_global_logits,
                           true_target_labels=None, target_labels=None, support_labels=None, 
                           batch_class_list=None, disc_prob=None, consistency_scores=None, 
                           uncertainty_scores=None):
        true_target_labels = target_labels
        known_indices = true_target_labels != -1
        similarity_matrix_k = similarity_matrix[known_indices]
        known_true_target_labels = true_target_labels[known_indices]
        
        # Compute L1 loss (few-shot classification loss)
        l1_loss = F.cross_entropy(similarity_matrix_k, known_true_target_labels)

        # Add consistency regularization loss
        if consistency_scores is not None:
            # Encourage high consistency for known samples
            consistency_loss = F.mse_loss(
                consistency_scores[known_indices], 
                torch.ones_like(consistency_scores[known_indices])
            )
            l1_loss = l1_loss + 0.1 * consistency_loss

        # Compute L2 loss (global classification loss)
        if self.use_l2_loss:
            global_support_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
            if self.gc:
                known_indices_global = target_labels != (self.way-1)
            else:
                known_indices_global = known_indices
            global_query_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[target_labels[known_indices_global]]]).cuda()
            l2_loss_support = F.cross_entropy(support_global_logits, global_support_labels)
            l2_loss_query = F.cross_entropy(query_global_logits[known_indices_global], global_query_labels)
            l2_loss = l2_loss_support + l2_loss_query
        else:
            l2_loss = 0.

        self.debug_data = {
            "similarity_matrix": wandb.Table(columns=list(range(similarity_matrix.shape[-1])), data=similarity_matrix.detach().cpu().numpy().tolist()),
            "true_target_labels": wandb.Table(columns=[0], data=target_labels.detach().cpu().numpy()[..., None]),
            "similarity_matrix_mean": similarity_matrix.mean().item(),
            "consistency_scores_mean": consistency_scores.mean().item() if consistency_scores is not None else 0,
            "uncertainty_scores_mean": uncertainty_scores.mean().item() if uncertainty_scores is not None else 0
        }
        
        return {"l1_loss": l1_loss, "l2_loss": self.alpha * l2_loss}

    def get_debug_data(self):
        return self.debug_data

    def compute_additional_metrics(self, similarity_matrix, support_global_logits, query_global_logits, 
                                 support_labels, target_labels, batch_class_list, disc_prob=None, 
                                 consistency_scores=None, uncertainty_scores=None):
        known_indices = target_labels != -1
        if known_indices.sum() > 0:
            similarity_matrix_k = similarity_matrix[known_indices]
            target_labels_k = target_labels[known_indices]

            true_target_labels = torch.argsort(support_labels)[target_labels_k].cuda()
            fs_acc = compute_accuracy(similarity_matrix_k, true_target_labels)

            if self.use_l2_loss:
                query_global_logits_k = query_global_logits[known_indices]
                unique_classes = self.train_unique_classes
                global_support_labels = torch.tensor([unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
                global_query_labels_k = torch.tensor([unique_classes.index(x) for x in batch_class_list[target_labels_k]]).cuda()
                global_support_acc = compute_accuracy(support_global_logits, global_support_labels)
                global_query_acc = compute_accuracy(query_global_logits_k, global_query_labels_k)
            else:
                global_support_acc = 0
                global_query_acc = 0

            # Add temporal-aware metrics
            additional_metrics = {
                "global_support_acc": global_support_acc, 
                "global_query_acc": global_query_acc
            }
            
            if consistency_scores is not None:
                additional_metrics["avg_consistency"] = consistency_scores.mean().item()
            if uncertainty_scores is not None:
                additional_metrics["avg_uncertainty"] = uncertainty_scores.mean().item()
        else:
            additional_metrics = {
                "global_support_acc": None, 
                "global_query_acc": None,
                "avg_consistency": 0,
                "avg_uncertainty": 0
            }

        return additional_metrics

    def visual_debug(self, similarity_matrix=None, support_global_logits=None, query_global_logits=None, 
                   videodataset=None, support_labels=None, target_labels=None, batch_class_list=None, 
                   support_set=None, target_set=None, disc_prob=None, unknown_labels=None, 
                   consistency_scores=None, uncertainty_scores=None):
        try:
            import cv2
            import imageio
        except ImportError:
            print("Warning: cv2 and/or imageio not available, skipping visual debug")
            return
            
        import os

        # Save support set gif
        support_set = support_set.reshape(-1, self.seq_len, 3, 224, 224)
        support_set = support_set[torch.argsort(support_labels)]
        support_set = support_set.reshape(self.way, self.shot, self.seq_len, 3, 224, 224)
        concatenated_frames = []
        support_classes = [videodataset.class_folders[int(batch_class_list[i])] for i in range(self.way)]

        for i in range(self.way):
            for j in range(self.shot):
                frames = []
                for k in range(self.seq_len):
                    frame_rgb = support_set[i, j, k].cpu().numpy().transpose(1, 2, 0)
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    frames.append(frame_rgb)
                concatenated_frames.append(frames)

        concatenated_frames = np.stack([np.stack(x) for x in concatenated_frames])
        concatenated_frames = concatenated_frames.reshape(self.way, self.shot, self.seq_len, 224, 224, 3)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        os.makedirs(f'visual_debug/{self.debug_samples_counter}', exist_ok=True)
        imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/ss.gif', concatenated_frames, duration=250, loop=0)
        with open(f'visual_debug/{self.debug_samples_counter}/ss.txt', 'w') as f:
            for item in support_classes:
                f.write("%s\n" % item)

        # Check predictions with TAOSAR-specific analysis
        good_closed = similarity_matrix.argmax(dim=-1) == target_labels
        if disc_prob is None:
            accept_score = similarity_matrix.max(dim=-1).values
        else:
            accept_score = 1 - disc_prob.squeeze(1)  # Convert rejection to acceptance
        true_open = target_labels != -1
        pred_open = accept_score > 0.5

        # Save queries gif with temporal analysis
        target_set = target_set.reshape(self.way, self.query_per_class, self.seq_len, 3, 224, 224)
        unknown_counter = 0
        query_labels = []
        for i in range(self.way):
            for j in range(self.query_per_class):
                if target_labels[i*self.query_per_class+j] == -1:
                    query_label = videodataset.class_folders[int(unknown_labels[i*self.query_per_class+j])]
                    unknown_counter += 1
                else:
                    query_label = videodataset.class_folders[int(batch_class_list[target_labels[i*self.query_per_class+j]])]
                query_labels.append(query_label)
                concatenated_frame = []
                for k in range(self.seq_len):
                    frame_rgb = target_set[i, j, k].cpu().numpy().transpose(1, 2, 0)
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    concatenated_frame.append(frame_rgb)
                
                # Determine classification result
                cur = i*self.query_per_class+j
                res = ""
                if true_open[cur] and pred_open[cur] and good_closed[cur]:
                    res = "TP"
                elif not true_open[cur] and not pred_open[cur]:
                    res = "TN"
                elif true_open[cur] and not pred_open[cur]:
                    res = "FN"
                elif not true_open[cur] and pred_open[cur]:
                    res = "FP"
                
                # Include temporal metrics in filename
                consistency_val = consistency_scores[cur].item() if consistency_scores is not None else 0
                uncertainty_val = uncertainty_scores[cur].item() if uncertainty_scores is not None else 0
                filename = f'{res}_acc{accept_score[cur]:.3f}_cons{consistency_val:.3f}_unc{uncertainty_val:.3f}_{query_label}.gif'
                imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/{filename}', concatenated_frame, duration=250, loop=0)

        self.debug_samples_counter += 1
