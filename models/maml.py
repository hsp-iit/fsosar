import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy
from utils import compute_accuracy, BinaryClassificationModelSAFSAR
import wandb
import numpy as np
import torchvision.models as models

class MAML(nn.Module):
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(MAML, self).__init__()
        
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]
        
        # Feature extractor backbone
        self.backbone_type = config.get("backbone", "resnet18")
        self.feature_extractor = self._build_feature_extractor()
        
        # Distribute feature extractor for 5-shot training
        if dp:
            self.feature_extractor = torch.nn.DataParallel(self.feature_extractor)
            
        # Get feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()
        
        # Temporal aggregation for video frames
        self.temporal_aggregator = self._build_transformer(
            self.feature_dim,
            config.get("num_layers_temporal", 2),
            config.get("num_heads", 8),
            config.get("intermediate_size", 2048)
        )
        
        # Meta-learnable classifier (this will be adapted during meta-learning)
        self.classifier = nn.Linear(self.feature_dim, self.way)
        
        # MAML specific parameters
        self.inner_lr = config.get("inner_lr", 0.01)
        self.inner_steps = config.get("inner_steps", 5)
        self.first_order = config.get("first_order", False)
        
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
            # Remove the final classification layer
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
            batch_first=True
        )
        return TransformerEncoder(encoder_layer, num_layers=num_layers)

    def extract_features(self, videos):
        """Extract features from videos by processing each frame independently then aggregating"""
        batch_size, seq_len, channels, height, width = videos.shape
        
        # Reshape to process all frames together: (batch_size * seq_len, channels, height, width)
        frames = videos.view(-1, channels, height, width)
        
        # Extract frame features using 2D CNN
        frame_features = self.feature_extractor(frames)  # (batch_size * seq_len, feature_dim, h, w)
        
        # Global average pooling to get fixed-size features
        frame_features = F.adaptive_avg_pool2d(frame_features, (1, 1))
        frame_features = frame_features.squeeze(-1).squeeze(-1)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to video format
        frame_features = frame_features.view(batch_size, seq_len, self.feature_dim)
        
        # Apply temporal aggregation using transformer
        aggregated_features = self.temporal_aggregator(frame_features)
        
        # Use mean pooling across temporal dimension to get video-level features
        video_features = aggregated_features.mean(dim=1)  # (batch_size, feature_dim)
        
        return video_features

    def adapt_classifier(self, support_features, support_labels):
        """Adapt the classifier using MAML inner loop updates"""
        # Clone classifier parameters for adaptation
        adapted_params = {}
        for name, param in self.classifier.named_parameters():
            adapted_params[name] = param.clone()
        
        # Inner loop adaptation
        for step in range(self.inner_steps):
            # Forward pass with current adapted parameters
            logits = F.linear(support_features, adapted_params['weight'], adapted_params['bias'])
            
            # Compute loss
            loss = F.cross_entropy(logits, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                adapted_params.values(), 
                create_graph=not self.first_order,
                retain_graph=True,
                allow_unused=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params

    def forward(self, support_set, support_labels, target_set, batch_class_list=None, precomputed_context_features=None):
        
        # Extract features from support and query sets
        support_set = support_set.reshape(-1, self.seq_len, 3, 224, 224)
        support_features = self.extract_features(support_set)
        
        # Process target set
        if len(target_set.shape) == 4:  # n_q, seq_len, 3, 224, 224
            n_queries = int(target_set.shape[0] / self.seq_len)
        elif len(target_set.shape) == 5:
            n_queries = target_set.shape[0]
        target_set = target_set.reshape(n_queries, self.seq_len, 3, 224, 224)
        
        query_features = self.extract_features(target_set)
        
        # MAML adaptation: adapt classifier on support set
        adapted_params = self.adapt_classifier(support_features, support_labels)
        
        # Compute similarity matrix using adapted classifier
        similarity_matrix = F.linear(query_features, adapted_params['weight'], adapted_params['bias'])
        
        # Add garbage class if using open-set
        if self.gc:
            # Compute similarity to garbage prototype
            garbage_similarity = F.cosine_similarity(
                query_features.unsqueeze(1), 
                self.garbage_prototype.unsqueeze(0), 
                dim=-1
            )
            similarity_matrix = torch.cat([similarity_matrix, garbage_similarity], dim=-1)
        
        # Discriminator for open-set recognition
        if self.disc:
            # Use the adapted classifier predictions to compute discriminator input
            predictions = torch.argmax(similarity_matrix, dim=-1)
            # Create prototype representations from support features
            prototypes = []
            for c in support_labels.unique():
                prototypes.append(support_features[support_labels == c].mean(dim=0))
            prototypes = torch.stack(prototypes)
            
            # Compute differences for discriminator
            query_expanded = query_features.unsqueeze(1).expand(-1, prototypes.size(0), -1)
            prototype_expanded = prototypes.unsqueeze(0).expand(query_features.size(0), -1, -1)
            all_differences = query_expanded - prototype_expanded
            
            # Use the difference corresponding to the predicted class
            best_diffs = all_differences[torch.arange(len(all_differences)), predictions % self.way]
            disc_prob = self.discriminator(best_diffs)
        else:
            disc_prob = None

        # Compute global logits for L2 loss
        if self.use_l2_loss:
            support_global_logits = self.global_classification_layer(support_features)
            query_global_logits = self.global_classification_layer(query_features)
        else:
            support_global_logits = 0.
            query_global_logits = 0.

        return {
            "similarity_matrix": similarity_matrix,
            "support_global_logits": support_global_logits,
            "query_global_logits": query_global_logits,
            "disc_prob": disc_prob
        }

    def compute_known_losses(self, similarity_matrix, support_global_logits, query_global_logits,
                           true_target_labels=None, target_labels=None, support_labels=None, batch_class_list=None, disc_prob=None):
        true_target_labels = target_labels
        known_indices = true_target_labels != -1
        similarity_matrix_k = similarity_matrix[known_indices]
        known_true_target_labels = true_target_labels[known_indices]
        
        # Compute L1 loss (meta-learning loss)
        l1_loss = F.cross_entropy(similarity_matrix_k, known_true_target_labels)

        # Compute L2 loss (global classification loss)
        if self.use_l2_loss:
            global_support_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
            if self.gc:
                known_indices = target_labels != (self.way-1)
            global_query_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[target_labels[known_indices]]]).cuda()
            l2_loss_support = F.cross_entropy(support_global_logits, global_support_labels)
            l2_loss_query = F.cross_entropy(query_global_logits[known_indices], global_query_labels)
            l2_loss = l2_loss_support + l2_loss_query
        else:
            l2_loss = 0.

        self.debug_data = {
            "similarity_matrix": wandb.Table(columns=list(range(self.way)), data=similarity_matrix.detach().cpu().numpy().tolist()),
            "true_target_labels": wandb.Table(columns=[0], data=target_labels.detach().cpu().numpy()[..., None]),
            "similarity_matrix_mean": similarity_matrix.mean().item()
        }
        
        return {"l1_loss": l1_loss, "l2_loss": self.alpha * l2_loss}

    def get_debug_data(self):
        return self.debug_data

    def compute_additional_metrics(self, similarity_matrix, support_global_logits, query_global_logits, support_labels, target_labels, batch_class_list, disc_prob=None):
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
        else:
            global_support_acc = None
            global_query_acc = None

        return {"global_support_acc": global_support_acc, "global_query_acc": global_query_acc}

    def visual_debug(self, similarity_matrix=None, support_global_logits=None, query_global_logits=None, videodataset=None, support_labels=None, target_labels=None, batch_class_list=None, support_set=None, target_set=None, disc_prob=None, unknown_labels=None):
        import cv2
        import imageio
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
                    frame_rgb = support_set[i, j, k].cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
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

        # Check predictions
        good_closed = similarity_matrix.argmax(dim=-1) == target_labels
        if disc_prob is None:
            accept_score = similarity_matrix.max(dim=-1).values
        else:
            accept_score = disc_prob.detach().cpu().numpy()
        true_open = target_labels != -1
        pred_open = accept_score > 0.5

        # Save queries gif
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
                    frame_rgb = target_set[i, j, k].cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    concatenated_frame.append(frame_rgb)
                # Determine if TP TN FP FN
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
                imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/{res}_{accept_score[cur]}_{query_label}.gif', concatenated_frame, duration=250, loop=0)

        self.debug_samples_counter += 1
