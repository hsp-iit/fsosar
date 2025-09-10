import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import clip
from utils import compute_accuracy, OpenSetLoss
import wandb
import numpy as np
from utils import BinaryClassificationModelSAFSAR
import copy

class ActionCLIP(nn.Module):
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(ActionCLIP, self).__init__()
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(config["clip_model"], device="cuda")
        
        # Convert CLIP model to float32 to avoid dtype issues
        self.clip_model.float()
        
        # Freeze CLIP parameters (can be made configurable)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Distribute feature extractor for 5-shot training
        if dp:
            self.clip_model = torch.nn.DataParallel(self.clip_model)
            
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]
        
        # Get CLIP feature dimension
        self.clip_dim = self.clip_model.visual.output_dim if not dp else self.clip_model.module.visual.output_dim
        
        # Temporal fusion module for combining frame features
        self.temporal_fusion_module = self._build_transformer(self.clip_dim,
                                                              config["num_layers_temporal"],
                                                              config["num_heads"],
                                                              config["intermediate_size"])
        
        # Task-specific learning module
        self.task_specific_learning_module = self._build_transformer(self.clip_dim,
                                                                     config["num_layers_task"],
                                                                     config["num_heads"],
                                                                     config["intermediate_size"],
                                                                     batch_first=True)
        
        # Similarity computation
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        
        # Global classification layer for L2 loss
        self.use_l2_loss = config["use_l2_loss"]
        if self.use_l2_loss:
            self.global_classification_layer = nn.Linear(self.clip_dim, config["n_train_classes"])
            
        # Text embeddings for class names
        self.use_textual_embedding = config["use_textual_embedding"]
        self.class_name_embeddings = self.get_textual_embeddings(config["classes_names"])
        
        # Multimodal fusion module
        if self.use_textual_embedding:
            self.mm_fusion_module = self._build_transformer(self.clip_dim,
                                                            config["num_layers_mm"],
                                                            config["num_heads"],
                                                            config["intermediate_size"])
        
        self.alpha = config["alpha"]
        self.debug_samples_counter = 0
        
        # Open set components
        if gc:
            self.garbage_prototype = nn.Parameter(torch.randn((1, self.clip_dim))).cuda()
        self.discriminator = BinaryClassificationModelSAFSAR(self.clip_dim).cuda()
        self.gc = gc
        self.disc = disc

    def set_train(self):
        self.use_l2_loss = True

    def set_eval(self):
        self.use_l2_loss = False

    def _build_transformer(self, hidden_size, num_layers, num_heads, intermediate_size, batch_first=False):
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                dim_feedforward=intermediate_size, batch_first=batch_first)
        return TransformerEncoder(encoder_layer, num_layers=num_layers)

    def get_textual_embeddings(self, classes_names):
        """Get CLIP text embeddings for class names"""
        class_name_embeddings = []
        with torch.no_grad():
            for class_name in classes_names:
                plain_class_name = class_name.replace('_', ' ')
                # Truncate long descriptions and create simple text prompt
                if len(plain_class_name.split()) > 10:
                    # Take only the first few words for very long descriptions
                    plain_class_name = ' '.join(plain_class_name.split()[:10])
                
                # Create simple text prompt for action recognition
                text_prompt = f"a video of {plain_class_name}"
                
                # Truncate the prompt if it's still too long for CLIP (77 tokens max)
                try:
                    text_tokens = clip.tokenize([text_prompt], truncate=True).cuda()
                except RuntimeError:
                    # Fallback: use just the class name if prompt is still too long
                    text_prompt = plain_class_name
                    text_tokens = clip.tokenize([text_prompt], truncate=True).cuda()
                
                if hasattr(self.clip_model, 'module'):
                    text_features = self.clip_model.module.encode_text(text_tokens)
                else:
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                # Normalize text features and ensure float32
                text_features = text_features.float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                class_name_embeddings.append(text_features.squeeze(0))
                
        return class_name_embeddings

    def extract_visual_features(self, videos):
        """Extract CLIP visual features from video frames"""
        batch_size, seq_len, channels, height, width = videos.shape
        
        # Reshape to process all frames together
        frames = videos.view(-1, channels, height, width)
        
        # Ensure frames are in float32
        frames = frames.float()
        
        # Extract CLIP visual features
        with torch.no_grad():
            if hasattr(self.clip_model, 'module'):
                visual_features = self.clip_model.module.encode_image(frames)
            else:
                visual_features = self.clip_model.encode_image(frames)
            
            # Normalize visual features and ensure float32
            visual_features = visual_features.float()
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        
        # Reshape back to video format
        visual_features = visual_features.view(batch_size, seq_len, self.clip_dim)
        
        return visual_features

    def forward(self, support_set, support_labels, target_set, batch_class_list=None, precomputed_context_features=None):
        
        # Generate support set prototypes
        support_set = support_set.reshape(-1, self.seq_len, 3, 224, 224)
        support_features = self.extract_visual_features(support_set)
        
        # Apply temporal fusion to combine frame features
        support_features_fused = []
        for i in range(support_features.shape[0]):
            # Apply temporal transformer to each video
            temp_features = self.temporal_fusion_module(support_features[i].unsqueeze(0))
            # Use mean pooling to get a single representation per video
            support_features_fused.append(temp_features.mean(dim=1).squeeze(0))
        support_features_fused = torch.stack(support_features_fused)
        
        # Compute support prototypes (mean of each class)
        support_features_mean = []
        for c in support_labels.unique():
            support_features_mean.append(support_features_fused[support_labels == c].mean(dim=0))
        support_features_mean = torch.stack(support_features_mean)
        
        # Add textual features if enabled
        if self.use_textual_embedding:
            textual_embeddings = [self.class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
            raw_support_mm_features = [torch.cat((v.unsqueeze(0), t.unsqueeze(0))) for v, t in zip(support_features_mean, textual_embeddings)]
            support_mm_features = [self.mm_fusion_module(emb)[0] for emb in raw_support_mm_features]
            support_mm_features = torch.stack(support_mm_features)
        else:
            support_mm_features = support_features_mean

        # Add unknown class if GC
        if self.gc:
            support_mm_features = torch.cat((support_mm_features, self.garbage_prototype), dim=0)

        # Generate query features
        if len(target_set.shape) == 4:  # n_q, seq_len, 3, 224, 224
            n_queries = int(target_set.shape[0] / self.seq_len)
        elif len(target_set.shape) == 5:
            n_queries = target_set.shape[0]
        target_set = target_set.reshape(n_queries, self.seq_len, 3, 224, 224)
        
        query_features = self.extract_visual_features(target_set)
        
        # Apply temporal fusion to query features
        query_features_fused = []
        for i in range(query_features.shape[0]):
            temp_features = self.temporal_fusion_module(query_features[i].unsqueeze(0))
            query_features_fused.append(temp_features.mean(dim=1).squeeze(0))
        query_features_fused = torch.stack(query_features_fused)

        # Repeat embeddings for each query
        support_mm_features = support_mm_features.unsqueeze(0).repeat(n_queries, 1, 1)
        combined_features = torch.cat((query_features_fused.unsqueeze(1), support_mm_features), dim=1)
        combined_features = self.task_specific_learning_module(combined_features)
        query_features_aug, support_mm_features_aug = combined_features.split([1, self.way], dim=1)

        similarity_matrix = self.cosine_similarity(
            query_features_aug.expand(-1, support_mm_features_aug.size(1), -1),
            support_mm_features_aug
        )

        # If discriminator, use it
        if self.disc:
            all_prototypes_differences = query_features_aug.expand(-1, support_mm_features_aug.size(1), -1) - support_mm_features_aug
            predictions = torch.argmax(similarity_matrix, dim=-1)
            best_diffs = all_prototypes_differences[torch.arange(len(all_prototypes_differences)), predictions]
            disc_prob = self.discriminator(best_diffs)
        else:
            disc_prob = None

        # Compute global logits
        if self.use_l2_loss:
            support_global_logits = self.global_classification_layer(support_features_fused)
            query_global_logits = self.global_classification_layer(query_features_fused)
        else:
            support_global_logits = 0.
            query_global_logits = 0.

        return {"similarity_matrix": similarity_matrix,
                "support_global_logits": support_global_logits,
                "query_global_logits": query_global_logits,
                "disc_prob": disc_prob}

    def compute_known_losses(self, similarity_matrix, support_global_logits, query_global_logits,
                           true_target_labels=None, target_labels=None, support_labels=None, batch_class_list=None, disc_prob=None):
        true_target_labels = target_labels
        known_indices = true_target_labels != -1
        similarity_matrix_k = similarity_matrix[known_indices]
        known_true_target_labels = true_target_labels[known_indices]
        
        # Compute L1 loss
        l1_loss = torch.nn.functional.cross_entropy(similarity_matrix_k, known_true_target_labels)

        # Compute L2 loss
        if self.use_l2_loss:
            global_support_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
            if self.gc:
                known_indices = target_labels != (self.way-1)
            global_query_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[target_labels[known_indices]]]).cuda()
            l2_loss_support = torch.nn.functional.cross_entropy(support_global_logits, global_support_labels)
            l2_loss_query = torch.nn.functional.cross_entropy(query_global_logits[known_indices], global_query_labels)
            l2_loss = l2_loss_support + l2_loss_query
        else:
            l2_loss = 0.

        self.debug_data = {"similarity_matrix": wandb.Table(columns=list(range(self.way)), data=similarity_matrix.detach().cpu().numpy().tolist()),
                           "true_target_labels": wandb.Table(columns=[0], data=target_labels.detach().cpu().numpy()[..., None]),
                           "similarity_matrix_mean": similarity_matrix.mean().item()}
        return {"l1_loss": l1_loss, "l2_loss": self.alpha*l2_loss}

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
        support_set = support_set.reshape(self.way, self.shot, self.seq_len, 3, 224, 224).permute(0, 1, 2, 3, 4, 5)
        concatenated_frames = []
        support_classes = [videodataset.class_folders[int(batch_class_list[i])] for i in range(self.way)]

        for i in range(self.way):
            for j in range(self.shot):
                frames = []
                for k in support_set[i, j]:
                    frame_rgb = k.cpu().numpy()
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    frames.append(frame_rgb)
                concatenated_frames.append(frames)

        concatenated_frames = np.stack([np.stack(x) for x in concatenated_frames])
        concatenated_frames = concatenated_frames.reshape(self.way, self.shot, self.seq_len, 3, 224, 224)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        # Convert from CHW to HWC for saving
        concatenated_frames = concatenated_frames.transpose(0, 2, 3, 1)
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
                for k in target_set[i, j]:
                    frame_rgb = k.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
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
