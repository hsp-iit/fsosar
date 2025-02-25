import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoImageProcessor, AutoModelForVideoClassification, BertTokenizer, BertModel
from utils import compute_accuracy, OpenSetLoss
import wandb
import numpy as np
from utils import BinaryClassificationModelSAFSAR

class SAFSAR(nn.Module):
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(SAFSAR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(config["processor_name"])
        self.model = AutoModelForVideoClassification.from_pretrained(config["mm_model_name"], output_hidden_states=True)
        # Freeze patch_embeddings
        for param in self.model.videomae.embeddings.parameters():
            param.requires_grad = False
        # Distribute feature extractor for 5-shot training
        if dp:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.module
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]

        self.mm_fusion_module = self._build_transformer(config["hidden_size"],
                                                        config["num_layers_mm"],
                                                        config["num_heads"],
                                                        config["intermediate_size"])  # We do not use batch here
        self.task_specific_learning_module = self._build_transformer(config["hidden_size"],
                                                                     config["num_layers_task"],
                                                                     config["num_heads"],
                                                                     config["intermediate_size"],
                                                                     batch_first=True)  # We pass batch as first dimension
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.use_l2_loss = config["use_l2_loss"]
        if self.use_l2_loss:
            self.global_classification_layer = nn.Linear(config["hidden_size"], config["n_train_classes"])

        self.use_textual_embedding = config["use_textual_embedding"]
        self.class_name_embeddings = self.get_textual_embeddings(config["classes_names"])
        self.alpha = config["alpha"]
        self.debug_samples_counter = 0

        if gc:
            self.garbage_prototype = nn.Parameter(torch.randn((1, 768))).cuda()
        elif disc:
            self.discriminator = BinaryClassificationModelSAFSAR(768).cuda()
        self.gc = gc
        self.disc = disc

    # Override methods to avoid using l2 loss during evaluation
    def set_train(self):
        self.use_l2_loss = True

    def set_eval(self):
        # During training, l2 loss is useless
        self.use_l2_loss = False

    def _build_transformer(self, hidden_size, num_layers, num_heads, intermediate_size, batch_first=False):
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                dim_feedforward=intermediate_size, batch_first=batch_first)
        return TransformerEncoder(encoder_layer, num_layers=num_layers)

    def get_textual_embeddings(self, classes_names):
        # Get features of class names with BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.cuda()
        bert_model.eval()
        class_name_embeddings = []
        for class_name in classes_names:
            plain_class_name = class_name.replace('_', ' ')
            inputs = tokenizer(plain_class_name, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            class_name_embeddings.append(outputs.last_hidden_state.squeeze(0))
        bert_model = None
        tokenizer = None
        return class_name_embeddings

    def forward(self, support_set, support_labels, target_set, batch_class_list=None, precomputed_context_features=None):

        # Generate support set prototypes
        support_set = support_set.reshape(-1, self.seq_len, 224, 3, 224)
        support_set = support_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": support_set.repeat_interleave(2, dim=1).cuda()}
        outputs = self.model(**inputs)
        support_features = outputs.hidden_states[-1].mean(dim=1)
        support_features = self.model.fc_norm(support_features)
        support_features_mean = []
        for c in support_labels.unique():
            support_features_mean.append(support_features[support_labels == c].mean(dim=0))
        support_features_mean = torch.stack(support_features_mean)
        # add textual features
        if self.use_textual_embedding:
            textual_embeddings = [self.class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
            raw_support_mm_features = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(support_features_mean, textual_embeddings)]
            support_mm_features = [self.mm_fusion_module(emb)[0] for emb in raw_support_mm_features]
            support_mm_features = torch.stack(support_mm_features)
        else:
            support_mm_features = support_features_mean

        # Add unknown class if GC
        if self.gc:
            support_mm_features = torch.cat((support_mm_features, self.garbage_prototype), dim=0)

        # Generate query prototypes
        if len(target_set.shape) == 4:  # n_q, seq_len, 224, 3, 224
            n_queries = int(target_set.shape[0] / self.seq_len)
        elif len(target_set.shape) == 5:
            n_queries = target_set.shape[0]
        target_set = target_set.reshape(n_queries, self.seq_len, 224, 3, 224)
        target_set = target_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": target_set.repeat_interleave(2, dim=1).cuda()}
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_features = outputs.hidden_states[-1].mean(dim=1)
        query_features = self.model.fc_norm(query_features)

        # Repeat embeddings for each query
        support_mm_features = support_mm_features.unsqueeze(0).repeat(n_queries, 1, 1)
        combined_features = torch.cat((query_features.unsqueeze(1), support_mm_features), dim=1)
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
            support_global_logits = self.global_classification_layer(support_features)
            query_global_logits = self.global_classification_layer(query_features)
        else:
            support_global_logits = 0. # We need to return something
            query_global_logits = 0. # We need to return something

        return {"similarity_matrix": similarity_matrix,
                "support_global_logits": support_global_logits,
                "query_global_logits": query_global_logits,
                "disc_prob": disc_prob}

    def compute_known_losses(self, similarity_matrix, support_global_logits, query_global_logits,
                           true_target_labels=None, target_labels=None, support_labels=None, batch_class_list=None, disc_prob=None):
        true_target_labels = target_labels  # if ordered in model. this is is the best way
        known_indices = true_target_labels != -1
        similarity_matrix_k = similarity_matrix[known_indices]
        known_true_target_labels = true_target_labels[known_indices]
        # Compute L1 loss
        l1_loss = torch.nn.functional.cross_entropy(similarity_matrix_k, known_true_target_labels)

        # Compute L2 loss
        if self.use_l2_loss:
            global_support_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
            if self.gc:  # here we need to get rid of the garbage class
                known_indices = target_labels != (self.way-1)
            global_query_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[target_labels[known_indices]]]).cuda()
            # Make support label one hot to apply them correctly
            l2_loss_support = torch.nn.functional.cross_entropy(support_global_logits, global_support_labels)
            l2_loss_query = torch.nn.functional.cross_entropy(query_global_logits[known_indices], global_query_labels)
            l2_loss = l2_loss_support + l2_loss_query
        else:
            l2_loss = 0. # We need to return something

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
                global_support_acc = 0 # We need to return something
                global_query_acc = 0 # We need to return something
        else:
            global_support_acc = None
            global_query_acc = None

        return {"global_support_acc": global_support_acc, "global_query_acc": global_query_acc}

    def visual_debug(self, similarity_matrix=None, support_global_logits=None, query_global_logits=None, videodataset=None, support_labels=None, target_labels=None, batch_class_list=None, support_set=None, target_set=None):
        import cv2
        import imageio
        import os

        # Save support set gif
        support_set = support_set.reshape(self.way, self.shot, self.seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        concatenated_frames = []
        support_classes = []
        for i in range(self.way):
            for j in range(self.shot):
                support_classes.append(videodataset.class_folders[int(batch_class_list[support_labels[i*self.shot+j]])])
                frames = []
                for k in support_set[i, j]:
                    frame_rgb = k.cpu().numpy()
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    frames.append(frame_rgb)
                concatenated_frames.append(frames)
        concatenated_frames = np.concatenate(concatenated_frames, axis=1)  # 3 for vertival, 2 for horizontal
        os.makedirs(f'visual_debug/{self.debug_samples_counter}', exist_ok=True)
        imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/ss.gif', concatenated_frames, duration=250, loop=0)
        with open(f'visual_debug/{self.debug_samples_counter}/ss.txt', 'w') as f:
            for item in support_classes:
                f.write("%s\n" % item)

        # Save queries gif
        target_set = target_set.reshape(self.way, self.query_per_class, self.seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        # query_labels = torch.argsort(support_labels)[target_labels]
        # query_labels[target_labels == -1] = -1
        unknown_counter = 0
        query_labels = []
        for i in range(self.way):
            for j in range(self.query_per_class):
                if target_labels[i*self.query_per_class+j] == -1:
                    query_label = f"unknown_{unknown_counter}"
                    unknown_counter += 1
                else:
                    query_label = videodataset.class_folders[int(batch_class_list[target_labels[i*self.query_per_class+j]])]
                query_labels.append(query_label)
                concatenated_frame = []
                for k in target_set[i, j]:
                    frame_rgb = k.cpu().numpy()
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    concatenated_frame.append(frame_rgb)
                imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/{i}_{query_label}.gif', concatenated_frame, duration=250, loop=0)

        # Save results
        true_targets = torch.argsort(support_labels)[target_labels]
        true_targets[target_labels == -1] = -1
        similarity_matrix = similarity_matrix.detach().cpu().numpy()
        with open(f'visual_debug/{self.debug_samples_counter}/similarity_matrix.txt', 'w') as f:
            for item in support_classes:
                f.write("%s " % item)
            f.write("\n")
            lazy_counter = 0
            for item in similarity_matrix:
                f.write(f"%s\t{true_targets[lazy_counter]} {query_labels[lazy_counter]}\n" % item)
                lazy_counter += 1
        self.debug_samples_counter += 1
