from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SAFSAR(nn.Module):
    def __init__(self, processor_name, model_name, hidden_size=768, num_layers_mm=2, num_heads=8, 
                 intermediate_size=3072, num_layers_task=1, n_train_classes=1, use_l2_loss=False,
                 use_textual_embedding=True):
        super(SAFSAR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.model = AutoModelForVideoClassification.from_pretrained(model_name, output_hidden_states=True)

        # Freeze patch_embeddings
        for param in self.model.videomae.embeddings.parameters():
            param.requires_grad = False

        self.mm_fusion_module = self._build_transformer(hidden_size, num_layers_mm, num_heads, intermediate_size)
        self.task_specific_learning_module = self._build_transformer(hidden_size, num_layers_task, num_heads, intermediate_size, batch_first=True)

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        if use_l2_loss:
            self.global_classification_layer = nn.Linear(hidden_size, n_train_classes)
        self.use_l2_loss = use_l2_loss
        self.use_textual_embedding = use_textual_embedding

    def _build_transformer(self, hidden_size, num_layers, num_heads, intermediate_size):
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size)
        return TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, support_set, support_labels, target_set, target_labels, class_name_embeddings, batch_class_list, dataset):

        # Generate support set prototypes
        support_set = support_set.reshape(dataset.way*dataset.shot, dataset.seq_len, 224, 3, 224)
        support_set = support_set.permute(0, 1, 3, 4, 2)
        if dataset.seq_len == 8:
            inputs = {"pixel_values": support_set.repeat_interleave(2, dim=1).cuda()}
        outputs = self.model(**inputs)
        support_features = outputs.hidden_states[-1].mean(dim=1)
        support_features = self.model.fc_norm(support_features).reshape(dataset.way, dataset.shot, -1).mean(dim=1)
        # add textual features
        if self.use_textual_embedding:
            textual_embeddings = [class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
            raw_support_mm_features = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(support_features, textual_embeddings)]
            support_mm_features = [self.mm_fusion_module(emb)[0] for emb in raw_support_mm_features]
            support_mm_features = torch.stack(support_mm_features)
        else:
            support_mm_features = support_features

        # Generate query prototypes
        target_set = target_set.reshape(dataset.query_per_class*dataset.way, dataset.seq_len, 224, 3, 224)
        target_set = target_set.permute(0, 1, 3, 4, 2)
        if dataset.seq_len == 8:
            inputs = {"pixel_values": target_set.repeat_interleave(2, dim=1).cuda()}
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_features = outputs.hidden_states[-1].mean(dim=1)
        query_features = self.model.fc_norm(query_features)

        # Repeat embeddings for each query
        support_mm_features = support_mm_features.unsqueeze(0).repeat(dataset.way*dataset.query_per_class, 1, 1)
        combined_features = torch.cat((query_features.unsqueeze(1), support_mm_features), dim=1)
        combined_features = self.task_specific_learning_module(combined_features)
        query_features_aug, support_mm_features_aug = combined_features.split([1, dataset.way], dim=1)

        similarity_matrix = self.cosine_similarity(
            query_features_aug.expand(-1, support_mm_features_aug.size(1), -1),
            support_mm_features_aug
        )

        # Compute global logits
        if self.use_l2_loss:
            support_global_logits = self.global_classification_layer(support_features)
            query_global_logits = self.global_classification_layer(query_features)
        else:
            support_global_logits = None
            query_global_logits = None

        return similarity_matrix, support_global_logits, query_global_logits