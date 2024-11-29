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
        self.task_specific_learning_module = self._build_transformer(hidden_size, num_layers_task, num_heads, intermediate_size)

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
        video_embeddings = outputs.hidden_states[-1].mean(dim=1)
        video_embeddings = self.model.fc_norm(video_embeddings).reshape(dataset.way, dataset.shot, -1).mean(dim=1)
        # add textual features
        if self.use_textual_embedding:
            textual_embeddings = [class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
            raw_mm_embeddings = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(video_embeddings, textual_embeddings)]
            # add batch dimension for transformer, remove it after, get only first element (agumented support)
            mm_embeddings = [self.mm_fusion_module(emb.unsqueeze(0)).squeeze(0)[0] for emb in raw_mm_embeddings]
            mm_embeddings = torch.stack(mm_embeddings)
        else:
            mm_embeddings = video_embeddings

        # Generate query prototypes
        target_set = target_set.reshape(dataset.query_per_class*dataset.way, dataset.seq_len, 224, 3, 224)
        target_set = target_set.permute(0, 1, 3, 4, 2)
        if dataset.seq_len == 8:
            inputs = {"pixel_values": target_set.repeat_interleave(2, dim=1).cuda()}
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_embeddings = outputs.hidden_states[-1].mean(dim=1)
        query_embeddings = self.model.fc_norm(query_embeddings)

        # Repeat embeddings for each query
        mm_embeddings = mm_embeddings.unsqueeze(0).repeat(dataset.way*dataset.query_per_class, 1, 1)
        embeddings = torch.cat((query_embeddings.unsqueeze(1), mm_embeddings), dim=1)
        embeddings = self.task_specific_learning_module(embeddings)
        query_embeddings_aug, support_embeddings = embeddings.split([1, 5], dim=1)

        similarity_matrix = self.cosine_similarity(
            query_embeddings_aug.expand(-1, support_embeddings.size(1), -1),
            support_embeddings
        )

        # Compute global logits
        if self.use_l2_loss:
            support_global_logits = self.global_classification_layer(support_embeddings)
            query_global_logits = self.global_classification_layer(query_embeddings)
        else:
            support_global_logits = None
            query_global_logits = None

        return similarity_matrix, support_global_logits, query_global_logits