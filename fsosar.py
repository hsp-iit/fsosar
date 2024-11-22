from transformers import AutoImageProcessor, AutoModelForVideoClassification, BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FSOSAR(nn.Module):
    def __init__(self, processor_name, model_name, bert_name, hidden_size=768, num_layers_mm=2, num_heads=8, intermediate_size=3072, num_layers_task=1):
        super(FSOSAR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.model = AutoModelForVideoClassification.from_pretrained(model_name, output_hidden_states=True)
        self.model.cuda()
        self.model.train()

        # Freeze patch_embeddings and first 6 layers
        for param in self.model.videomae.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.videomae.encoder.layer[:6].parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.bert_model = BertModel.from_pretrained(bert_name)
        self.bert_model.cuda()
        self.bert_model.eval()

        self.mm_fusion_module = self._build_transformer(hidden_size, num_layers_mm, num_heads, intermediate_size)
        self.task_specific_learning_module = self._build_transformer(hidden_size, num_layers_task, num_heads, intermediate_size)

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def _build_transformer(self, hidden_size, num_layers, num_heads, intermediate_size):
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size)
        return TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, support_set, support_labels, target_set, target_labels, class_name_embeddings, batch_class_list, dataset):
        # Reorder support set w.r.t. support labels
        support_set = support_set.reshape(dataset.way*dataset.shot, dataset.seq_len, 3, 224, 224)
        support_set = support_set[support_labels.argsort()]
        support_labels = support_labels[support_labels.argsort()]

        # # Visualize support
        # support_set_flat_labels = [dataloader.class_folders[int(x.item())] for x in elem['batch_class_list'][support_labels]]
        # support_set_flat = support_set.reshape(dataset.way, dataset.shot, dataset.seq_len, 3, 224, 224)
        # counter = 0
        # for k in support_set_flat:
        #     for n in k:
        #         print(support_set_flat_labels[counter])
        #         for i in n:
        #             cv2.imshow("image", i.permute(1, 2, 0).numpy())
        #             cv2.waitKey(0)
        #         counter += 1

        # Generate support set prototypes
        support_set = support_set.reshape(dataset.way*dataset.shot*dataset.seq_len, 3, 224, 224)
        inputs = self.processor(torch.unbind(support_set), return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].reshape(dataset.way*dataset.shot, dataset.seq_len, 3, 224, 224)
        if dataset.seq_len == 8:
            inputs['pixel_values'] = inputs['pixel_values'].repeat_interleave(2, dim=1)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        video_embeddings = outputs.hidden_states[-1].mean(dim=1)
        video_embeddings = self.model.fc_norm(video_embeddings).reshape(dataset.way, dataset.shot, -1).mean(dim=1)

        textual_embeddings = [class_name_embeddings[x] for x in batch_class_list.long()]
        raw_mm_embeddings = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(video_embeddings, textual_embeddings)]

        mm_embeddings = [self.mm_fusion_module(emb.unsqueeze(0)).squeeze(0)[0] for emb in raw_mm_embeddings]
        mm_embeddings = torch.stack(mm_embeddings)

        # Generate query prototypes
        target_set = target_set.reshape(dataset.query_per_class*dataset.way*dataset.seq_len, 3, 224, 224)
        inputs = self.processor(torch.unbind(target_set), return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].reshape(dataset.way*dataset.query_per_class, dataset.seq_len, 3, 224, 224)
        if dataset.seq_len == 8:
            inputs['pixel_values'] = inputs['pixel_values'].repeat_interleave(2, dim=1)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_embeddings = outputs.hidden_states[-1].mean(dim=1)
        query_embeddings = self.model.fc_norm(query_embeddings)

        mm_embeddings = mm_embeddings.unsqueeze(0).repeat(5, 1, 1)
        embeddings = torch.cat((query_embeddings.unsqueeze(1), mm_embeddings), dim=1)
        embeddings = self.task_specific_learning_module(embeddings)
        query_embeddings, support_embeddings = embeddings.split([1, 5], dim=1)

        similarity_matrix = torch.zeros(query_embeddings.size(0), support_embeddings.size(1)).cuda()
        for i in range(query_embeddings.size(0)):
            for j in range(support_embeddings.size(1)):
                similarity_matrix[i, j] = self.cosine_similarity(query_embeddings[i], support_embeddings[i, j])

        scores = self.softmax(similarity_matrix)
        return scores