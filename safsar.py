import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoImageProcessor, AutoModelForVideoClassification, BertTokenizer, BertModel
from utils import compute_accuracy, OpenSetLoss
from sklearn.metrics import roc_auc_score


class SAFSAR(nn.Module):
    def __init__(self, config):
        super(SAFSAR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(config["processor_name"])
        self.model = AutoModelForVideoClassification.from_pretrained(config["model_name"], output_hidden_states=True)
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]

        # Freeze patch_embeddings
        for param in self.model.videomae.embeddings.parameters():
            param.requires_grad = False

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
        self.open_set_loss = OpenSetLoss()

    # Override methods to avoid using l2 loss during evaluation
    def set_train(self):
        self.use_l2_loss = True

    def set_eval(self):
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

    def forward(self, support_set, support_labels, target_set, batch_class_list):

        # Generate support set prototypes
        support_set = support_set.reshape(self.way*self.shot, self.seq_len, 224, 3, 224)
        support_set = support_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": support_set.repeat_interleave(2, dim=1).cuda()}
        outputs = self.model(**inputs)
        support_features = outputs.hidden_states[-1].mean(dim=1)
        support_features = self.model.fc_norm(support_features).reshape(self.way, self.shot, -1).mean(dim=1)
        # add textual features
        if self.use_textual_embedding:
            textual_embeddings = [self.class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
            raw_support_mm_features = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(support_features, textual_embeddings)]
            support_mm_features = [self.mm_fusion_module(emb)[0] for emb in raw_support_mm_features]
            support_mm_features = torch.stack(support_mm_features)
        else:
            support_mm_features = support_features

        # Generate query prototypes
        target_set = target_set.reshape(self.query_per_class*self.way, self.seq_len, 224, 3, 224)
        target_set = target_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": target_set.repeat_interleave(2, dim=1).cuda()}
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_features = outputs.hidden_states[-1].mean(dim=1)
        query_features = self.model.fc_norm(query_features)

        # Repeat embeddings for each query
        support_mm_features = support_mm_features.unsqueeze(0).repeat(self.way*self.query_per_class, 1, 1)
        combined_features = torch.cat((query_features.unsqueeze(1), support_mm_features), dim=1)
        combined_features = self.task_specific_learning_module(combined_features)
        query_features_aug, support_mm_features_aug = combined_features.split([1, self.way], dim=1)

        similarity_matrix = self.cosine_similarity(
            query_features_aug.expand(-1, support_mm_features_aug.size(1), -1),
            support_mm_features_aug
        )

        # Compute global logits
        if self.use_l2_loss:
            support_global_logits = self.global_classification_layer(support_features)
            query_global_logits = self.global_classification_layer(query_features)
        else:
            support_global_logits = 0. # We need to return something
            query_global_logits = 0. # We need to return something

        return {"similarity_matrix": similarity_matrix,
                "support_global_logits": support_global_logits,
                "query_global_logits": query_global_logits}

    def compute_loss(self, similarity_matrix, support_global_logits, query_global_logits,
                           support_labels, target_labels, batch_class_list):
        # KNOWN LOSS ######################
        known_indices = target_labels != -1
        similarity_matrix_k = similarity_matrix[known_indices]
        # support_global_logits_k = support_global_logits[known_indices]
        # query_global_logits_k = query_global_logits[known_indices]
        target_labels_k = target_labels[known_indices]
        
        if known_indices.sum() > 0:
            # Compute L1 loss
            # NO TARGET LABELS!
            # ordering doesn't matter, we need to check where target labels is equal to support labels
            true_target_labels = torch.argsort(support_labels)[target_labels_k].cuda()
            l1_loss = torch.nn.functional.cross_entropy(similarity_matrix_k, true_target_labels)

            # Compute L2 loss
            if self.use_l2_loss:
                global_support_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
                global_query_labels = torch.tensor([self.train_unique_classes.index(x) for x in batch_class_list[target_labels]]).cuda()  # it was real_target_labels
                # Make support label one hot to apply them correctly
                l2_loss_support = torch.nn.functional.cross_entropy(support_global_logits, global_support_labels)
                l2_loss_query = torch.nn.functional.cross_entropy(query_global_logits, global_query_labels)
                l2_loss = l2_loss_support + l2_loss_query
                # l2_loss = self.alpha * l2_loss
            else:
                l2_loss = 0. # We need to return something
        else:
            l1_loss = None
            l2_loss = None

        # UNKNOWN LOSS ######################
        os_known_loss, os_unknown_loss = self.open_set_loss(similarity_matrix, target_labels.cuda())

        return {"l1_loss": l1_loss, "l2_loss": l2_loss, "os_known_loss": os_known_loss, "os_unknown_loss": os_unknown_loss}

    def optimize(self, l1_loss, l2_loss, os_known_loss, os_unknown_loss, optimizer):
        optimizer.zero_grad()
        l1_loss = l1_loss if l1_loss is not None else torch.FloatTensor([0]).cuda()
        l2_loss = l2_loss if l2_loss is not None else torch.FloatTensor([0]).cuda()
        os_known_loss = os_known_loss if os_known_loss is not None else torch.FloatTensor([0]).cuda()
        os_unknown_loss = os_unknown_loss if os_unknown_loss is not None else torch.FloatTensor([0]).cuda()
        open_set_loss = os_known_loss + os_unknown_loss
        if self.use_l2_loss:
            (l1_loss + self.alpha*l2_loss + open_set_loss).backward()
        else:
            (l1_loss + open_set_loss).backward()
        optimizer.step()

    def compute_metrics(self, similarity_matrix, support_global_logits, query_global_logits,
                              support_labels, target_labels, batch_class_list):
        known_indices = target_labels != -1
        true_target_labels = []  # to make the open set part work when no known query is present
        if known_indices.sum() > 0:
            similarity_matrix_k = similarity_matrix[known_indices]
            target_labels_k = target_labels[known_indices]

            # NO TARGET LABELS!
            # ordering doesn't matter, we need to check where target labels is equal to support labels
            true_target_labels = torch.argsort(support_labels)[target_labels_k].cuda()
            fs_acc = compute_accuracy(similarity_matrix_k, true_target_labels)

            if self.use_l2_loss:
                unique_classes = self.train_unique_classes
                global_support_labels = torch.tensor([unique_classes.index(x) for x in batch_class_list[support_labels]]).cuda()
                global_query_labels = torch.tensor([unique_classes.index(x) for x in batch_class_list[target_labels]]).cuda()
                global_support_acc = compute_accuracy(support_global_logits, global_support_labels)
                global_query_acc = compute_accuracy(query_global_logits, global_query_labels)
            else:
                global_support_acc = 0 # We need to return something
                global_query_acc = 0 # We need to return something
        else:
            fs_acc = None
            global_support_acc = None
            global_query_acc = None

        # OPEN SET PART: AUROC
        target_os_matrix = torch.zeros_like(similarity_matrix).cuda()
        for i, elem in enumerate(true_target_labels):
            if elem != -1:
                target_os_matrix[i, elem] = 1
        os_auroc = roc_auc_score(target_os_matrix.reshape(-1).detach().cpu().numpy(), similarity_matrix.reshape(-1).detach().cpu().numpy())

        return {"fs_acc": fs_acc, "global_support_acc": global_support_acc, "global_query_acc": global_query_acc, "os_auroc": os_auroc}

    def visualize_debug(self):
        pass
        # # Visualize support NOTE for debug use
        # support_set_flat_labels = [videodataset.class_folders[int(x.item())] for x in batch_class_list[support_labels]]
        # support_set_flat = support_set.reshape(dataset.way, dataset.shot, dataset.seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        # counter = 0
        # for k in support_set_flat:
        #     for n in k:
        #         print(support_set_flat_labels[counter])
        #         for i in n:
        #             cv2.imshow("image", i.numpy())
        #             cv2.waitKey(0)
        #         counter += 1
        # # Visualize queries
        # target_set_flat_labels = [videodataset.class_folders[int(x.item())] for x in real_target_labels]
        # target_set_flat = target_set.reshape(dataset.way, dataset.query_per_class, dataset.seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        # counter = 0
        # for k in target_set_flat:
        #     print(target_set_flat_labels[counter])
        #     for n in k:
        #         for i in n:
        #             cv2.imshow("image", i.numpy())
        #             cv2.waitKey(0)
        #     counter += 1