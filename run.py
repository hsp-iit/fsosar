from transformers import AutoImageProcessor, AutoModelForVideoClassification, BertTokenizer, BertModel
from videoloader import VideoDataset, SSv2, HMDB, UCF
import torch
import cv2
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Load data
dataset = SSv2()
dataset.shot = 1  # NOTE: remove on the server!
dataset.seq_len = 8  # NOTE: !
dataset.query_per_class = 1 # NOTE: change on the server!
dataloader = VideoDataset(dataset)
# TODO DEFINE LOADER
# dataloader = torch.utils.data.DataLoader(dataset_loader, batch_size=1, shuffle=False)

# Load model and processor
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", output_hidden_states=True)
model.cuda()
model.train()

# Freeze patch_embeddings and first 6 layers
for param in model.videomae.embeddings.parameters():
    param.requires_grad = False
for param in model.videomae.encoder.layer[:6].parameters():
    param.requires_grad = False

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.cuda()
bert_model.eval()

# Define the two modules
class PlainTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size):
        super(PlainTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

mm_fusion_module = PlainTransformer(hidden_size=768, num_layers=2, num_heads=8, intermediate_size=3072)
mm_fusion_module.cuda()
mm_fusion_module.train()

task_specific_learning_module = PlainTransformer(hidden_size=768, num_layers=1, num_heads=8, intermediate_size=3072)
task_specific_learning_module.cuda()
task_specific_learning_module.train()

# Get features of class names with BERT
class_names = dataloader.class_folders
class_name_embeddings = []
plain_class_names = []
for class_name in class_names:
    plain_class_name = class_name.replace('_', ' ')
    plain_class_names.append(plain_class_name)
    inputs = tokenizer(plain_class_name, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    class_name_embeddings.append(outputs.last_hidden_state.squeeze(0))

# Define cosine similarity and softmax
cosine_similarity = nn.CosineSimilarity(dim=-1)
softmax = nn.Softmax(dim=-1)

# Define cross-entropy loss
cross_entropy = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(list(mm_fusion_module.parameters()) + list(task_specific_learning_module.parameters()), lr=1e-4)

for elem in dataloader:
    support_set = elem["support_set"]
    target_set = elem["target_set"]
    target_labels = elem["target_labels"]

    # Reorder suport set w.r.t. support labels
    support_set = support_set.reshape(dataset.way*dataset.shot, dataset.seq_len, 3, 224, 224)
    support_labels = elem['support_labels'].long()
    support_set = support_set[support_labels.argsort()]
    support_labels = support_labels[support_labels.argsort()]

    # Visualize support
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
    # TODO: visualize query

    #####################################################
    ### MULTI-MODAL SUPPORT SET PROTOTYPES GENERATION ###
    #####################################################
    support_set = support_set.reshape(dataset.way*dataset.shot*dataset.seq_len, 3, 224, 224)
    inputs = processor(torch.unbind(support_set), return_tensors="pt", do_rescale=False)
    inputs['pixel_values'] = inputs['pixel_values'].reshape(dataset.way*dataset.shot, dataset.seq_len, 3, 224, 224)
    # Since VideoMAE has been trained with 16 frames, we need to repeat the frames to reach 16
    if dataset.seq_len == 8:
        inputs['pixel_values'] = inputs['pixel_values'].repeat_interleave(2, dim=1)
    inputs['pixel_values'] = inputs['pixel_values'].cuda()
    outputs = model(**inputs)
    video_embeddings = outputs.hidden_states[-1]
    video_embeddings = video_embeddings.mean(dim=1)
    video_embeddings = model.fc_norm(video_embeddings)
    # Note: now the embeddings are the same given to the VideoMAE classifier
    # transformers/models/videomae/modeling_video.py L1101
    video_embeddings = video_embeddings.reshape(dataset.way, dataset.shot, -1)
    video_embeddings = video_embeddings.mean(dim=1)

    # Concatenate each class prototype with the class name embedding
    textual_embeddings = [class_name_embeddings[x] for x in elem['batch_class_list'].long()]
    raw_mm_embeddings = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(video_embeddings, textual_embeddings)]

    # Pass each multi-modal embedding through a 2-layer transformer
    mm_embeddings = []
    for emb in raw_mm_embeddings:
        emb = emb.unsqueeze(0)
        mm_embeddings.append(mm_fusion_module(emb).squeeze(0)[0])  # Just the first token (video)
    mm_embeddings = torch.stack(mm_embeddings)

    ##################################
    ### QUERY PROTOYPES GENERATION ###
    ##################################
    target_set = target_set.reshape(dataset.query_per_class*dataset.way*dataset.seq_len, 3, 224, 224)
    inputs = processor(torch.unbind(target_set), return_tensors="pt", do_rescale=False)
    inputs['pixel_values'] = inputs['pixel_values'].reshape(dataset.way*dataset.query_per_class, dataset.seq_len, 3, 224, 224)
    # Since VideoMAE has been trained with 16 frames, we need to repeat the frames to reach 16
    if dataset.seq_len == 8:
        inputs['pixel_values'] = inputs['pixel_values'].repeat_interleave(2, dim=1)
    inputs['pixel_values'] = inputs['pixel_values'].cuda()
    outputs = model(**inputs)
    query_embeddings = outputs.hidden_states[-1]
    query_embeddings = query_embeddings.mean(dim=1)
    query_embeddings = model.fc_norm(query_embeddings)

    # Task Specific Learning Module
    mm_embeddings = mm_embeddings.unsqueeze(0).repeat(5, 1, 1)
    embeddings = torch.cat((query_embeddings.unsqueeze(1), mm_embeddings), dim=1)
    embeddings = task_specific_learning_module(embeddings)
    query_embeddings, support_embeddings = embeddings.split([1, 5], dim=1)

    # Compute cosine similarity
    similarity_matrix = torch.zeros(query_embeddings.size(0), support_embeddings.size(1)).cuda()
    for i in range(query_embeddings.size(0)):
        for j in range(support_embeddings.size(1)):
            similarity_matrix[i, j] = cosine_similarity(query_embeddings[i], support_embeddings[i, j])

    print(similarity_matrix)
    scores = softmax(similarity_matrix)
    l1_loss = cross_entropy(scores, target_labels.long().cuda())

    # TODO add l2 loss
    optimizer.zero_grad()
    l1_loss.backward()
    optimizer.step()
   
    print(l1_loss.item())
