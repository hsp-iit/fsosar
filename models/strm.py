import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations 
from utils import OpenSetLoss, compute_accuracy
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import roc_auc_score
import wandb
from utils import BinaryClassificationModelSTRM


NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)

class DistanceLoss(nn.Module):
    "Compute the Query-class similarity on the patch-enriched features."
    def __init__(self, args, temporal_set_size=3, use_cosine_sim=False):
        super(DistanceLoss, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args["seq_len"] * 1.5)
        self.dropout = nn.Dropout(p = 0.1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args["seq_len"])]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) # 28 for tempset_2

        # nn.Linear(4096, 1024)
        self.clsW = nn.Linear(self.args["trans_linear_in_dim"] * self.temporal_set_size, self.args["trans_linear_in_dim"]//2)
        self.relu = torch.nn.ReLU() 

        # Open set part
        self.use_cosine_sim = use_cosine_sim
        self.similarity_function = nn.CosineSimilarity()


    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25
        
        # Add a dropout before creating tuples
        support_set = self.dropout(support_set) # 25 x 8 x 2048
        queries = self.dropout(queries) # 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2).to(device) # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2) # 20 x 28 x 4096
        support_labels = support_labels.to(device)
        unique_labels = torch.unique(support_labels) # 5

        query_embed = self.clsW(queries.view(-1, self.args["trans_linear_in_dim"]*self.temporal_set_size)) # 560[20x28] x 1024

        # Add relu after clsW
        query_embed = self.relu(query_embed) # 560 x 1024        

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_all = torch.zeros(n_queries, self.args["way"]).to(support_set.device) # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 4096

            # Reshaping the selected keys
            class_k = class_k.view(-1, self.args["trans_linear_in_dim"]*self.temporal_set_size) # 140 x 4096

            # Get the support set projection from the current class
            support_embed = self.clsW(class_k.to(queries.device))  # 140[5 x 28] x1024

            # Add relu after clsW
            support_embed = self.relu(support_embed) # 140 x 1024

            if self.use_cosine_sim:
                # Normalize features before cosine similarity
                query_embed = query_embed / torch.norm(query_embed, dim=-1).unsqueeze(-1)
                support_embed = support_embed / torch.norm(support_embed, dim=-1).unsqueeze(-1)
                distmat = torch.mm(query_embed, support_embed.T)
                distmat = -distmat  # Since they use a distance and we use a similarity, we need to do this
            else:
                # Calculate p-norm distance between the query embedding and the support set embedding
                distmat = torch.cdist(query_embed, support_embed) # 560[20 x 28] x 140[28 x 5]  # Closed set

            # Across the 140 tuples compared against, get the minimum distance for each of the 560 queries
            min_dist = distmat.min(dim=1)[0].reshape(n_queries, self.tuples_len) # 20[5-way x 4-queries] x 28

            # Average across the 28 tuples
            query_dist = min_dist.mean(dim=1)  # 20

            # Make it negative as this has to be reduced.
            distance = -1.0 * query_dist
            c_idx = c.long()
            dist_all[:,c_idx] = distance # Insert into the required location.

        return_dict = {'logits': dist_all}
        
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3, use_cosine_sim=False):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args["seq_len"] * 1.5)
        self.pe = PositionalEncoding(self.args["trans_linear_in_dim"], self.args["trans_dropout"], max_len=max_len)

        self.k_linear = nn.Linear(self.args["trans_linear_in_dim"] * temporal_set_size, self.args["trans_linear_out_dim"])
        self.v_linear = nn.Linear(self.args["trans_linear_in_dim"] * temporal_set_size, self.args["trans_linear_out_dim"])

        self.norm_k = nn.LayerNorm(self.args["trans_linear_out_dim"])
        self.norm_v = nn.LayerNorm(self.args["trans_linear_out_dim"])
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args["seq_len"])]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) #28

        # Open set part
        self.use_cosine_sim = use_cosine_sim
        self.similarity_function = nn.CosineSimilarity()
    
    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25
        
        # static pe after adding the position embedding
        support_set = self.pe(support_set) # Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        queries = self.pe(queries) # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2) # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2) # 20 x 28 x 4096

        # apply linear maps for performing self-normalization in the next step and the key map's output
        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set) # 25 x 28 x 1152
        queries_ks = self.k_linear(queries) # 20 x 28 x 1152
        support_set_vs = self.v_linear(support_set) # 25 x 28 x 1152
        queries_vs = self.v_linear(queries) # 20 x 28 x 1152
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks).to(device) # 25 x 28 x 1152
        mh_queries_ks = self.norm_k(queries_ks).to(device) # 20 x 28 x 1152
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.to(device) # 25 x 28 x 1152
        mh_queries_vs = queries_vs.to(device) # 20 x 28 x 1152
        
        unique_labels = torch.unique(support_labels) # 5

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args["way"]).to(device) # 20 x 5
        all_prototypes = torch.zeros(n_queries, self.args["way"], 28, 1152).to(device) # 20 x 5 x 1152

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class 
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 1152
            k_bs = class_k.shape[0] # 5

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args["trans_linear_out_dim"]) # 20 x 5 x 28 x 28

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3) # 20 x 28 x 5 x 28 
            
            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1) # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)] # list(20) x 28 x 140
            class_scores = torch.cat(class_scores) # 560 x 140 - concatenate all the scores for the tuples
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len) # 20 x 28 x 5 x 28
            class_scores = class_scores.permute(0,2,1,3) # 20 x 5 x 28 x 28
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v) # 20 x 5 x 28 x 1152 
            query_prototype = torch.sum(query_prototype, dim=1).to(device) # 20 x 28 x 1152 -> Sum across all the support set values of the corres. class
            
            if self.use_cosine_sim:
                # Normalize features before cosine similarity
                mh_queries_vs = mh_queries_vs / torch.norm(mh_queries_vs, dim=-1).unsqueeze(-1)
                query_prototype = query_prototype / torch.norm(query_prototype, dim=-1).unsqueeze(-1)
                distance = torch.matmul(mh_queries_vs, query_prototype.transpose(-1, -2))
                distance = distance.mean(dim=[-1, -2])
            else:
                # calculate distances from queries to query-specific class prototypes
                diff = mh_queries_vs - query_prototype # 20 x 28 x 1152
                norm_sq = torch.norm(diff, dim=[-2,-1])**2 # 20 
                distance = torch.div(norm_sq, self.tuples_len) # 20
                
                # multiply by -1 to get logits
                distance = distance * -1

            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance # 20
            all_prototypes[:,c_idx] = (mh_queries_vs - query_prototype) # 20 x 5 x 1152
        
        return all_distances_tensor, all_prototypes

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class Token_Perceptron(torch.nn.Module):
    '''
        2-layer Token MLP
    '''
    def __init__(self, in_dim):
        super(Token_Perceptron, self).__init__()
        # in_dim 8
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):

        # Applying the linear layer on the input
        output = self.inp_fc(x) # B x 2048 x 8

        # Apply the relu non-linearity
        output = self.relu(output) # B x 2048 x 8

        # Apply the 2nd linear layer
        output = self.out_fc(output)
        
        return output

class Bottleneck_Perceptron_2_layer(torch.nn.Module):
    '''
        2-layer Bottleneck MLP
    '''
    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_2_layer, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.out_fc(output)
        
        return output 

class Bottleneck_Perceptron_3_layer_res(torch.nn.Module):
    '''
        3-layer Bottleneck MLP followed by a residual layer
    '''
    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_3_layer_res, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim//2)
        self.hid_fc = nn.Linear(in_dim//2, in_dim//2)
        self.out_fc = nn.Linear(in_dim//2, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.relu(self.hid_fc(output)) 
        output = self.out_fc(output)
        
        return output + x # Residual output

class Self_Attn_Bot(nn.Module):
    """ Self attention Layer
        Attention-based frame enrichment
    """
    def __init__(self,in_dim, seq_len):
        super(Self_Attn_Bot,self).__init__()
        self.chanel_in = in_dim # 2048
        
        # Using Linear projections for Key, Query and Value vectors
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        self.softmax  = nn.Softmax(dim=-1) #
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Bot_MLP = Bottleneck_Perceptron_3_layer_res(in_dim)
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):

        """
            inputs :
                x : input feature maps( B X C X W )[B x 16 x 2048]
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width)
        """

        # Add a position embedding to the 16 patches
        x = self.pe(x) # B x 16 x 2048

        m_batchsize,C,width = x.size() # m = 200/160, C = 2048, width = 16

        # Save residual for later use
        residual = x # B x 16 x 2048

        # Perform query projection
        proj_query  = self.query_proj(x) # B x 16 x 2048

        # Perform Key projection
        proj_key = self.key_proj(x).permute(0, 2, 1) # B x 2048  x 16

        energy = torch.bmm(proj_query,proj_key) # transpose check B x 16 x 16
        attention = self.softmax(energy) #  B x 16 x 16

        # Get the entire value in 2048 dimension 
        proj_value = self.value_conv(x).permute(0, 2, 1) # B x 2048 x 16

        # Element-wise multiplication of projected value and attention: shape is x B x C x N: 1 x 2048 x 8
        out = torch.bmm(proj_value,attention.permute(0,2,1)) # B x 2048 x 16

        # Reshaping before passing through MLP
        out = out.permute(0, 2, 1) # B x 16 x 2048

        # Passing via gamma attention
        out = self.gamma*out + residual # B x 16 x 2048

        # Pass it via a 3-layer Bottleneck MLP with Residual Layer defined within MLP
        out = self.Bot_MLP(out)  # B x 16 x 2048

        return out

class MLP_Mix_Enrich(nn.Module):
    """ 
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """
    def __init__(self,in_dim, seq_len):
        super(MLP_Mix_Enrich,self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len) # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)

        max_len = int(seq_len * 1.5) # seq_len = 8
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature 
        """

        # Add a position embedding to the 8 frames
        x = self.pe(x) # B x 8 x 2048

        # Store the residual for use later
        residual1 = x # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8 
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1 # B x 8 x 2048

        # Storing a residual 
        residual2 = out # B x 8 x 2048
        
        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2 # B x 8 x 2048

        return out
        

class STRM(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args, disc=None, gc=None, dp=None):
        super(STRM, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args["method"] == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args["method"] == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args["method"] == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        if dp:
            self.resnet = nn.DataParallel(self.resnet)
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        self.use_cosine_sim = args["use_cosine_sim"]

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s, use_cosine_sim=self.use_cosine_sim) for s in args["temp_set"]]) 

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = nn.ModuleList([DistanceLoss(args, s, use_cosine_sim=self.use_cosine_sim) for s in args["temp_set"]])

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(self.args["trans_linear_in_dim"], self.num_patches)

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args["trans_linear_in_dim"], self.args["seq_len"])
        
        if gc:
            self.garbage_support = nn.Parameter(torch.randn(1, self.args["seq_len"], self.args["trans_linear_in_dim"]), requires_grad=True).cuda()
        elif disc:
            self.discriminator = BinaryClassificationModelSTRM(1152).cuda()
        self.gc = gc
        self.disc = disc
        
        # Initialize debug counter for visual debugging
        self.debug_samples_counter = 0


    def loss(self, test_logits_sample, test_labels, device):
        """
        Compute the classification loss.
        """
        weights = torch.tensor([1.0]*self.args["way"], dtype=torch.float, device=device, requires_grad=False)
        if self.gc:
            weights[-1] = 1/5  # each class appear 1/10 of the time, os 1/2 of the time, 1/2*1/5 = 1/10
        if len(test_logits_sample.shape) == 2:
            test_logits_sample = test_logits_sample.unsqueeze(0)
        size = test_logits_sample.size()
        sample_count = size[0]  # scalar for the loop counter
        num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

        log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none', weight=weights)
        score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0)

    def forward(self, context_images, context_labels, target_images, precomputed_context_features=None, batch_class_list=None):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''

        if precomputed_context_features == None:
            context_features = self.resnet(context_images) # 200 x 2048 x 7 x 7
            context_features = self.adap_max(context_features) # 200 x 2048 x 4 x 4
            context_features = context_features.reshape(-1, self.args["trans_linear_in_dim"], self.num_patches) # 200 x 2048 x 16
            context_features = context_features.permute(0, 2, 1) # 200 x 16 x 2048
            context_features = self.attn_pat(context_features) # 200 x 16 x 2048 
            context_features = torch.mean(context_features, dim = 1) # 200 x 2048
            context_features = context_features.reshape(-1, self.args["seq_len"], self.args["trans_linear_in_dim"]) # 25 x 8 x 2048
        else:
            context_features = precomputed_context_features

        if self.gc:
            context_features = torch.cat((context_features, self.garbage_support), 0)
            context_labels = torch.cat((context_labels, torch.tensor(self.args["way"]-1).cuda().unsqueeze(0)))
        # Line below added by me to make this work
        target_images = target_images.reshape(-1, 3, self.args["img_size"], self.args["img_size"])
        target_features = self.resnet(target_images) # 160 x 2048 x 7 x 7
        target_features = self.adap_max(target_features) # 160 x 2048 x 4 x 4
        target_features = target_features.reshape(-1, self.args["trans_linear_in_dim"], self.num_patches) # 160 x 2048 x 16       
        target_features = target_features.permute(0, 2, 1) # 160 x 16 x 2048
        target_features = self.attn_pat(target_features) # 160 x 16 x 2048
        target_features = torch.mean(target_features, dim = 1) # 160 x 2048
        target_features = target_features.reshape(-1, self.args["seq_len"], self.args["trans_linear_in_dim"]) # 20 x 8 x 2048

        # Compute logits using the new loss before applying frame-level attention
        all_logits_post_pat = [n(context_features, context_labels, target_features)['logits'] for n in self.new_dist_loss_post_pat]
        all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        # Combing the patch and frame-level logits
        sample_logits_post_pat = all_logits_post_pat
        sample_logits_post_pat = torch.mean(sample_logits_post_pat, dim=[-1]) # 20 x 5

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features) # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features) # 20 x 8 x 2048

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr) for t in self.transformers]
        all_logits_fr, all_prototypes = [x[0] for x in all_logits_fr],  [x[1] for x in all_logits_fr]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1]) # 20 x 5

        logits = split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]).squeeze(0)
        all_prototypes = torch.stack(all_prototypes).squeeze(0)
        if self.disc:
            predictions = torch.argmax(logits, dim=-1)
            best_diffs = all_prototypes[torch.arange(all_prototypes.shape[0]), predictions]
            disc_prob = self.discriminator(best_diffs)
        else:
            disc_prob = None

        return_dict = {'similarity_matrix': logits,
                    'logits_post_pat': 0.1*split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]]).squeeze(0),
                    'disc_prob': disc_prob,
                    'support_features': context_features,  # 25 x 8 x 2048
                    'query_features': target_features,     # 20 x 8 x 2048  
                    'support_mm_features_aug': context_features_fr,  # 25 x 8 x 2048 (frame-enriched)
                    'query_features_aug': target_features_fr}        # 20 x 8 x 2048 (frame-enriched)

        return return_dict  #, context_features  # Precomputed context features needed

    def forward_hook(self, module, input, output):
        self.activations.append(output)

    def register_activation_hook(self):
        # Assuming the ResNet backbone is an attribute of your model
        resnet_backbone = self.resnet
        self.activations = []
        resnet_backbone[-1][-1].register_forward_hook(self.forward_hook)


    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args["num_gpus"] > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args["num_gpus"])])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [n.cuda(0) for n in self.new_dist_loss_post_pat]

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=[i for i in range(0, self.args["num_gpus"])])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[i for i in range(0, self.args["num_gpus"])])


    def set_train(self):
        return  # Nothing to do for STRM, train is called in main loop

    def compute_additional_metrics(self, *args, **kwargs):
        return {}

    def compute_known_losses(self, similarity_matrix, logits_post_pat, target_labels=None, **kwargs):

        known_indices = target_labels != -1
        task_loss = self.loss(similarity_matrix[known_indices], target_labels[known_indices], similarity_matrix.device) / known_indices.sum()
        task_loss_post_pat = self.loss(logits_post_pat[known_indices], target_labels[known_indices], similarity_matrix.device) / known_indices.sum()
        task_loss_post_pat = task_loss_post_pat*0.1
        self.debug_data = {"similarity_matrix": wandb.Table(columns=list(range(similarity_matrix.shape[1])), data=similarity_matrix.detach().cpu().numpy().tolist()),
                           "true_target_labels": wandb.Table(columns=[0], data=target_labels.detach().cpu().numpy()[..., None])}

        return {"task_loss": task_loss, "task_loss_post_pat": task_loss_post_pat}

    def get_debug_data(self):
        return self.debug_data

    def set_eval(self):
        return

    def visual_debug(self, similarity_matrix=None, support_global_logits=None, query_global_logits=None, videodataset=None, support_labels=None, target_labels=None, batch_class_list=None, support_set=None, target_set=None, disc_prob=None, unknown_labels=None, support_features=None, query_features=None, support_mm_features_aug=None, query_features_aug=None):
        import cv2
        import imageio
        import os
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        os.makedirs(f'visual_debug/{self.debug_samples_counter}', exist_ok=True)

        # t-SNE visualization of features
        if support_mm_features_aug is not None and query_features_aug is not None:
            # Prepare features for t-SNE
            all_features = []
            labels = []
            colors = []
            
            # For STRM, we'll use the frame-enriched features and average across temporal dimension
            # support_mm_features_aug: [n_support, seq_len, feature_dim] = [25, 8, 2048]
            # query_features_aug: [n_queries, seq_len, feature_dim] = [20, 8, 2048]
            
            # Average support features across temporal dimension for visualization
            support_features_for_viz = support_mm_features_aug.mean(dim=1)  # [25, 2048] -> [n_ways * n_shots, 2048]
            
            # Get unique support labels to organize by class
            unique_labels = torch.unique(support_labels)
            
            # Add support features to visualization (organized by class)
            support_idx = 0
            for class_idx, class_label in enumerate(unique_labels):
                class_mask = support_labels == class_label
                class_support_features = support_features_for_viz[class_mask]
                
                for i, feat in enumerate(class_support_features):
                    all_features.append(feat.detach().cpu().numpy())
                    class_name = videodataset.class_folders[int(batch_class_list[class_idx])]
                    labels.append(f"Support: {class_name}")
                    colors.append(f"C{class_idx}")  # Different color for each support class
            
            # Separate known and unknown queries
            known_indices = target_labels != -1
            unknown_indices = target_labels == -1
            
            # Add known query features (frame-enriched, averaged across temporal dimension)
            if known_indices.sum() > 0:
                query_features_aug_known = query_features_aug[known_indices].mean(dim=1)  # Average across seq_len
                target_labels_known = target_labels[known_indices]
                for i, feat in enumerate(query_features_aug_known):
                    all_features.append(feat.detach().cpu().numpy())
                    class_idx = target_labels_known[i].item()
                    class_name = videodataset.class_folders[int(batch_class_list[class_idx])]
                    labels.append(f"Known Query: {class_name}")
                    colors.append(f"C{class_idx}")  # Same color as corresponding support class
            
            # Add unknown query features (frame-enriched, averaged across temporal dimension)
            if unknown_indices.sum() > 0:
                query_features_aug_unknown = query_features_aug[unknown_indices].mean(dim=1)  # Average across seq_len
                # Get the actual indices where unknown_indices is True
                unknown_idx_positions = torch.where(unknown_indices)[0].cpu()
                # Convert unknown_labels to tensor if it's not already
                if not isinstance(unknown_labels, torch.Tensor):
                    unknown_labels_tensor = torch.tensor(unknown_labels)
                else:
                    unknown_labels_tensor = unknown_labels.cpu()
                # Use the positions to index into unknown_labels
                unknown_labels_subset = unknown_labels_tensor[unknown_idx_positions]
                for i, feat in enumerate(query_features_aug_unknown):
                    all_features.append(feat.detach().cpu().numpy())
                    class_name = videodataset.class_folders[int(unknown_labels_subset[i])]
                    labels.append(f"Unknown Query: {class_name}")
                    colors.append('red')  # Red for unknown queries
            
            if len(all_features) > 1:
                # Convert to numpy array and standardize
                features_array = np.stack(all_features)
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)
                
                # Apply t-SNE
                perplexity = min(30, len(features_array) - 1)  # Ensure perplexity < n_samples
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                features_2d = tsne.fit_transform(features_scaled)
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                # Count support features
                support_count = sum(len(support_features_for_viz[support_labels == label]) for label in unique_labels)
                
                # Plot support features (colored squares)
                plt.scatter(features_2d[:support_count, 0], features_2d[:support_count, 1], 
                           c=[colors[i] for i in range(support_count)], 
                           marker='s', s=100, alpha=0.8, label='Support Features')
                
                # Plot known query features (circles with class colors)
                known_start = support_count
                known_count = known_indices.sum().item()
                if known_count > 0:
                    known_end = known_start + known_count
                    # Use the class-specific colors for known queries
                    known_colors = [colors[i] for i in range(known_start, known_end)]
                    plt.scatter(features_2d[known_start:known_end, 0], features_2d[known_start:known_end, 1], 
                               c=known_colors, marker='o', s=60, alpha=0.8, label='Known Query Features')
                
                # Plot unknown query features (black crosses)
                unknown_count = unknown_indices.sum().item()
                if unknown_count > 0:
                    unknown_start = known_start + known_count
                    plt.scatter(features_2d[unknown_start:, 0], features_2d[unknown_start:, 1], 
                               c='black', marker='x', s=80, alpha=0.8, label='Unknown Query Features')
                
                plt.title('t-SNE Visualization of STRM Features')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Set axis limits with some padding to ensure all points are visible
                x_min, x_max = features_2d[:, 0].min(), features_2d[:, 0].max()
                y_min, y_max = features_2d[:, 1].min(), features_2d[:, 1].max()
                x_padding = (x_max - x_min) * 0.15  # 15% padding
                y_padding = (y_max - y_min) * 0.15  # 15% padding
                plt.xlim(x_min - x_padding, x_max + x_padding)
                plt.ylim(y_min - y_padding, y_max + y_padding)
                
                plt.tight_layout()
                plt.savefig(f'visual_debug/{self.debug_samples_counter}/tsne_features.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save feature info to text file
                with open(f'visual_debug/{self.debug_samples_counter}/tsne_info.txt', 'w') as f:
                    f.write("t-SNE Feature Visualization Info (STRM)\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Total features: {len(all_features)}\n")
                    f.write(f"Support features: {support_count}\n")
                    f.write(f"Known query features: {known_count}\n")
                    f.write(f"Unknown query features: {unknown_count}\n\n")
                    f.write("Feature Labels:\n")
                    for i, label in enumerate(labels):
                        f.write(f"{i}: {label}\n")

        self.debug_samples_counter += 1          
        return
        # Save support set gif (similar to SAFSAR)
        # support_set shape: [n_support, seq_len, H, W, C] - need to handle different input shapes
        if support_set.dim() == 4:  # [n_support * seq_len, H, W, C]
            support_set = support_set.reshape(-1, self.args["seq_len"], support_set.shape[1], support_set.shape[2], support_set.shape[3])
        elif support_set.dim() == 5:  # [n_support, seq_len, H, W, C] 
            pass  # Already correct shape
        else:
            # Try to reshape based on sequence length
            total_frames = support_set.shape[0]
            n_support = total_frames // self.args["seq_len"]
            support_set = support_set.reshape(n_support, self.args["seq_len"], support_set.shape[1], support_set.shape[2], support_set.shape[3])
        
        concatenated_frames = []
        for s in range(support_set.size(0)):
            class_id = support_labels[s].item()
            class_name = videodataset.class_folders[int(batch_class_list[class_id])]
            single_video = support_set[s] * 255  # [seq_len, H, W, C]
            single_video = single_video.detach().cpu().numpy().astype(np.uint8)
            
            # Ensure we have the right shape for concatenation [seq_len, H, W, C]
            if single_video.shape[-1] != 3:  # If channels are not in the last dimension
                if single_video.shape[1] == 3:  # Channels in dimension 1
                    single_video = np.transpose(single_video, (0, 2, 3, 1))  # [seq_len, C, H, W] -> [seq_len, H, W, C]
            
            concatenated_frame = np.concatenate([single_video[i] for i in range(len(single_video))], axis=1)
            text_img = np.ones((30, concatenated_frame.shape[1], 3), dtype=np.uint8) * 255
            concatenated_frame = np.concatenate((text_img, concatenated_frame), axis=0)
            concatenated_frames.append(concatenated_frame)

        os.makedirs(f'visual_debug/{self.debug_samples_counter}', exist_ok=True)
        imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/ss.gif', concatenated_frames, duration=250, loop=0)
        with open(f'visual_debug/{self.debug_samples_counter}/ss.txt', 'w') as f:
            for s in range(support_set.size(0)):
                class_id = support_labels[s].item()
                class_name = videodataset.class_folders[int(batch_class_list[class_id])]
                f.write(f"Support {s}: Class {class_id} ({class_name})\n")

        # Save query samples
        # target_set shape handling similar to support_set
        if target_set.dim() == 4:  # [n_queries * seq_len, H, W, C]
            target_set = target_set.reshape(-1, self.args["seq_len"], target_set.shape[1], target_set.shape[2], target_set.shape[3])
        elif target_set.dim() == 5:  # [n_queries, seq_len, H, W, C] 
            pass  # Already correct shape
        else:
            # Try to reshape based on sequence length
            total_frames = target_set.shape[0]
            n_queries = total_frames // self.args["seq_len"]
            target_set = target_set.reshape(n_queries, self.args["seq_len"], target_set.shape[1], target_set.shape[2], target_set.shape[3])
            
        predictions = torch.argmax(similarity_matrix, dim=-1)
        accept_score = torch.nn.functional.softmax(similarity_matrix, dim=-1).max(dim=-1)[0]
        
        for q in range(target_set.size(0)):
            query_label = target_labels[q].item()
            if query_label == -1:
                actual_class = int(unknown_labels[q])
                query_class_name = videodataset.class_folders[actual_class]
                res = "unknown"
            else:
                query_class_name = videodataset.class_folders[int(batch_class_list[query_label])]
                res = "known"
            
            pred_class = predictions[q].item()
            pred_class_name = videodataset.class_folders[int(batch_class_list[pred_class])]
            
            single_video = target_set[q] * 255  # [seq_len, H, W, C]
            single_video = single_video.detach().cpu().numpy().astype(np.uint8)
            
            # Ensure we have the right shape for concatenation [seq_len, H, W, C]
            if single_video.shape[-1] != 3:  # If channels are not in the last dimension
                if single_video.shape[1] == 3:  # Channels in dimension 1
                    single_video = np.transpose(single_video, (0, 2, 3, 1))  # [seq_len, C, H, W] -> [seq_len, H, W, C]
            
            concatenated_frame = np.concatenate([single_video[i] for i in range(len(single_video))], axis=1)
            
            # Add text information
            text_img = np.ones((60, concatenated_frame.shape[1], 3), dtype=np.uint8) * 255
            concatenated_frame = np.concatenate((text_img, concatenated_frame), axis=0)
            
            cur = q
            imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/{res}_{accept_score[cur]:.3f}_{query_class_name}_pred_{pred_class_name}.gif', [concatenated_frame], duration=250, loop=0)

        self.debug_samples_counter += 1
