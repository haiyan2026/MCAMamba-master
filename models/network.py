import numpy as np

import torch
import torch.nn as nn

from .util import initialize_weights
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention
from mamba_ssm import Mamba, SRMamba, BiMamba
import torch.nn.functional as F

from torch import einsum
from tqdm import tqdm
from einops import rearrange

'''

'''

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=256):
        super(SelfAttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, input_dim))
        self.fc = nn.Linear(input_dim, input_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_proj = self.fc(x)  # [batch_size, seq_length, input_dim]
        attn_scores = torch.matmul(x_proj, self.query.transpose(-2, -1))  # [batch_size, seq_length, 1]
        attn_scores = attn_scores.squeeze(-1)  # [batch_size, seq_length]
        attn_weights = F.softmax(attn_scores, dim=-1) 
        
        x_pool = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, input_dim]

        return x_pool

class MMAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=256,
            heads=8,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.1,
            num_pathways=284,
            attn_mode='cross'
    ):
        """

        Args:
            dim:
            dim_head:
            heads:
            residual:
            residual_conv_kernel:
            eps:
            dropout:
            num_pathways:
            attn_mode: ['full', 'partial', 'cross']
                'full': All pairs between P and H
                'partial': P->P, H->P, P->H
                'cross': P->H, H->P
                'self': P->P, H->H
        """
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.residual = residual
        self.attn_mode = attn_mode

        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def set_attn_mode(self, attn_mode):
        self.attn_mode = attn_mode

    def forward(self, x):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        # if mask != None:
        #     mask = rearrange(mask, 'b n -> b () n')
        #     q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        attn_histology = einsum(einops_eq, q_histology, k_histology)

        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        if self.attn_mode == 'full': # H->P, P->H, P->P, H->H
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_histology_pathways = torch.cat((cross_attn_histology, attn_histology), dim=-1).softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_pathways_histology @ v
            out_histology = attn_histology_pathways @ v
        elif self.attn_mode == 'cross': # P->H, H->P
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)

            # compute output
            out_pathways = cross_attn_pathways @ v[:, :, self.num_pathways:]
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'self': # P->P, H->H (Late fusion)
            attn_histology = attn_histology.softmax(dim=-1)
            attn_pathways = attn_pathways.softmax(dim=-1)

            out_pathways = attn_pathways @ v[:, :, :self.num_pathways]
            out_histology = attn_histology @ v[:, :, self.num_pathways:]
        elif self.attn_mode == 'partial': # H->P, P->H, P->P (SURVPATH)
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_pathways_histology @ v
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'mcat': # P->P, P->H
            cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)

            out_pathways = q_pathways
            out_histology = cross_attn_pathways @ v[:, :, self.num_pathways:]
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}")
        
        # # 添加门控机制控制交互强度
        # gate_p = self.gate_pathways(out_pathways)
        # gate_h = self.gate_histology(out_histology)
        # out_pathways = out_pathways * gate_p
        # out_histology = out_histology * gate_h

        out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        
        # return three matrices
        return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()
       
class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
            self,
            norm_layer=nn.LayerNorm,
            dim=256,
            dim_head=256,
            heads=6,
            residual=True,
            dropout=0.,
            num_pathways=188,
            attn_mode='cross'
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn_mode = attn_mode

        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            attn_mode=attn_mode
        )

    def set_attn_mode(self, attn_mode):
        self.attn.set_attn_mode(attn_mode)

    def forward(self, x=None):
        x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x))
        return x, attn_pathways, cross_attn_pathways, cross_attn_histology

class ClusteringLayer(nn.Module):
    def __init__(self, num_features, num_clusters):
        super(ClusteringLayer, self).__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, num_features))

    def forward(self, x):
        """
        Forward pass of the clustering layer.
        
        Args:
        x : torch.Tensor
            Input tensor of shape (1, n, num_features)
        
        Returns:
        torch.Tensor
            Output tensor of shape (1, num_clusters, num_features)
        """
        # Ensure the input is correctly shaped
        assert x.shape[1] > self.num_clusters and x.shape[2] == self.num_features
        
        # Calculate the distance from each input feature vector to each cluster center
        # x shape: (1, n, num_features)
        # cluster_centers shape: (num_clusters, num_features)
        # Expanded x shape: (1, n, 1, num_features)
        # Expanded cluster_centers shape: (1, 1, num_clusters, num_features)
        x_expanded = x.unsqueeze(2)
        centers_expanded = self.cluster_centers.unsqueeze(0).unsqueeze(0)
        
        # Compute distances
        distances = torch.norm(x_expanded - centers_expanded, dim=3)  # shape: (1, n, num_clusters)
        
        # Find the closest input features to each cluster center
        # We use argmin to find the index of the minimum distance
        _, indices = torch.min(distances, dim=1)  # Closest input feature index for each cluster
        
        # Gather the closest features
        selected_features = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, self.num_features))
        
        return selected_features

class MCAMamba(nn.Module):
    def __init__(self,num_pathway,omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat", model_size="small"):
        super(MCAMamba, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_pathway = num_pathway

        ###
        self.size_dict = {
            "pathomics": {"small": [768, 256, 256], "large": [768, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Self-attention pooling for feature
        self.path_att_pooling = SelfAttentionPooling()
        self.gene_att_pooling = SelfAttentionPooling()

        self.clustering = ClusteringLayer(num_features=256, num_clusters=256)

        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Encoder
        self.genomics_encoder = nn.ModuleList([
                            Mamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])
        # Decoder
        self.genomics_decoder = nn.ModuleList([
                            SRMamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])


        # Encoder
        self.pathomics_encoder = nn.ModuleList([
                            nn.Sequential(
                                nn.LayerNorm(256),
                                SRMamba(d_model=256, # Model dimension d_model
                                    d_state=16,  # SSM state expansion factor
                                    d_conv=4,    # Local convolution width
                                    expand=2,    # Block expansion factor
                                    )
                            )
                    for i in range(2)])
        
        # Decoder
        self.pathomics_decoder = nn.ModuleList([
                            Mamba(d_model=256, # Model dimension d_model
                                d_state=16,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
                    for i in range(1)])

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=256,
            dim_head=256,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathway
        )

        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omics"][i] for i in range(self.num_pathway)] 

        
        #---------------This segment could be remarked/modified for better performance---------------#
        # To save memory, you can also choose a subset of all patches from WSIs
        # as some cases have more than one WSI, the large number of patches will result in OOM
        max_features  = 20000 # can be adjusted by your GPU Memory
        num_features = x_path.size()[0]
        if num_features  > max_features:
            indices = np.random.choice(num_features,size = max_features,replace=False)
            x_path = x_path[indices]
        
        
        #------------------------------- embedding  -------------------------------#
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        if pathomics_features.shape[1] > 256:  
            pathomics_features = self.clustering(pathomics_features) 
        else:
            print(f"pathomics_features.shape[1]:{pathomics_features.shape[1]}")

        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)

        #------------------------------- encoder  -------------------------------#
        for g_mamba in self.genomics_encoder:
            genomics_features = g_mamba(genomics_features)

        for p_mamba in self.pathomics_encoder:
            pathomics_features = p_mamba(pathomics_features)

        
        #------------------------------- cross-omics attention  -------------------------------#
        tokens = torch.cat([genomics_features, pathomics_features], dim=1)
        tokens = self.identity(tokens)
        
        mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens)
        self.cross_attn_pathways = cross_attn_pathways
        self.cross_attn_histology = cross_attn_histology   
        
        #---> aggregate 
        # modality specific mean 
        genomics_in_pathomics = mm_embed[:, :self.num_pathway, :]
        
        pathomics_in_genomics = mm_embed[:, self.num_pathway:, :]
        
        #------------------------------- decoder  -------------------------------#
        for p_mamba in self.pathomics_decoder:
            pathomics_in_genomics = p_mamba(pathomics_features.transpose(0,1))

        for g_mamba in self.genomics_decoder:
            genomics_in_pathomics = g_mamba(genomics_features.transpose(0,1))

        #------------------------------- fusion  -------------------------------#
        path_fusion = self.path_att_pooling(pathomics_in_genomics.transpose(1,0))
        gene_fusion = self.gene_att_pooling(genomics_in_pathomics.transpose(1,0))
        
        if self.fusion == "concat":
            fusion = self.mm(torch.concat((path_fusion,gene_fusion),dim=1))  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(gene_fusion, gene_fusion)  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

       # predict
        logits = self.classifier(fusion)  # [1, n_classes]
        print(f"MCAMamba-->forward-->logits:{logits.shape}")  #[1, 4]

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S

