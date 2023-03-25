import torch.nn as nn
import pdb
import torch 
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set
from dgl.nn.pytorch import RelGraphConv
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value):
        
        scores = torch.matmul(query.transpose(-1, -2), key) / (query.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value.transpose(-2, -1))

class BaseRGCN(nn.Module):
    def __init__(self, num_ntypes,i_dim, h_dim, g_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False):
        super(BaseRGCN, self).__init__()
        self.num_ntypes = num_ntypes
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.g_dim = g_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.num_ntypes, self.i_dim)

        self.layers = nn.ModuleList()

        i2h = self.build_input_layer()

        if i2h is not None:
            self.layers.append(i2h)

        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            # h2h = nn.DataParallel(h2h)
            self.layers.append(h2h)

        self.output_layer = self.build_output_layer()
        
        self.attention = ScaledDotProductAttention()


    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, type, norm, sketchs):

        h = g.ndata['node_type']
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(g, h, type, norm)  
        
        # h = h[train_idx]
        # Max pooling
        graph_h, index = torch.max(h, 0, True)
        
        # 将四个sketch拼接成一个矩阵
        sketchs_tensor = torch.stack(sketchs, dim=0)
        
        # 调整graph_h的维度以匹配注意力层的输入要求
        graph_h_expanded = graph_h.unsqueeze(0).repeat(1, sketchs_tensor.size(0), 1)


        # 使用注意力机制处理graph_h和四个sketch
        attn_output = self.attention(graph_h_expanded, sketchs_tensor, sketchs_tensor)
        
        # 求和
        attn_output = torch.sum(attn_output, dim=0)

        # 拼接graph_h与注意力处理后的输出
        # print(graph_h.shape,attn_output.shape)
        final_output = torch.cat((graph_h, attn_output.transpose(0,1)), dim=0)
        final_output = torch.sum(final_output, dim = 0).unsqueeze(0)
        
        graph_h = final_output
        
        h = self.output_layer(graph_h)

        return h

    
class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return RelGraphConv(self.i_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    
    def build_output_layer(self):
        # return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
        #         self.num_bases, activation=None,
        #         self_loop=self.use_self_loop)
        # return torch.nn.Linear(self.h_dim + self.g_dim, self.out_dim)
        return torch.nn.Linear(self.h_dim + self.g_dim, self.out_dim)

    def build_pooling_layer(self):
        return MaxPooling()
