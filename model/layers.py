# -*- coding: utf-8 -*-
"""
Created on Sun April 12 19:37:02 2021

@author: Cunling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.utils import check_eq_shape

dgl.random.seed(1)
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OMP_NUM_THREADS=1

class GCN(nn.Module):
    '''
    in_feats:
        Input feature size
    out_feats:
        Output feature size
    activation:
        Applies an activation function to the updated node features
    '''
    
    def __init__(self, in_feats, out_feats, aggregator_type, feat_drop=0.2, bias=True, activation=None):
        super(GCN, self).__init__()
        self._in_feats = in_feats  # "C" in the paper
        self._out_feats = out_feats  # "F" in the paper
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout2d(feat_drop)
        self.head_num = 1
        self._activation_func = activation  # "ReLu" in the paper
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_feats, self._in_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_feats, self._in_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_feats, self._out_feats, bias = bias)
        self.fc_neigh = nn.Linear(2*self._in_feats, self._out_feats, bias = bias)
        self.bias = nn.Parameter(torch.Tensor(out_feats))  # bias, optional
        self.atten_l = nn.Parameter(torch.Tensor(out_feats))  # bias, optional
        self.atten_r = nn.Parameter(torch.Tensor(out_feats))  # bias, optional

        self.at_fc = nn.Linear(self._in_feats, self._in_feats, bias = False)
        self.atat_fc = nn.Linear(2*self._in_feats, 1, bias = False)


        
        # initialize the weight and bias
        self.reset_parameters()
    
        
    def reset_parameters(self):
        '''
        Reinitialize learnable parameters
        '''
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        # if self.bias is not None:
        #     nn.init.zeros_(self.bias)
    
    def _lstm_reducer(self, nodes):
        """
        docstring
        """
        m = nodes.mailbox['m']
        Node, Neibor, B, T, C = m.shape
        m = m.permute(0, 2, 3, 1, 4).contiguous().view(-1, T, C)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_feats)),m.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(m, h)
        print(rst.shape)
        return {'neigh':rst.squeeze(0)}

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=-1)
        a = self.atat_fc(z2)
        return {'e' : F.softmax(F.leaky_relu(a), 0)}
    
            
    def forward(self, g, features, PERSON_NUM, JOINT_NUM, edge_weight=None):
        '''
        formular: 
            h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})
        
        Inputs:
            g: 
                The fixed graph
            features: 
                H^{l}, i.e. Node features with shape [num_nodes, features_per_node]
                
        Returns:
            rst:
                H^{l+1}, i.e. Node embeddings of the l+1 layer with the 
                shape [num_nodes, hidden_per_node]
                
        Variables:
            gcn_msg: 
                Message function of GCN, i.e. What to be aggregated 
                (e.g. Sending node embeddings)
            gcn_reduce: 
                Reduce function of GCN, i.e. How to aggregate 
                (e.g. Summing neighbor embeddings)
                
        Notice: 'h' means node feature/embedding itself, 'm' means node's mailbox
        '''
        with g.local_scope():    
            # sampling graph
            # if self.training:
            #     g = dgl.sampling.sample_neighbors(g,[i for i in range(PERSON_NUM*JOINT_NUM)], int((PERSON_NUM-1))*JOINT_NUM*0.95)
            z = self.at_fc(features)
            g.ndata['z'] = z
            g.apply_edges(self.edge_attention)
            g = g.to('cpu')
            g = dgl.sampling.sample_neighbors(g,[i for i in range(PERSON_NUM*JOINT_NUM)], 5, prob='e')
            # g = dgl.sampling.sample_neighbors(g,[i for i in range(PERSON_NUM*JOINT_NUM)], 10)
            g = g.to('cuda:0')


            h_self = features         
            # normalize features by node's out-degrees
            # out_degs = g.out_degrees().to(features.device).float().clamp(min=1)  # shape [num_nodes]
            # norm1 = torch.pow(out_degs, -0.5)
            # shape1 = norm1.shape + (1,) * (features.dim() - 1)
            # norm1 = torch.reshape(norm1, shape1)
            # features = features * norm1

            feat_src = feat_dst = self.feat_drop(features)
            aggreagte_fn = fn.copy_src(src='h', out='m')
            if g.number_of_edges() == 0:
                g.dstdata['neigh'] = torch.zeros(feat_dst.shape[0], self._in_feats).to(feat_dst)
            
            if self._aggre_type == 'mean':
                g.srcdata['h'] = feat_src
                g.update_all(aggreagte_fn, fn.mean('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
            elif self._aggre_type == 'gcn':
                check_eq_shape(features)
                g.srcdata['h'] = feat_src
                g.dstdata['h'] = feat_dst
                g.update_all(aggreagte_fn, fn.sum('m', 'neigh'))
                degs = g.in_degrees().to(feat_dst)
                h_neigh = g.dstdata['neigh'] + g.dstdata['h']
                in_degs = g.in_degrees().to(h_neigh.device).float().clamp(min=1)  # shape [num_nodes]
                norm2 = torch.pow(in_degs, -0.5)
                shape2 = norm2.shape + (1,) * (h_neigh.dim() - 1)
                norm2 = torch.reshape(norm2, shape2)
                h_neigh = h_neigh * norm2
            elif self._aggre_type == 'pool':
                g.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                # g.srcdata['h'] = feat_src
                g.update_all(aggreagte_fn, fn.max('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
            elif self._aggre_type == 'lstm':
                g.srcdata['h'] = feat_src
                g.update_all(aggreagte_fn, self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            elif self._aggre_type == 'gat':
                # dot gat
                # v, b, t, c = feat_src.shape
                # feat_dst  = feat_src= self.fc_gat(feat_src).view(v, b, t, self.head_num, c)
                # g.srcdata.update({'ft': feat_src})
                # g.dstdata.update({'ft': feat_dst})
                # g.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
                # g.edata['sa'] = dgl.nn.functional.edge_softmax(g, g.edata['a'])/(self._out_feats**0.5)
                # g.update_all(fn.u_mul_e('ft', 'sa', 'attn'), fn.sum('attn', 'agg_u'))
                # h_neigh = self._activation_func(g.dstdata['agg_u']).sum(-2)
                g.srcdata['h'] = feat_src
                g.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'neigh'))
                h_neigh = g.dstdata['neigh']
            else:
                raise KeyError('Aggregator type {} not recognized'.format(self._aggre_type))

            
            # normalize features by node's in-degrees


            if self._aggre_type == 'gcn':
                # normalize features by node's in-degrees
                rst = self.fc_neigh(h_neigh)
            else:
                rst = self.fc_self(h_self + h_neigh)
     
            
            # activation
            if self._activation_func is not None:
                rst = self._activation_func(rst)

            return rst
    

class SageGraph(nn.Module):
    '''
    Section 3.2 in the paper
    SageGraph convolution layer (GCN used here as the spatial CNN)
    
    Inputs:
        c_in: input channels
        c_out: output channels
        g: DGLGraph
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]
        
    Return:
        y: output with the shape [batch_size, c_out, timesteps, num_nodes]
    '''
    def __init__(self, c_in, c_out, aggregator):
        super(SageGraph, self).__init__()
        self.gc = GCN(c_in, c_out, aggregator, activation=nn.ReLU())
        # self.gc = dgl.nn.pytorch.conv.SAGEConv(c_in, c_out, aggregator, activation=nn.ReLU()

    def forward(self, x, PERSON_NUM, JOINT_NUM):

        strat_joint = []
        end_joint = []
        total_point = PERSON_NUM*JOINT_NUM
        total_each_edge = (PERSON_NUM-1)*JOINT_NUM
        for person_id in range(PERSON_NUM):
            for joint_id in range(JOINT_NUM):
                start_joint_piece = [person_id*JOINT_NUM + joint_id] * total_each_edge
                end_joint_piece = [i for i in range(total_point)]
                for i in range(JOINT_NUM):
                    end_joint_piece.remove(person_id*JOINT_NUM + i)
                strat_joint.extend(start_joint_piece)
                end_joint.extend(end_joint_piece)
        # src_ids = torch.tensor(strat_joint).cuda()
        # dst_ids = torch.tensor(end_joint).cuda()
        src_ids = strat_joint
        dst_ids = end_joint
        g = dgl.graph((src_ids, dst_ids), num_nodes = total_point).to('cuda:0')
        if self.training:
            # g = dgl.sampling.sample_neighbors(g,[i for i in range(PERSON_NUM*JOINT_NUM)], 10)
            g = dgl.sampling.sample_neighbors(g,[i for i in range(PERSON_NUM*JOINT_NUM)], 10)
        BM, C, T, V = x.shape
        B = int(BM/PERSON_NUM) 
        # [B*M, C, T, V] --> [B, M, C, T, V]
        x = x.reshape(-1, PERSON_NUM, C, T, V)
        # [B, M, C, T, V] -->  [B, M*V, C, T]
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B, PERSON_NUM * V, C, T)
        # [B, M*V, C, T] -->  [M*V, B, T, C]
        x = x.permute(1, 0, 3, 2).contiguous()
        # output: [M*V, B, T, c_out]
        output = self.gc(g, x, PERSON_NUM, JOINT_NUM)
        # [M*V, B, T, c_out] --> [M, V, B, T, c_out]
        output = output.reshape(PERSON_NUM, V, B, T, C)
        # [M, V, B, T, c_out] --> [B*M, c_out, T, V]
        output = output.permute(2, 0, 4, 3, 1).contiguous().view(B*PERSON_NUM, C, T, V)
        # return with the shape: [B*M, c_out, T, V]
        return torch.relu(output)

