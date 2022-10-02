import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import random
from .layers import SageGraph
import sys
sys.path.append("./model/Temporal_shift/")

import dgl
import dgl.function as fn
from dgl import DGLGraph

random.seed(1)
dgl.random.seed(1)

from cuda.shift import Shift

N, C, T, V, M = 0, 0, 0, 0, 0

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

# class GraphSage(nn.Module):
#     """
#     docstring for GraphSAGE
#     """
#     def __init__(self, embeding_size, person_num, agg_func='MEAN'):
#         super(GraphSage, self).__init__()
#         self.person_num = person_num
#         self.embeding_size = embeding_size
#         self.agg_func = agg_func
#         # self.weight = nn.Parameter(torch.FloatTensor(embeding_size, embeding_size * 2))
#         self.linear = nn.Linear(2*embeding_size, embeding_size, bias=False)
#         self.init_params()

#     def init_params(self):
#         nn.init.xavier_uniform_(self.linear.weight)
#         # nn.init.constant_(self.linear.bias, 0)

#     def forward(self, x):
#         aggregate_feats = self.aggregate(x, 8, 12)
#         combined = torch.cat([aggregate_feats,x], dim=1)
#         combined = combined.permute(0, 2, 3, 1)
#         combined = F.relu(self.linear(combined)).permute(0, 3, 1, 2)
#         # combined = x + aggregate_feats
#         return combined

#     def aggregate(self, x, num_sample_person=3, num_sample_joint=3):
#         """
#         docstring
#         """
#         NM, C, T, V = x.size()
#         M = self.person_num
#         N = int(NM/M)
#         colum_indices = []

#         # keyjoint to keyjoint
#         # for i in range(M):
#         #     for j in range(V):
#         #         other_person = [m for m in range(M)]
#         #         other_person.remove(i)
#         #         if self.training:
#         #             person_sample = random.sample(other_person, num_sample_person)
#         #             joint_sample = random.sample(range(V), num_sample_joint)
#         #         else:
#         #             person_sample = other_person
#         #             joint_sample = [i for i in range(V)]
#         #         colum_indices.extend([m*V+n for m in person_sample for n in joint_sample] )
#         # if self.training:
#         #     row_indices = [i for i in range(M*V) for j in range(num_sample_person*num_sample_joint)]
#         # else:
#         #     row_indices = [i for i in range(M*V) for j in range((M -1)*V)]
#         # mask = torch.zeros(M*V, M*V)
#         # mask[row_indices, colum_indices] = 1
#         # if self.agg_func == 'MEAN':
#         #     num_neigh = mask.sum(0, keepdim=True)
#         #     mask = mask.div(num_neigh).to(x.device)         
#         #     x = x.view(N, M, C, T, V).permute(0, 3, 1, 4, 2).contiguous().view(N, T, M * V, C)
#         #     aggregate_feats = torch.matmul(mask, x)
#         #     aggregate_feats = aggregate_feats.view(N, T, M, V, C).permute(0, 2, 4, 1, 3).contiguous().view(N * M, C, T, V)
#         # if self.agg_func == 'MAX':
#         #     mask = mask.t()
#         #     indexs = [index.nonzero() for index in mask==1]
#         #     aggregate_feats = []
#         #     x = x.view(N, M, C, T, V).permute(0, 3, 2, 1, 4).contiguous().view(N, T, C, M * V)
#         #     for feat in [x[..., index.squeeze()] for index in indexs]:
#         #         aggregate_feats.append(torch.max(feat,-1)[0].unsqueeze(-1))
#         #     aggregate_feats = torch.cat(aggregate_feats, -1)
#         #     aggregate_feats = aggregate_feats.view(N, T, C, M, V).permute(0, 3, 2, 1, 4).contiguous().view(N * M, C, T, V)

#         # keyjoint to person
#         for i in range(M):
#             for j in range(V):
#                 other_person = [m for m in range(M)]
#                 other_person.remove(i)
#                 if self.training:
#                     person_sample = random.sample(other_person, num_sample_person)
#                 else:
#                     person_sample = other_person
#                 colum_indices.extend(person_sample)
#         if self.training:
#             row_indices = [i for i in range(M*V) for j in range(num_sample_person)]
#         else:
#             row_indices = [i for i in range(M*V) for j in range((M -1))]
#         mask = torch.zeros(M*V, M)
#         mask[row_indices, colum_indices] = 1
#         if self.agg_func == 'MEAN':
#             num_neigh = mask.sum(0, keepdim=True)
#             mask = mask.div(num_neigh).to(x.device)         
#             x = x.view(N, M, C, T, V).permute(0, 3, 1, 2, 4).contiguous()
#             x = torch.max(x, dim=-1,keepdim=False)[0]
#             aggregate_feats = torch.matmul(mask, x)
#             aggregate_feats = aggregate_feats.view(N, T, M, V, C).permute(0, 2, 4, 1, 3).contiguous().view(N * M, C, T, V)

#         return aggregate_feats




class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x.contiguous())
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant_(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,17,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(17*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(17*in_channels).astype(np.int)
        for i in range(17):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*17)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(17*out_channels).astype(np.int)
        for i in range(17):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*17)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, sage=False):
        super(TCN_GCN_unit, self).__init__()
        self.sage = sage
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        if self.sage:
            # self.graphsage = GraphSage(out_channels, 12)
            self.graphsage = SageGraph(out_channels, out_channels, 'mean')
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.sage:
            gcnx = self.gcn1(x)
            if M > 1:
                sagex = self.graphsage(gcnx, M, V)
            else:
                sagex = gcnx
            tcnx = self.tcn1(sagex)
            x = tcnx + self.residual(x)
        else:
            x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_subclass=2, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=6, distillation=False):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_selfsurp = nn.BatchNorm1d(12 * in_channels * num_point)


        self.l1 = TCN_GCN_unit(6, 64, A, residual=True, sage=True)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A, sage=True)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, sage=True)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, sage=True)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A, sage=True)

        self.fc = nn.Linear(256, num_class)
        self.fc_single = nn.Linear(256, num_subclass)

        self.single_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512)
            )
        self.fc_single_a = nn.Linear(256, 128)
        self.fc_single_b = nn.Linear(256, 128)

        bn_init(self.data_bn, 1)

    def forward(self, x):
        global N, C, T, V, M
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        if M == 12:
            x = self.data_bn(x)
        else:
            x = self.data_bn_selfsurp(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        # x = self.l2(x)
        x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        x = self.l10(x)




        # N*M,C,T,V

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        group_x = x.mean(3).mean(1)
        single_x_feature = x.mean(3)
        # # self-supervised
        single_x_a = self.single_head(single_x_feature)
        single_x = torch.bmm(single_x_a, single_x_a.transpose(1,2))/math.sqrt(single_x_a.size(-1))
        return self.fc(group_x), single_x, group_x, single_x_feature

        # # contrast
        # group_x_a = F.normalize(self.single_head(group_x), dim=1)
        # return self.fc(group_x), group_x_a, group_x, single_x_feature

        # fully-supervised
        # return self.fc(group_x), self.fc_single(single_x_feature)

class Classifier(nn.Module):
    """
    linear classifer for group or person level classification
    """
    def __init__(self, feature_shape=256, group_class=8, person_class=10):
        super(Classifier, self).__init__()
        self.linear_classifier_group = nn.Linear(feature_shape, group_class)
        self.linear_classifier_person = nn.Linear(feature_shape, person_class)

    def forward(self, group_feature, person_feature):
        """
        docstring
        """
        group_pred = self.linear_classifier_group(group_feature)
        person_pred = self.linear_classifier_person(person_feature)
        return group_pred, person_pred

