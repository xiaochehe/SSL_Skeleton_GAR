import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import random
from model.layers import SageGraph
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

class DecodeGcn(nn.Module):
    
    def __init__(self, in_channels, out_channels, k_num,
                 kernel_size=1, stride=1, padding=0,
                 dilation=1, dropout=0.5, bias=True):
        super(DecodeGcn, self).__init__()

        self.k_num = k_num
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels*(k_num), 
                              kernel_size=kernel_size,
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_skl):      # x: [64, 256, 21] = N, d, V
        x = self.conv(x)
        x = self.dropout(x)
        n, kc, v = x.size()
        x = x.view(n, (self.k_num), kc//(self.k_num), v)          # [64, 4, 256, 21]
        x = torch.einsum('nkcv,kvw->ncw', (x, A_skl))           # [64, 256, 21]
        return x.contiguous()




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
            self.graphsage = SageGraph(out_channels, out_channels, 'mean')
            self.sageresidual = lambda x: x
            self.sagetcn1 = Shift_tcn(out_channels, out_channels, stride=1)
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
            gcnx = self.tcn1(self.gcn1(x)) + self.residual(x)
            if M > 1:
                sagex = self.graphsage(gcnx, M, V)
            else:
                sagex = gcnx
            x = self.sagetcn1(sagex) + self.sageresidual(gcnx)
            # gcnx = self.gcn1(x)
            # if M > 1:
            #     sagex = self.graphsage(gcnx, M, V)
            # else:
            #     sagex = gcnx
            # tcnx = self.tcn1(sagex)
            # x = tcnx + self.residual(x)
        else:
            x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Encoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, distillation=False):
        super(Encoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(6, 64, A, residual=False, sage=True)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, sage=True)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, sage=True)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A, sage=False)


    def forward(self, x):
        global N, C, T, V, M
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        x = self.l10(x)


        # N*M,C,T,V
        c_new = x.size(1)
        unsuper_feature = x.mean(2)
        x = x.view(N, M, c_new, -1)
        group_x = x.mean(3).mean(1)
        single_x = x.mean(3)
        return unsuper_feature, group_x, single_x
        # return self.fc(group_x), self.fc_single(single_x), group_x, single_x

class Decoder(nn.Module):
   
    def __init__(self, n_in_dec, n_hid_dec, graph, graph_args_j, edge_weighting=True, dropout=0.3, **kwargs):
        super(Decoder, self).__init__()
        Graph = import_class(graph)
        self.graph = Graph(**graph_args_j)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        k_num, self.V = self.A.size(0), self.A.size(1)
        if edge_weighting:
            self.emul = nn.Parameter(torch.ones(self.A.size()))
            self.eadd = nn.Parameter(torch.ones(self.A.size()))
        else:
            self.emul = 1
            self.eadd = nn.Parameter(torch.ones(self.A.size()))

        self.msg_in = DecodeGcn(n_hid_dec, n_hid_dec, k_num)

        self.input_r = nn.Linear(n_in_dec, n_hid_dec, bias=True)
        self.input_i = nn.Linear(n_in_dec, n_hid_dec, bias=True)
        self.input_n = nn.Linear(n_in_dec, n_hid_dec, bias=True)

        self.hidden_r = nn.Linear(n_hid_dec, n_hid_dec, bias=False)
        self.hidden_i = nn.Linear(n_hid_dec, n_hid_dec, bias=False)
        self.hidden_h = nn.Linear(n_hid_dec, n_hid_dec, bias=False)

        self.out_fc1 = nn.Linear(n_hid_dec, n_hid_dec)
        self.out_fc2 = nn.Linear(n_hid_dec, n_hid_dec)
        self.out_fc3 = nn.Linear(n_hid_dec, 3)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.mask = torch.ones(78).cuda().detach()
        self.zero_idx = torch.tensor([5, 11, 17, 23, 45, 49, 50,54, 55, 63, 67, 68 ,72, 73]).cuda().detach()
        self.mask[self.zero_idx] = 0.

    def step_forward(self, x, hidden, step):                     # inputs: [64, 21, 3]; hidden: [64, 256, 21]=N, hid, V
        N, V, d = x.size()
        if step<10:
            msg = self.msg_in(hidden, self.A*self.emul+self.eadd)    # msg: [64, 256, 21]=N, hid, V
        else:
            msg = hidden
        msg, hidden = msg.permute(0, 2, 1), hidden.permute(0, 2, 1)             # msg: [64, 21, 256]=N, V, hid, hidden: [64, 21, 256]
        r = torch.sigmoid(self.input_r(x) + self.hidden_r(msg))            # r: [64, 21, 256]=N, V, hid
        z = torch.sigmoid(self.input_i(x) + self.hidden_i(msg))            # z: [64, 21, 256]=N, V, hid
        n = torch.tanh(self.input_n(x) + r*self.hidden_h(msg))             # n: [64, 21, 256]=N, V, hid
        hidden = (1-z)*n + z*hidden                                             # hidden: [64, 21, 256]
        pred = hidden.new_zeros((N, V, 3))

        hidd = hidden
        hidd = self.dropout1(self.leaky_relu(self.out_fc1(hidd)))
        hidd = self.dropout2(self.leaky_relu(self.out_fc2(hidd)))
        pred = self.out_fc3(hidd)
        pred_ = x[:,:,:3] + pred                         # pred_: [64, 21, 3]
        hidden = hidden.permute(0, 2, 1)                                        # hidden: [64, 256, 21] for next convolution
        return pred_, hidden, pred                                              # pred: [64, 21, 3], hidden: [64, 256, 21]
        
    def forward(self, inputs, inputs_previous, inputs_previous2, hidden, t):      # inputs:[64, 1, 63];  hidden:[64, 256, 21]
        pred_all = []
        res_all = []

        N, T, D = inputs.size()
        inputs = inputs.contiguous().view(N, T, self.V, -1)                        # [64, 1, 21, 3]
        inputs_previous = inputs_previous.contiguous().view(N, T, self.V, -1)
        inputs_previous2 = inputs_previous2.contiguous().view(N, T, self.V, -1)
        # self.mask = self.mask.view(self.V, 3)

        for step in range(0, t):
            if step < 1:
                ins_p = inputs[:, 0, :, :]                                         # ins_p: [64, 21, 3]
                ins_v = (inputs_previous-inputs_previous2)[:, 0, :, :]                          # ins_v: [64, 21, 3]
                ins_a = ins_p-inputs_previous[:, 0, :, :]-ins_v
                ins_v_dec = (inputs-inputs_previous)[:, 0, :, :]
            elif step==1:
                ins_p = pred_all[step-1]                                           # ins_p: [64, 21, 3]
                ins_v = (inputs-inputs_previous)[:, 0, :, :]                                   # ins_v: [64, 21, 3]
                ins_a = ins_p-inputs[:, 0, :, :]-ins_v # ins_v-(inputs-inputs_previous)[:, 0, :, :]
                ins_v_dec = pred_all[step-1]-inputs[:, 0, :, :]
            elif step==2:
                ins_p = pred_all[step-1]
                ins_v = pred_all[step-2]-inputs[:, 0, :, :]
                ins_a = ins_p-pred_all[step-2]-ins_v # ins_v-(pred_all[step-2]-inputs[:, 0, :, :])
                ins_v_dec = pred_all[step-1]-pred_all[step-2]
            else:
                ins_p = pred_all[step-1]
                ins_v = pred_all[step-2]-pred_all[step-3]
                ins_a = ins_p-pred_all[step-2]-ins_v # ins_v-(pred_all[step-2]-pred_all[step-3])
                ins_v_dec = pred_all[step-1]-pred_all[step-2]
            n = torch.randn(ins_p.size()).cuda()*0.0005
            ins = torch.cat((ins_p+n, ins_v, ins_a), dim=-1)
            pred_, hidden, res_ = self.step_forward(ins, hidden, step)
            pred_all.append(pred_)                                                 # [t, 64, 21, 3]
            res_all.append(res_)

        preds = torch.stack(pred_all, dim=1)                                       # [64, t, 21, 3]
        reses = torch.stack(res_all, dim=1)
        # preds = preds * self.mask
       
        return preds.transpose(1, 2).contiguous()      # [64, 21, t, 3]

class Model(nn.Module):
    
    def __init__(self,num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=6, distillation=False, n_in_dec=9, n_hid_dec=256, **kwargs):
        super(Model, self).__init__()
        self.encoder = Encoder(num_class, num_point, num_person, graph, graph_args, in_channels, distillation)
        self.decoder = Decoder(n_in_dec, n_hid_dec, graph, graph_args, **kwargs)

    def forward(self, dec_curr, dec_prev, dec_prev2, x, t):
        hidden,_,_ = self.encoder(x)
        # print(hidden.shape)
        pred = self.decoder(dec_curr, dec_prev, dec_prev2, hidden, t)
        return pred



class Classifier(nn.Module):
    """
    linear classifer for group or person level classification
    """
    def __init__(self, feature_shape=256, group_class=8, person_class=10):
        super(Classifier, self).__init__()
        self.linear_classifier_group = nn.Linear(feature_shape, group_class)
        self.linear_classifier_person = nn.Linear(feature_shape, person_class)

    def forward(self, group_feature, person_pred):
        """
        docstring
        """
        group_pred = self.linear_classifier_group(group_feature)
        person_pred = self.linear_classifier_person(person_feature)
        return group_pred, person_pred

