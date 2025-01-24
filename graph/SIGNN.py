import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            self.reset_parameters_uniform()
        elif init == 'xavier':
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)#.unsqueeze(0).expand(input.size(0), -1, -1))
        output = torch.spmm(adj, support)  #torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Feat2Graph(nn.Module):
    def __init__(self, num_feats):
        super(Feat2Graph, self).__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

    def forward(self, x):
        qx = self.wq(x)
        kx = self.wk(x)

        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)
        return x, adj

class before_RPN_samfuse(nn.Module):
    def __init__(self, iter):
        super(before_RPN_samfuse, self).__init__()
        self.iter = iter
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1)
        self.conv_fuse = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=1)
        self.linear_e = nn.Linear(384, 384, bias=False)
        self.gate = nn.Conv2d(384, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv_fusion = nn.Conv2d(384 * 2, 384, kernel_size=3, padding=1, bias=True)
        self.ConvGRU = ConvGRUCell(384, 384, kernel_size=1)

    def generate_attention(self, exemplar, query):
        fea_size = exemplar.size()[2:]
        exemplar_flat = exemplar.view(1, 384, -1)
        query_flat = query.view(1, 384, -1)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, 384, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input1_mask = self.gate_s(input1_mask)
        input1_att = input1_att * input1_mask

        return input1_att
    def layer2_fused_module(self, tar_feat):
        feat0 = F.interpolate(tar_feat[0], size=tar_feat[2].shape[-2:], mode='bilinear', align_corners=True)
        feat1 = F.interpolate(tar_feat[1], size=tar_feat[2].shape[-2:], mode='bilinear', align_corners=True)
        feat3 = F.interpolate(tar_feat[3], size=tar_feat[2].shape[-2:], mode='bilinear', align_corners=True)
        feat4 = F.interpolate(tar_feat[4], size=tar_feat[2].shape[-2:], mode='bilinear', align_corners=True)
        return (feat0 + feat1 + tar_feat[2] + feat3 + feat4) / 5

    def layer2_restore_module(self, tar_result, tar_feat, is_tar):
        if is_tar:
            tar_result = self.conv2(tar_result)
        feat0 = F.interpolate(tar_result, size=tar_feat[0].shape[-2:], mode='bilinear', align_corners=True)
        feat1 = F.interpolate(tar_result, size=tar_feat[1].shape[-2:], mode='bilinear', align_corners=True)
        feat2 = tar_result
        feat3 = F.interpolate(tar_result, size=tar_feat[3].shape[-2:], mode='bilinear', align_corners=True)
        feat4 = F.interpolate(tar_result, size=tar_feat[4].shape[-2:], mode='bilinear', align_corners=True)
        return [feat0+tar_feat[0], feat1+tar_feat[1], feat2+tar_feat[2], feat3+tar_feat[3], feat4+tar_feat[4]]


    def layer0_fused_module(self, ref_feat):

        feat0 = ref_feat[0]
        feat1 = F.interpolate(ref_feat[1], size=ref_feat[0].shape[-2:], mode='bilinear', align_corners=True)
        feat2 = F.interpolate(ref_feat[2], size=ref_feat[0].shape[-2:], mode='bilinear', align_corners=True)
        feat3 = F.interpolate(ref_feat[3], size=ref_feat[0].shape[-2:], mode='bilinear', align_corners=True)
        feat4 = F.interpolate(ref_feat[4], size=ref_feat[0].shape[-2:], mode='bilinear', align_corners=True)
        return (feat0 + feat1 + feat2 + feat3 + feat4) / 5

    def layer0_restore_module(self, ref_result, ref_feat):
        ref_result = self.conv2(ref_result)
        feat0 = ref_result
        feat1 = F.interpolate(ref_result, size=ref_feat[1].shape[-2:], mode='bilinear', align_corners=True)
        feat2 = F.interpolate(ref_result, size=ref_feat[2].shape[-2:], mode='bilinear', align_corners=True)
        feat3 = F.interpolate(ref_result, size=ref_feat[3].shape[-2:], mode='bilinear', align_corners=True)
        feat4 = F.interpolate(ref_result, size=ref_feat[4].shape[-2:], mode='bilinear', align_corners=True)
        return [feat0 + ref_feat[0], feat1 + ref_feat[1], feat2 + ref_feat[2], feat3 + ref_feat[3], feat4 + ref_feat[4]]

    def forward(self, tar_feat, ref_feat, match_feat, sam_feat):
        tar_fused = self.layer2_fused_module(tar_feat)
        tars = self.conv_fuse(torch.cat([tar_fused, sam_feat], dim=1))
        tars = self.conv1(tars)

        ref_fused = self.layer0_fused_module(ref_feat)
        refs = self.conv1(ref_fused)

        matchs = self.layer2_fused_module(match_feat)

        batch_size = tar_feat[0].size(0)
        tar_result = torch.zeros(batch_size, 384, tar_fused.size()[2:][0], tar_fused.size()[2:][1]).cuda()
        ref_result = torch.zeros(batch_size, 384, ref_fused.size()[2:][0], ref_fused.size()[2:][1]).cuda()
        match_result = torch.zeros(batch_size, 384, matchs.size()[2:][0], matchs.size()[2:][1]).cuda()
        for i in range(batch_size):
            tar = tars[i,:,:,:][None].contiguous().clone()
            ref = refs[i, :, :, :][None].contiguous().clone()
            match = matchs[i, :, :, :][None].contiguous().clone()
            for j in range(self.iter):
                attention1 = self.conv_fusion(torch.cat([self.generate_attention(ref, tar),
                                                                 self.generate_attention(ref, match)], 1))

                attention2 = self.conv_fusion(torch.cat([self.generate_attention(tar, ref),
                                                                 self.generate_attention(tar, match),], 1))

                attention3 = self.conv_fusion(torch.cat([self.generate_attention(match, ref),
                                                                 self.generate_attention(match, tar)], 1))

                h_v1 = self.ConvGRU(attention1, ref)
                h_v2 = self.ConvGRU(attention2, tar)
                h_v3 = self.ConvGRU(attention3, match)

                ref = h_v1.clone()
                tar = h_v2.clone()
                match = h_v3.clone()

                if j == self.iter - 1:
                    tar_result[i,:,:,:] = tar
                    ref_result[i,:,:,:] = ref
                    match_result[i,:,:,:] = match

        new_tar = self.layer2_restore_module(tar_result, tar_feat, is_tar=True)
        new_ref = self.layer0_restore_module(ref_result, ref_feat)
        new_match = self.layer2_restore_module(match_result, match_feat, is_tar=False)
        return new_match, new_tar, new_ref

class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = int((kernel_size - 1) / 2)
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)


        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update
        #new_state = prev_state + out_inputs
        return new_state


