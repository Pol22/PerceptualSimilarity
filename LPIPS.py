import tensorflow as tf
import torch
import torch.nn as nn
from collections import namedtuple
from torchvision import models as tv


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64,64)):
    # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    scale_factor_H = 1. * out_HW[0] / in_H
    scale_factor_W = 1. * out_HW[1] / in_W
    return nn.Upsample(scale_factor=(scale_factor_H, scale_factor_W),
                       mode='bilinear', align_corners=False)(in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            'shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer(
            'scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs",
            ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(
            h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class PNetLinVGG16(nn.Module):
    def __init__(self, use_dropout=True, spatial=False, lpips=True):
        super(PNetLinVGG16, self).__init__()

        self.spatial = spatial
        self.lpips = lpips
        self.scaling_layer = ScalingLayer()

        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)

        self.net = vgg16()

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input = self.scaling_layer(in0)
        in1_input = self.scaling_layer(in1)
        outs0 = self.net.forward(in0_input)
        outs1 = self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk] = normalize_tensor(outs0[kk])
            feats1[kk] = normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = []
        if(self.lpips):
            if(self.spatial):
                for kk in range(self.L):
                    res.append(upsample(self.lins[kk].model(diffs[kk]),
                                        out_HW=in0.shape[2:]))
            else:
                for kk in range(self.L):
                    res.append(spatial_average(self.lins[kk].model(diffs[kk]),
                                               keepdim=True))
        else:
            if(self.spatial):
                for kk in range(self.L):
                    res.append(upsample(diffs[kk].sum(dim=1, keepdim=True),
                                        out_HW=in0.shape[2:]))
            else:
                for kk in range(self.L):
                    res.append(spatial_average(
                        diffs[kk].sum(dim=1, keepdim=True), keepdim=True))

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val


class LearnedPerceptualImagePatchSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name='LPIPS', **kwargs):
        super(LearnedPerceptualImagePatchSimilarity, self).__init__(
            name=name, **kwargs)
        self.evaluator = PNetLinVGG16()
        # TODO fix use gpu
        use_gpu = False
        kw = {}
        if not use_gpu:
            kw['map_location'] = 'cpu'
        self.evaluator.load_state_dict(
            torch.load('models/weights/v0.1/vgg.pth', **kw),
            strict=False)
        self.evaluator.eval()
        # TODO
        # if(use_gpu):
        #     self.net.to(gpu_ids[0])
        #     self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        in0 = tf.transpose(y_true, perm=(0, 3, 1, 2))
        in1 = tf.transpose(y_pred, perm=(0, 3, 1, 2))
        # normalize to [-1, 1]
        in0 = in0 * 2. - 1.
        in1 = in1 * 2. - 1.
        in0 = torch.from_numpy(in0.numpy())
        in1 = torch.from_numpy(in1.numpy())
        lpips = self.evaluator.forward(in0, in1)
        # TODO fix with cuda
        self.sum.assign_add(tf.reduce_sum(lpips.detach().numpy()))
        self.count.assign_add(1.)

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)

from PIL import Image
import numpy as np

metric = LearnedPerceptualImagePatchSimilarity()
img0 = Image.open('./imgs/ex_ref.png')
img1 = Image.open('./imgs/ex_p0.png')
# img0 = Image.open('DIV2K_valid_HR/0801.png')
# img1 = Image.open('DIV2K_valid_LR_bicubic/X2/0801x2.png')
# img1 = img1.resize((2040, 1356), Image.BICUBIC)
img0 = np.asarray(img0, dtype=np.float32)
img1 = np.asarray(img1, dtype=np.float32)
img0 = img0 / 255.
img1 = img1 / 255.
img0 = np.expand_dims(img0, axis=0)
img1 = np.expand_dims(img1, axis=0)
metric.update_state(img0, img1)
print(metric.result())