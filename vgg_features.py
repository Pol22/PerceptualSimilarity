import tensorflow as tf
from PIL import Image
import numpy as np


class ScalingLayer(tf.keras.Model):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = tf.constant([-.030, -.088, -.188], shape=(1, 1, 1, 3),
                                 dtype=tf.float32)
        self.scale = tf.constant([.458, .448, .450], shape=(1, 1, 1, 3),
                                 dtype=tf.float32)

    def call(self, inputs):
        return (inputs - self.shift) / self.scale


def NetLinLayer(chn_in, chn_out=1, use_dropout=False):
    inputs = tf.keras.Input(shape=(None, None, chn_in))
    if use_dropout:
        x = tf.keras.layers.Dropout(rate=.5)(inputs)
    else:
        x = inputs
    x = tf.keras.layers.Conv2D(chn_out, 1, use_bias=False)(x)
    return tf.keras.Model(inputs, x)


def upsample(in_tens, out_HW=(64,64)):
    # assumes scale factor is same for H and W
    return tf.image.resize(in_tens, size=out_HW,
                           method=tf.image.ResizeMethod.BILINEAR)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = tf.sqrt(tf.reduce_sum(in_feat ** 2, axis=3, keepdims=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdims=True):
    return tf.reduce_mean(in_tens, axis=(1, 2), keepdims=keepdims)


class VGG16(tf.keras.Model):
    def __init__(self, require_grad=False, pretrained=True):
        super(VGG16, self).__init__()
        vgg_pretrained = tf.keras.applications.VGG16()
        vgg_pretrained.trainable = False
        for layer in vgg_pretrained.layers:
            layer.trainable =  False
        self.slice1 = tf.keras.Sequential(vgg_pretrained.layers[0:3])
        self.slice2 = tf.keras.Sequential(vgg_pretrained.layers[3:6])
        self.slice3 = tf.keras.Sequential(vgg_pretrained.layers[6:10])
        self.slice4 = tf.keras.Sequential(vgg_pretrained.layers[10:14])
        self.slice5 = tf.keras.Sequential(vgg_pretrained.layers[14:18])

    def call(self, inputs):
        out = []
        x = self.slice1(inputs)
        out.append(x)
        x = self.slice2(x)
        out.append(x)
        x = self.slice3(x)
        out.append(x)
        x = self.slice4(x)
        out.append(x)
        x = self.slice5(x)
        out.append(x)
        return out


class PNetLin(tf.keras.Model):
    def __init__(self, use_dropout=True, spatial=False):
        super(PNetLin, self).__init__()
        self.spatial = spatial
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)

        self.net = VGG16()

        self.lins = []
        # TODO import weights from TORCH model
        for i in range(self.L):
            self.lins.append(NetLinLayer(self.chns[i],
                                         use_dropout=use_dropout))

    def call(self, in0, in1):
        in0_input = self.scaling_layer(in0)
        in1_input = self.scaling_layer(in1)
        # print(tf.transpose(in1, perm=(0, 3, 1, 2)))
        # in0_input = tf.expand_dims(in0, axis=0)
        # in1_input = tf.expand_dims(in0, axis=0)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk] = normalize_tensor(outs0[kk])
            feats1[kk] = normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = []
        if self.spatial:
            for kk in range(self.L):
                res.append(upsample(self.lins[kk](diffs[kk]),
                                    out_HW=in0.shape[1:3]))
        else:
            for kk in range(self.L):
                print(spatial_average(self.lins[kk](diffs[kk]),
                                           keepdims=True))
                res.append(spatial_average(self.lins[kk](diffs[kk]),
                                           keepdims=True))

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        return val.numpy().squeeze()


img0 = Image.open('./imgs/ex_ref.png')
img1 = Image.open('./imgs/ex_p0.png')
img0 = np.asarray(img0, dtype=np.float32)
img1 = np.asarray(img1, dtype=np.float32)
img0 = img0 / 255. * 2. - 1.
img1 = img1 / 255. * 2. - 1.
img0 = np.expand_dims(img0, axis=0)
img1 = np.expand_dims(img1, axis=0)





f = PNetLin()
val = f(img0, img1)
print(val)
