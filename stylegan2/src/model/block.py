# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Stylegan2 blocks
"""

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Parameter, Tensor
from mindspore.nn import CellList

from utils.ops import conv2d_resample, upfirdn2d, bias_act

# Resample filter's values captured from pre-trained model's checkpoint
resample_filter = Tensor([[0.0156, 0.0469, 0.0469, 0.0156],
                          [0.0469, 0.1406, 0.1406, 0.0469],
                          [0.0469, 0.1406, 0.1406, 0.0469],
                          [0.0156, 0.0469, 0.0469, 0.0156]], ms.float32)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    """
    Normalize 2nd moment.

    Args:
        x (Tensor): Input tensor.
        dim (int): Axis. Default: 1.
        eps (float): Small value added to the denominator. Default: 1e-8.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> x = normalize_2nd_moment(x)
    """

    sqrt = ops.Sqrt()
    square = ops.Square()
    return x / sqrt(square(x).mean(axis=dim, keep_dims=True) + eps)


def modulated_conv2d(x, weight, styles, noise=None, up=1, down=1, padding=0, r_filter=None,
                     demodulate=True, flip_weight=True, fused_modconv=True, conv_info=None):
    """
    Modulated conv2d.

    Args:
        x (Tensor): Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight (Tensor): Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles (Tensor): Modulation coefficients of shape [batch_size, in_channels].
        noise (Tensor): Optional noise tensor to add to the output activations. Default: None.
        up (int): Integer upsampling factor. Default: 1.
        down (int): Integer downsampling factor. Default: 1.
        padding (int): Padding with respect to the upsampled image. Default: 0.
        r_filter (Tensor): Low-pass filter to apply when resampling activations. Default: None.
        demodulate (bool): Apply weight demodulation?  False = convolution, True = correlation. Default: True.
        flip_weight (bool): Need to flip the weight?. Default: True.
        fused_modconv (bool): Perform modulation, convolution, and demodulation
            as a single fused operation?. Default: True.
        conv_info (list): Information of convolutional operators. Default: None.

    Returns:
        Tensor, modulated output tensor.

    Examples:
        >>> x = modulated_conv2d(x, weight, styles, conv_info=conv_info)
    """

    square = ops.Square()
    sqrt = ops.Sqrt()
    batch_size = x.shape[0]
    _, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == ms.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw)) / weight.max(axis=[1, 2, 3], keepdims=True)
        styles = styles / styles.max(axis=1, keepdims=True)

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    reshape = ops.Reshape()
    if demodulate or fused_modconv:
        w = ops.ExpandDims()(weight, 0)
        w = w * reshape(styles, (batch_size, 1, -1, 1, 1))
    if demodulate:
        dcoefs = 1 / sqrt(square(w).sum(axis=2).sum(axis=2).sum(axis=2) + 1e-8)
    if demodulate and fused_modconv:
        w = w * reshape(dcoefs, (batch_size, -1, 1, 1, 1))

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.astype(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.astype(x.dtype), f=r_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight, conv_info=conv_info)
        if demodulate and noise is not None:
            x = x * dcoefs.astype(x.dtype).reshape(batch_size, - 1, 1, 1) + noise.astype(x.dtype)
        elif demodulate:
            x = x * dcoefs.astype(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x + noise.astype(x.dtype)
        return x

    # Execute as one fused op using grouped convolution.
    x = reshape(x, (1, -1, x.shape[2], x.shape[3]))
    w = reshape(w, (-1, in_channels, kh, kw))
    x = conv2d_resample.conv2d_resample(x=x, w=w.astype(x.dtype), f=r_filter, up=up, down=down, padding=padding,
                                        groups=batch_size, flip_weight=flip_weight, conv_info=conv_info)
    x = reshape(x, (batch_size, -1, x.shape[2], x.shape[3]))
    if noise is not None:
        x = x + noise
    return x


class FullyConnectedLayer(nn.Cell):
    """
    Fully Connected Layer.

    Args:
        in_features (int):  Number of input features.
        out_features(int):  Number of output features.
        bias (bool):  Apply additive bias before the activation function?. Default: True.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'linear'.
        lr_multiplier (float): Learning rate multiplier. Default: 1.
        bias_init (int): Initial value for the additive bias. Default: 0.

    Inputs:
        - **x** (Tensor) - Input tensor.

    Outputs:
        Tensor, fully connected layer output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = FullyConnectedLayer(in_features, out_features, lr_multiplier=lr_multiplier)
        >>> x = layer(x)
    """

    def __init__(self, in_features, out_features, bias=True, activation='linear', lr_multiplier=1, bias_init=0):
        super().__init__()
        self.activation = activation
        self.weight = Parameter(Tensor(np.random.randn(out_features, in_features) / lr_multiplier, ms.float32))
        self.bias = Parameter(Tensor(np.full([out_features], np.float32(bias_init)), ms.float32)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def construct(self, x):
        """Fully_connected_layer construct"""
        w = self.weight.astype(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.astype(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = ops.ExpandDims()(b, 0) + ops.matmul(x, w.transpose())
        else:
            x = ops.matmul(x, w.T)
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class Conv2dLayer(nn.Cell):
    """
    Conv2d Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels(int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        bias (bool):  Apply additive bias before the activation function? Default: True.
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'Linear'.
        up (int): Integer upsampling factor. Default: 1.
        down (int): Integer downsampling factor. Default: 1.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        trainable (bool): Update the weights of this layer during training? Default: True.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **gain** (int) - Gain on act_gain. Default: 1.
        - **conv_info** (list) - Information of convolutional operator. Default: None.

    Outputs:
        Tensor, 2d convolution output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = Conv2dLayer(in_channels, out_channels, kernel_size)
        >>> x = layer(x, conv_info=conv_info)
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
                 activation='linear', up=1, down=1, conv_clamp=None, trainable=True):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation]['def_gain']

        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), ms.float32)
        bias = ops.Zeros()(out_channels, ms.float32) if bias else None
        if trainable:
            self.weight = Parameter(weight)
            self.bias = Parameter(bias) if bias is not None else None
        else:
            self.weight = weight
            if bias is not None:
                self.bias = bias
            else:
                self.bias = None

    def construct(self, x, gain=1, conv_info=None):
        """Conv2d construct"""
        w = self.weight * self.weight_gain
        b = self.bias.astype(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.astype(x.dtype), f=resample_filter,
                                            up=self.up, down=self.down, padding=self.padding,
                                            flip_weight=flip_weight, conv_info=conv_info)

        act_gain = self.act_gain * gain
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain)
        return x


class SynthesisLayer(nn.Cell):
    """
    Synthesis Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        resolution (int): Resolution of this layer.
        kernel_size (int): Convolution kernel size. Default: 3.
        up: Integer upsampling factor. Default: 1.
        use_noise (bool): Enable noise input? Default: True
        activation (str): The activation function: 'relu', 'lrelu', etc. Default: 'lrelu'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **w** (Tensor) - Modulation tensor.
        - **noise_mode** (int) - Noise mode, 0 = none, 1 = constant, 2 = random. Default: 2.
        - **fused_modconv** (bool) - Perform modulation, convolution, and demodulation
            as a single fused operation? Default: True.
        - **gain** (int) - Gain on act_gain. Default: 1.
        - **conv_info** (list) - Information of convolutional operators. Default: None.

    Outputs:
        Tensor, synthesis layer output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = SynthesisLayer(in_channels, out_channels, w_dim, resolution, conv_clamp)
        >>> x = layer(x, w, conv_info=conv_info)
    """

    std_normal = ops.StandardNormal()

    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3,
                 up=1, use_noise=True, activation='lrelu', conv_clamp=None):
        super().__init__()
        std_normal = ops.StandardNormal()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation]['def_gain']
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = Parameter(Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size),
                                       ms.float32))
        if use_noise:
            self.noise_const = std_normal((resolution, resolution))
            self.noise_strength = Parameter(ops.Zeros()((), ms.float32))
        self.bias = Parameter(ops.Zeros()(out_channels, ms.float32))

    def construct(self, x, w, noise_mode=1, fused_modconv=True, gain=1, conv_info=None):
        """Synthesis layer construct"""
        styles = self.affine(w)
        noise = None
        if self.use_noise and noise_mode == 2:
            noise = ops.StandardNormal()((x.shape[0], 1, self.resolution, self.resolution)) * self.noise_strength
        if self.use_noise and noise_mode == 1:
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
                             padding=self.padding, r_filter=resample_filter,
                             flip_weight=flip_weight, fused_modconv=fused_modconv, conv_info=conv_info)

        act_gain = self.act_gain * gain
        x = bias_act.bias_act(x, self.bias.astype(x.dtype), act=self.activation, gain=act_gain)
        return x


class ToRGBLayer(nn.Cell):
    """
    To RGB Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        kernel_size (int): Convolution kernel size. Default: 1.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **w** (Tensor) - Modulation tensor.
        - **fused_modconv** (bool) - Perform modulation, convolution and demodulation
            as a single fused operation? Default: True.
        - **conv_info** (list) - Information of convolutional operators. Default: None.

    Outputs:
        Tensor, to rgb layer output tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> layer = ToRGBLayer(in_channels, out_channels, w_dim, kernel_size, conv_clamp)
        >>> x = layer(x, w, conv_info=conv_info)
    """

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = Parameter(ops.StandardNormal()((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = Parameter(ops.Zeros()(out_channels, ms.float32))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def construct(self, x, w, fused_modconv=True, conv_info=None):
        """To rgb layer construct"""
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False,
                             fused_modconv=fused_modconv, conv_info=conv_info)
        x = bias_act.bias_act(x, self.bias.astype(x.dtype))
        return x


class SynthesisBlock(nn.Cell):
    """
    Synthesis Block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        w_dim (int): Intermediate latent (W) dimensionality.
        resolution (int): Resolution of this layer.
        output_res (int): Resolution of final output image to differentiate between ffhq1024 and lsun_car512x384
        img_channels (int): Number of output color channels.
        is_last (bool): Is this the last block?
        architecture (str): Architecture: 'orig', 'skip', 'resnet'. Default: 'skip'.
        conv_clamp (bool): Clamp the output to +-X, None = disable clamping. Default: None.
        use_fp16 (bool): Use FP16 for this block? Default: False.
        batch_size (int): Batch size. Default: 1.
        train (bool): True = train, False = infer. Default: False.
        layer_kwargs (dict): Arguments for SynthesisLayer.

    Inputs:
        - **x** (Tensor) - Input feature.
        - **img** (Tensor) - Input image.
        - **ws** (Tensor) - Intermediate latents.
        - **force_fp32** (bool) - If force the input to float32. Default: False.
        - **fused_modconv** (bool) - Perform modulation, convolution and demodulation
            as a single fused operation? Default: None.
        - **layer_kwargs** (int) - Noise mode, 0 = none, 1 = constant, 2 = random. Default: 2.

    Outputs:
        Tensor, output feature.
        Tensor, output image.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> block = SynthesisBlock(in_channels, out_channels, w_dim, resolution, output_res, img_channels, is_last)
        >>> x, img = block(x, img, ws)
    """

    def __init__(self, in_channels, out_channels, w_dim, resolution, output_res, img_channels, is_last,
                 architecture='skip', conv_clamp=None, use_fp16=False, batch_size=1, train=False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.out_res = output_res
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.size = batch_size
        self.train = train
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = Parameter(ops.StandardNormal()((out_channels, resolution, resolution)))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        conv_clamp=conv_clamp, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp)
            self.num_torgb += 1

        if self.out_res == 512:
            self.name_list = ['conv2d'] * 2 + (['transpose2d'] + ['conv2d'] * 4) * 7
            self.dtype_list = ['float32' for _ in range(len(self.name_list))]

        if self.out_res == 1024:
            self.name_list = ['conv2d'] * 2 + (['transpose2d'] + ['conv2d'] * 4) * 8
            self.dtype_list = ['float32' for _ in range(len(self.name_list))]

        if train:
            input_list = [(self.size, 512, 4, 4), (self.size, 512, 4, 4), (self.size, 512, 4, 4),
                          (self.size, 512, 11, 11), (self.size, 512, 8, 8), (self.size, 3, 11, 11),
                          (self.size, 512, 8, 8), (self.size, 512, 8, 8), (self.size, 512, 19, 19),
                          (self.size, 512, 16, 16), (self.size, 3, 19, 19), (self.size, 512, 16, 16),
                          (self.size, 512, 16, 16), (self.size, 512, 35, 35), (self.size, 512, 32, 32),
                          (self.size, 3, 35, 35), (self.size, 512, 32, 32), (self.size, 512, 32, 32),
                          (self.size, 512, 67, 67), (self.size, 512, 64, 64), (self.size, 3, 67, 67),
                          (self.size, 512, 64, 64), (self.size, 512, 64, 64), (self.size, 256, 131, 131),
                          (self.size, 256, 128, 128), (self.size, 3, 131, 131), (self.size, 256, 128, 128),
                          (self.size, 256, 128, 128), (self.size, 128, 259, 259), (self.size, 128, 256, 256),
                          (self.size, 3, 259, 259), (self.size, 128, 256, 256), (self.size, 128, 256, 256),
                          (self.size, 64, 515, 515), (self.size, 64, 512, 512), (self.size, 3, 515, 515),
                          (self.size, 64, 512, 512)]

            weight_list = [(512, 512, 3, 3), (3, 512, 1, 1), (512, 512, 3, 3),
                           (512, 1, 4, 4), (512, 512, 3, 3), (3, 1, 4, 4),
                           (3, 512, 1, 1), (512, 512, 3, 3), (512, 1, 4, 4),
                           (512, 512, 3, 3), (3, 1, 4, 4), (3, 512, 1, 1),
                           (512, 512, 3, 3), (512, 1, 4, 4), (512, 512, 3, 3),
                           (3, 1, 4, 4), (3, 512, 1, 1), (512, 512, 3, 3),
                           (512, 1, 4, 4), (512, 512, 3, 3), (3, 1, 4, 4),
                           (3, 512, 1, 1), (512, 256, 3, 3), (256, 1, 4, 4),
                           (256, 256, 3, 3), (3, 1, 4, 4), (3, 256, 1, 1),
                           (256, 128, 3, 3), (128, 1, 4, 4), (128, 128, 3, 3),
                           (3, 1, 4, 4), (3, 128, 1, 1), (128, 64, 3, 3),
                           (64, 1, 4, 4), (64, 64, 3, 3), (3, 1, 4, 4), (3, 64, 1, 1)]

            if self.out_res == 512:
                self.input_list = input_list
                self.weight_list = weight_list

            if self.out_res == 1024:
                input_list.extend([(self.size, 64, 512, 512), (self.size, 32, 1027, 1027), (self.size, 32, 1024, 1024),
                                   (self.size, 3, 1027, 1027), (self.size, 32, 1024, 1024)])
                self.input_list = input_list
                weight_list.extend([(64, 32, 3, 3), (32, 1, 4, 4),
                                    (32, 32, 3, 3), (3, 1, 4, 4), (3, 32, 1, 1)])
                self.weight_list = weight_list

            conv_list = CellList([self._conv2d(512, 512, 3, 1, 1, 1, 1, 0),
                                  self._conv2d(512, 3, 1, 1, 0, 1, 1, 1),
                                  self._transpose2d(512, 512, 3, 2, 0, 1, 1, 2),
                                  self._conv2d(512, 512, 4, 1, 0, 1, 512 * 1, 3),
                                  self._conv2d(512, 512, 3, 1, 1, 1, 1, 4),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 5),
                                  self._conv2d(512, 3, 1, 1, 0, 1, 1, 6),
                                  self._transpose2d(512, 512, 3, 2, 0, 1, 1, 7),
                                  self._conv2d(512, 512, 4, 1, 0, 1, 512, 8),
                                  self._conv2d(512, 512, 3, 1, 1, 1, 1, 9),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 10),
                                  self._conv2d(512, 3, 1, 1, 0, 1, 1, 11),
                                  self._transpose2d(512, 512, 3, 2, 0, 1, 1, 12),
                                  self._conv2d(512, 512, 4, 1, 0, 1, 512, 13),
                                  self._conv2d(512, 512, 3, 1, 1, 1, 1, 14),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 15),
                                  self._conv2d(512, 3, 1, 1, 0, 1, 1, 16),
                                  self._transpose2d(512, 512, 3, 2, 0, 1, 1, 17),
                                  self._conv2d(512, 512, 4, 1, 0, 1, 512, 18),
                                  self._conv2d(512, 512, 3, 1, 1, 1, 1, 19),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 20),
                                  self._conv2d(512, 3, 1, 1, 0, 1, 1, 21),
                                  self._transpose2d(512, 256, 3, 2, 0, 1, 1, 22),
                                  self._conv2d(256, 256, 4, 1, 0, 1, 256, 23),
                                  self._conv2d(256, 256, 3, 1, 1, 1, 1, 24),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 25),
                                  self._conv2d(256, 3, 1, 1, 0, 1, 1, 26),
                                  self._transpose2d(256, 128, 3, 2, 0, 1, 1, 27),
                                  self._conv2d(128, 128, 4, 1, 0, 1, 128, 28),
                                  self._conv2d(128, 128, 3, 1, 1, 1, 1, 29),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 30),
                                  self._conv2d(128, 3, 1, 1, 0, 1, 1, 31),
                                  self._transpose2d(128, 64, 3, 2, 0, 1, 1, 32),
                                  self._conv2d(64, 64, 4, 1, 0, 1, 64, 33),
                                  self._conv2d(64, 64, 3, 1, 1, 1, 1, 34),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 35),
                                  self._conv2d(64, 3, 1, 1, 0, 1, 1, 36)])

            if self.out_res == 512:
                self.conv_list = conv_list

            if self.out_res == 1024:
                conv_list.extend([self._transpose2d(64, 32, 3, 2, 0, 1, 1, 37),
                                  self._conv2d(32, 32, 4, 1, 0, 1, 32, 38),
                                  self._conv2d(32, 32, 3, 1, 1, 1, 1, 39),
                                  self._conv2d(3, 3, 4, 1, 0, 1, 3, 40),
                                  self._conv2d(32, 3, 1, 1, 0, 1, 1, 41)])
                self.conv_list = conv_list

        else:
            input_list = [(1, 512 * self.size, 4, 4), (1, 512 * self.size, 4, 4), (1, 512 * self.size, 4, 4),
                          (1, 512 * self.size, 11, 11), (1, 512 * self.size, 8, 8), (self.size, 3, 11, 11),
                          (1, 512 * self.size, 8, 8), (1, 512 * self.size, 8, 8), (1, 512 * self.size, 19, 19),
                          (1, 512 * self.size, 16, 16), (self.size, 3, 19, 19), (1, 512 * self.size, 16, 16),
                          (1, 512 * self.size, 16, 16), (1, 512 * self.size, 35, 35), (1, 512 * self.size, 32, 32),
                          (self.size, 3, 35, 35), (1, 512 * self.size, 32, 32), (self.size, 512, 32, 32),
                          (self.size, 512, 67, 67), (self.size, 512, 64, 64), (self.size, 3, 67, 67),
                          (self.size, 512, 64, 64), (self.size, 512, 64, 64), (self.size, 256, 131, 131),
                          (self.size, 256, 128, 128), (self.size, 3, 131, 131), (self.size, 256, 128, 128),
                          (self.size, 256, 128, 128), (self.size, 128, 259, 259), (self.size, 128, 256, 256),
                          (self.size, 3, 259, 259), (self.size, 128, 256, 256), (self.size, 128, 256, 256),
                          (self.size, 64, 515, 515), (self.size, 64, 512, 512), (self.size, 3, 515, 515),
                          (self.size, 64, 512, 512)]

            weight_list = [(512 * self.size, 512, 3, 3), (3 * self.size, 512, 1, 1), (512 * self.size, 512, 3, 3),
                           (512 * self.size, 1, 4, 4), (512 * self.size, 512, 3, 3), (3, 1, 4, 4),
                           (3 * self.size, 512, 1, 1), (512 * self.size, 512, 3, 3), (512 * self.size, 1, 4, 4),
                           (512 * self.size, 512, 3, 3), (3, 1, 4, 4), (3 * self.size, 512, 1, 1),
                           (512 * self.size, 512, 3, 3), (512 * self.size, 1, 4, 4), (512 * self.size, 512, 3, 3),
                           (3, 1, 4, 4), (3 * self.size, 512, 1, 1), (512, 512, 3, 3),
                           (512, 1, 4, 4), (512, 512, 3, 3), (3, 1, 4, 4),
                           (3, 512, 1, 1), (512, 256, 3, 3), (256, 1, 4, 4),
                           (256, 256, 3, 3), (3, 1, 4, 4), (3, 256, 1, 1),
                           (256, 128, 3, 3), (128, 1, 4, 4), (128, 128, 3, 3),
                           (3, 1, 4, 4), (3, 128, 1, 1), (128, 64, 3, 3),
                           (64, 1, 4, 4), (64, 64, 3, 3), (3, 1, 4, 4), (3, 64, 1, 1)]

            if self.out_res == 512:
                self.input_list = input_list
                self.weight_list = weight_list
                conv_list = CellList([self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 0),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 1),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 2),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 3),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 4),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 5),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 6),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 7),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 8),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 9),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 10),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 11),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 12),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 13),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 14),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 15),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 16),
                                      self._transpose2d(512, 512, 3, 2, 0, 1, 1, 17),
                                      self._conv2d(512, 512, 4, 1, 0, 1, 512, 18),
                                      self._conv2d(512, 512, 3, 1, 1, 1, 1, 19),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 20),
                                      self._conv2d(512, 3, 1, 1, 0, 1, 1, 21),
                                      self._transpose2d(512, 256, 3, 2, 0, 1, 1, 22),
                                      self._conv2d(256, 256, 4, 1, 0, 1, 256, 23),
                                      self._conv2d(256, 256, 3, 1, 1, 1, 1, 24),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 25),
                                      self._conv2d(256, 3, 1, 1, 0, 1, 1, 26),
                                      self._transpose2d(256, 128, 3, 2, 0, 1, 1, 27),
                                      self._conv2d(128, 128, 4, 1, 0, 1, 128, 28),
                                      self._conv2d(128, 128, 3, 1, 1, 1, 1, 29),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 30),
                                      self._conv2d(128, 3, 1, 1, 0, 1, 1, 31),
                                      self._transpose2d(128, 64, 3, 2, 0, 1, 1, 32),
                                      self._conv2d(64, 64, 4, 1, 0, 1, 64, 33),
                                      self._conv2d(64, 64, 3, 1, 1, 1, 1, 34),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 35),
                                      self._conv2d(64, 3, 1, 1, 0, 1, 1, 36)])
                self.conv_list = conv_list

            if self.out_res == 1024:
                self.input_list = input_list[0:17] + \
                                  [(1, 512 * self.size, 32, 32), (1, 512 * self.size, 67, 67),
                                   (1, 512 * self.size, 64, 64), (self.size, 3, 67, 67),
                                   (1, 512 * self.size, 64, 64)] + input_list[22:] + \
                                  [(self.size, 64, 512, 512), (self.size, 32, 1027, 1027), (self.size, 32, 1024, 1024),
                                   (self.size, 3, 1027, 1027), (self.size, 32, 1024, 1024)]

                self.weight_list = weight_list[0:17] + [(512 * self.size, 512, 3, 3), (512 * self.size, 1, 4, 4),
                                                        (512 * self.size, 512, 3, 3), (3, 1, 4, 4),
                                                        (3 * self.size, 512, 1, 1)] + weight_list[22:] + \
                                   [(64, 32, 3, 3), (32, 1, 4, 4),
                                    (32, 32, 3, 3), (3, 1, 4, 4), (3, 32, 1, 1)]

                conv_list = CellList([self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 0),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 1),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 2),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 3),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 4),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 5),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 6),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 7),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 8),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 9),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 10),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 11),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 12),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 13),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 14),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 15),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 16),
                                      self._transpose2d(512 * self.size, 512 * self.size, 3, 2, 0, 1, self.size, 17),
                                      self._conv2d(512 * self.size, 512 * self.size, 4, 1, 0, 1, 512 * self.size, 18),
                                      self._conv2d(512 * self.size, 512 * self.size, 3, 1, 1, 1, self.size, 19),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 20),
                                      self._conv2d(512 * self.size, 3 * self.size, 1, 1, 0, 1, self.size, 21),
                                      self._transpose2d(512, 256, 3, 2, 0, 1, 1, 22),
                                      self._conv2d(256, 256, 4, 1, 0, 1, 256, 23),
                                      self._conv2d(256, 256, 3, 1, 1, 1, 1, 24),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 25),
                                      self._conv2d(256, 3, 1, 1, 0, 1, 1, 26),
                                      self._transpose2d(256, 128, 3, 2, 0, 1, 1, 27),
                                      self._conv2d(128, 128, 4, 1, 0, 1, 128, 28),
                                      self._conv2d(128, 128, 3, 1, 1, 1, 1, 29),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 30),
                                      self._conv2d(128, 3, 1, 1, 0, 1, 1, 31),
                                      self._transpose2d(128, 64, 3, 2, 0, 1, 1, 32),
                                      self._conv2d(64, 64, 4, 1, 0, 1, 64, 33),
                                      self._conv2d(64, 64, 3, 1, 1, 1, 1, 34),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 35),
                                      self._conv2d(64, 3, 1, 1, 0, 1, 1, 36),
                                      self._transpose2d(64, 32, 3, 2, 0, 1, 1, 37),
                                      self._conv2d(32, 32, 4, 1, 0, 1, 32, 38),
                                      self._conv2d(32, 32, 3, 1, 1, 1, 1, 39),
                                      self._conv2d(3, 3, 4, 1, 0, 1, 3, 40),
                                      self._conv2d(32, 3, 1, 1, 0, 1, 1, 41)])
                self.conv_list = conv_list

        self.conv_list_weight = ms.ParameterTuple(self.conv_list.get_parameters())
        for param in self.conv_list_weight:
            param.requires_grad = False
        self.conv_info = [self.conv_list, self.conv_list_weight, self.input_list, self.weight_list, self.name_list]

    def _conv2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, num):
        """
        Conv2d operator.

        Args:
            in_channels (int): Input channel.
            out_channels (int): Output channel.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            dilation(int): Dilation.
            group (int): Group.
            num (int): The number of the operator in all_conv.

        Returns:
            Cell, function of nn.Conv2d with given weight shape.

        Examples:
            >>> func = _conv2d(512, 512, 3, 1, 1, 1, 1, 0)
        """
        if self.dtype_list[num] == 'float32':
            func = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=group,
                             has_bias=False, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float32))
        else:
            func = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=group,
                             has_bias=False, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float16))
        return func

    def _transpose2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, num):
        """
        Transpose2d operator.

        Args:
            in_channels (int): Input channel.
            out_channels (int): Output channel.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            dilation(int): Dilation.
            group (int): Group.
            num (int): The number of the operator in all_conv.

        Returns:
            Cell, function of nn.Conv2dTranspose with given weight shape.

        Examples:
            >>> func = _transpose2d(512, 512, 3, 2, 0, 1, 1, 2)
        """
        if self.dtype_list[num] == 'float32':
            func = nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                                      group=group, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float32))
        else:
            func = nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                                      group=group, weight_init=Tensor(np.ones(self.weight_list[num]), ms.float16))
        return func

    def construct(self, x, img, ws, force_fp32=False, fused_modconv=None, noise_mode=0):
        """Synthesis block construct"""
        w_iter = iter(ops.Unstack(axis=1)(ws))
        d_type = ms.float16 if self.use_fp16 and not force_fp32 else ms.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (d_type == ms.float32 or x.shape[0] == 1)

        # Input.
        d_type = ms.float32
        if self.in_channels == 0:
            x = self.const.astype(d_type)
            x = mnp.tile(ops.ExpandDims()(x, 0), (ws.shape[0], 1, 1, 1))
        else:
            x = x.astype(d_type)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter)[0].astype(d_type), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           conv_info=self.conv_info)
        else:
            x = self.conv0(x, next(w_iter)[0].astype(d_type), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           conv_info=self.conv_info)
            x = self.conv1(x, next(w_iter)[0].astype(d_type), fused_modconv=fused_modconv, noise_mode=noise_mode,
                           conv_info=self.conv_info)

        # ToRGB.
        if img is not None:
            img = upfirdn2d.upsample2d(img, resample_filter, conv_info=self.conv_info)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter)[0], fused_modconv=fused_modconv, conv_info=self.conv_info)
            y = y.astype(ms.float32)
            img = img + y if img is not None else y
        return x, img
