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
Custom ops for efficient resampling of 2D images.
"""

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore._checkparam import Validator

from . import conv2d_gradfix


class Pad(ms.nn.Cell):
    """
    Pad operator, output the image after padding.

    Args:
        paddings (tuple): Paddings.
        mode (str): Padding mode. Options = ["CONSTANT", "REFLECT" or "SYMMETRIC"]. Default: "CONSTANT".

    Inputs:
        - **x** (Tensor) - input tensor.

    Outputs:
        Tensor, padding output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pad_out = Pad(paddings=p)
    """

    def __init__(self, paddings, mode="CONSTANT"):
        """Initialize Pad."""
        super(Pad, self).__init__()
        self.mode = mode
        self.paddings = paddings
        Validator.check_string(self.mode, ["CONSTANT", "REFLECT", "SYMMETRIC"], 'mode', self.cls_name)
        if not isinstance(paddings, tuple):
            raise TypeError(f"For '{self.cls_name}', the type of 'paddings' must be tuple, "
                            f"but got {type(paddings).__name__}.")
        for item in paddings:
            if len(item) != 2:
                raise ValueError(f"For '{self.cls_name}', the dimension of 'paddings' must be (n, 2), "
                                 f"but got {paddings}.")
        if mode == "CONSTANT":
            self.pad = P.Pad(self.paddings)
        else:
            self.paddings = Tensor(np.array(self.paddings), mstype.int64)
            self.pad = P.MirrorPad(mode=mode)

    def construct(self, x):
        """Pad construct"""
        if self.mode == "CONSTANT":
            x = self.pad(x)
        else:
            x = self.pad(x, self.paddings)
        return x


def parse_scaling(scaling):
    """
    Return the padding.

    Args:
        scaling (int): padding parameter.

    Returns:
        int, x scaling parameter.
        int, x scaling parameter.

    Examples:
        >>> padx0, padx1, pady0, pady1 = parse_padding(padding)
    """

    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    sx, sy = scaling
    return sx, sy


def parse_padding(padding):
    """
    Return the padding.

    Args:
        padding (int): padding parameter.

    Returns:
        int, x0 scaling parameter.
        int, x1 scaling parameter.
        int, y0 scaling parameter.
        int, y1 scaling parameter.

    Examples:
        >>> padx0, padx1, pady0, pady1 = parse_padding(padding)
    """

    if isinstance(padding, int):
        padding = [padding, padding]
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    return padding


def get_filter_size(f):
    """
    Get the size of the filter.

    Args:
        f (Tensor): Filter tensor.

    Returns:
        int, Filter width.
        int, Filter height.

    Examples:
        >>> fw, fh = get_filter_size(f)
    """

    if f is None:
        return 1, 1
    fw = f.shape[-1]
    fh = f.shape[0]
    return fw, fh


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, conv_info=None):
    """
    Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:
    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).
    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.
    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.
    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x (Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (Tensor):  Float32 FIR filter of the shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
        up (int): Integer upsampling factor. Can be a single int or a list/tuple `[x, y]`. Default: 1.
        down (int): Integer downsampling factor. Can be a single int or a list/tuple `[x, y]`. Default: 1.
        padding (int): Padding with respect to the upsampled image. Can be a single number
            or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`. Default: 0.
        flip_filter (bool): False = convolution, True = correlation. Default: False.
        gain (int): Overall scaling factor for signal magnitude. Default: 1.
        conv_info: Information of convolutional operators. Default: None.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.

    Examples:
        >>> x = upfirdn2d(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain, \
                          conv_info=conv_info)
    """

    # Validate arguments.
    if f is None:
        f = ms.ops.Ones()((1, 1), ms.float32)
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = parse_scaling(up)
    downx, downy = parse_scaling(down)
    padx0, padx1, pady0, pady1 = parse_padding(padding)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    pad1 = Pad(paddings=((0, 0), (0, 0), (0, 0), (0, upy - 1), (0, 0), (0, upx - 1)))
    x = pad1(x)
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    pad2 = Pad(paddings=((0, 0), (0, 0), (max(pady0, 0), max(pady1, 0)), (max(padx0, 0), max(padx1, 0))))
    x = pad2(x)
    max_y0 = max(-pady0, 0)
    max_y1 = max(-pady1, 0)
    max_x0 = max(-padx0, 0)
    max_x1 = max(-padx1, 0)
    x = x[:, :, max_y0: x.shape[2] - max_y1, max_x0: x.shape[3] - max_x1]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.astype(x.dtype)
    if not flip_filter:
        f_dim = f.ndim
        f = mnp.flip(f, list(range(f_dim)))

    # Convolve with the filter.
    ff = f.astype(mnp.float32)
    ff = mnp.tile(ff[np.newaxis, np.newaxis], ([num_channels, 1] + [1] * f.ndim))
    f = ff.astype(x.dtype)
    if f.ndim == 4:
        x = conv2d_gradfix.conv2d(x_input=x, weight=f, conv_info=conv_info)
    else:
        x = conv2d_gradfix.conv2d(x_input=x, weight=f.unsqueeze(2), conv_info=conv_info)
        x = conv2d_gradfix.conv2d(x_input=x, weight=f.unsqueeze(3), conv_info=conv_info)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


def filter2d(x, f, padding=0, flip_filter=False, gain=1, conv_info=None):
    """
    Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x (Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (Tensor): Float32 FIR filter of the shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
        padding (int): Padding with respect to the output. Can be a single number or a
            list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`. Default: 0.
        flip_filter (bool): False = convolution, True = correlation. Default: False.
        gain (int): Overall scaling factor for signal magnitude. Default: 1.
        conv_info: Information of convolutional operators. Default: None.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.

    Examples:
        >>> x = filter2d(x, f, conv_info=conv_info)
    """

    padx0, padx1, pady0, pady1 = parse_padding(padding)
    fw, fh = get_filter_size(f)
    p = [padx0 + fw // 2, padx1 + (fw - 1) // 2, pady0 + fh // 2, pady1 + (fh - 1) // 2]
    return upfirdn2d(x, f, padding=p, flip_filter=flip_filter, gain=gain, conv_info=conv_info)


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, conv_info=None):
    """
    Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x (Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (Tensor): Float32 FIR filter of the shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
        up (int): Integer upsampling factor. Can be a single int or a list/tuple `[x, y]`. Default: 2.
        padding (int): Padding with respect to the upsampled image. Can be a single number
            or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`. Default: 0.
        flip_filter (bool): False = convolution, True = correlation. Default: False.
        gain (int): Overall scaling factor for signal magnitude. Default: 1.
        conv_info: Information of convolutional operators. Default: None.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.

    Examples:
        >>> x = upsample2d(x, resample_filter, conv_info=conv_info)
    """

    upx, upy = parse_scaling(up)
    padx0, padx1, pady0, pady1 = parse_padding(padding)
    fw, fh = get_filter_size(f)
    p = [padx0 + (fw + upx - 1) // 2, padx1 + (fw - upx) // 2, pady0 + (fh + upy - 1) // 2, pady1 + (fh - upy) // 2]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy, conv_info=conv_info)


def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1, conv_info=None):
    """
    Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x (Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (Tensor): Float32 FIR filter of the shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
        down (int): Integer downsampling factor. Can be a single int or a list/tuple `[x, y]`. Default: 2.
        padding (int): Padding with respect to the upsampled image. Can be a single number
            or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`. Default: 0.
        flip_filter (bool): False = convolution, True = correlation. Default: False.
        gain (int): Overall scaling factor for signal magnitude. Default: 1.
        conv_info: Information of convolutional operators. Default: None.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.

    Examples:
        >>> x = downsample2d(x, resample_filter, conv_info=conv_info)
    """

    downx, downy = parse_scaling(down)
    padx0, padx1, pady0, pady1 = parse_padding(padding)
    fw, fh = get_filter_size(f)
    p = [padx0 + (fw - downx + 1) // 2, padx1 + (fw - downx) // 2,
         pady0 + (fh - downy + 1) // 2, pady1 + (fh - downy) // 2]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, conv_info=conv_info)
