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
Test the performance with one picture.
"""

import argparse
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import mindspore
import mindspore.dataset.vision as vision
from mindspore import Tensor
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore import ops
from skimage.color import rgb2lab

from model.model import ColorizationModel
from utils.utils import decode


def load_image(image_path):
    """load image"""
    image = Image.open(image_path)
    resize = vision.Resize(256)
    random_crop = vision.RandomCrop(224)
    image = random_crop(resize(image))
    image = rgb2lab(image)[:, :, 0] - 50.
    image = Tensor(image, dtype=mindspore.float32)
    expand_dims = mindspore.ops.ExpandDims()
    image = expand_dims(image, 0)
    return image


def infer(opt):
    """test model"""
    net = ColorizationModel()
    param_dict = load_checkpoint(opt.ckpt_path)
    load_param_into_net(net, param_dict)
    colorizer = Model(net)
    image = load_image(opt.img_path)
    image = ops.expand_dims(image, 0)
    img_ab_313 = colorizer.predict(image)
    out_max = np.argmax(img_ab_313[0].asnumpy(), axis=0)
    print('out_max', set(out_max.flatten()))
    color_img = decode(image, img_ab_313, opt.resource)
    plt.imsave('%s_infer150.png', color_img)


def parse_args():
    """import parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams3.jpg')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/net_00.ckpt')
    parser.add_argument('--resource', type=str, default='./resources/')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=1, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    infer(args)
