from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('--load_model', default='models/helmet.pth',
                             help='path to pretrained model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.6,
                             help='visualization threshold.')
    self.parser.add_argument('--num_classes', type=int, default=2,
                             help='num classes', )
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.',)
    self.parser.add_argument('--head_conv', type=int, default=256,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')

  def init(self):
    opt = self.parser.parse_args()
    opt.input_h, opt.input_w = [512, 512]
    opt.heads = {'hm': opt.num_classes, 'wh': 2}
    return opt
