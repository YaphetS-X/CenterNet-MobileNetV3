from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
from torchvision import transforms
from lib.models.model import create_model, load_model
from lib.utils.image import get_affine_transform
from lib.utils.debugger import Debugger


class BaseDetector(object):
  def __init__(self, opt):

    print('Creating model...')
    self.model = create_model(opt.heads, opt.head_conv)   # {'hm': opt.num_classes, 'wh': 2, 'reg': 2} ,  256
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.cuda()
    self.model.eval()

    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.opt = opt
    transform_val_list = [
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    self.trans_compose = transforms.Compose(transform_val_list)
    del transform_val_list

  def pre_process(self, image):
    height, width = image.shape[0:2]
    inp_height, inp_width = self.opt.input_h, self.opt.input_w
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    # print(c, s)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(
      image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    images = self.trans_compose(inp_image).unsqueeze(0)
    meta = {'c': c, 's': s,
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger()
    start_time = time.time()

    detections = []
    scale_start_time = time.time()
    images, meta = self.pre_process(image)
    images = images.cuda()
    torch.cuda.synchronize()
    pre_process_time = time.time()
    pre_time += pre_process_time - scale_start_time

    output, dets, forward_time = self.process(images, return_time=True)

    torch.cuda.synchronize()
    net_time += forward_time - pre_process_time
    decode_time = time.time()
    dec_time += decode_time - forward_time

    dets = self.post_process(dets, meta, 1)
    torch.cuda.synchronize()
    post_process_time = time.time()
    post_time += post_process_time - decode_time

    detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()

    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time
    self.show_results(debugger, image, results)

    return {'results': results, 'tot': tot_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}