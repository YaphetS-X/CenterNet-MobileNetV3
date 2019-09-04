#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project:        Helmet Detector
   Author:         Leo
-------------------------------------------------
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓          ┏┓
            ┏┛┻━━━━━━━━━━┛┻┓
            ┃      ☃       ┃
            ┃  ┳━┛    ┗━┳  ┃
            ┃      ┻       ┃
            ┗━━━┓        ┏━┛
                ┃        ┗━━━━━━ ┓
                ┃    神兽保佑          ┣┓
                ┃　  永无BUG！         ┏┛
                ┗━┓ ┓ ┏━━┳ ┓ ┏━┛
                   ┃  ┫ ┫  ┃ ┫ ┫
                   ┗━┻━┛  ┗━┻━┛
-------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from opts import opts
from lib.detectors.ctdet import CtdetDetector

time_stats = ['tot', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  detector = CtdetDetector(opt)
  cam = cv2.VideoCapture(0)
  while True:
    _, img = cam.read()
    ret = detector.run(img)
    time_str = ''
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)
    if cv2.waitKey(1) == 27:
        return  # esc to quit

if __name__ == '__main__':
  opt = opts().init()
  print(opt)
  demo(opt)
