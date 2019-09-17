from __future__ import print_function

import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from data import VOCroot_train, VOCroot_test, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer
import cv2
from tqdm import tqdm
import pdb

from models.MOD_vgg_1125 import build_net



num_classes = 7
net = build_net(512,num_classes)

PCB_CLASSES = ('__background__',  # always index 0
               'missing_hole', 'mouse_bite', 'open_circuit', 'short',
               'spur', 'spurious_copper')

#resume_net_path = os.path.join('weights/MOD_512/1213/MOD_VOC_epoches_160.pth')
resume_net_path = os.path.join('/data/Models/luzs/MOD/weights/MOD_512/MOD_VOC_epoches_70.pth')
print('Loading resume network', resume_net_path)
state_dict = torch.load(resume_net_path)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v

cfg = VOC_512

net.load_state_dict(new_state_dict)
net=torch.nn.DataParallel(net, device_ids=list(range(3)))
net.cuda()

cudnn.benmark = True
detector = Detect(num_classes, 0, cfg)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)
net.eval()
test_dir = '/data/Datasets/luzs/testpcb'

rgb_means = (104, 117, 123)
rgb_std = (1, 1, 1)
transforms = BaseTransform(net.module.size, rgb_means, rgb_std, (2, 0, 1))
print('start test')
for No,testname in tqdm(enumerate(os.listdir(test_dir))):
    img_ori = cv2.imread(os.path.join(test_dir,testname))
    h_ori,w_ori,_ = img_ori.shape
    img = cv2.resize(img_ori,(2048,2048))
    h,w,_ = img.shape
    win_size = 256
    stride = 128
    all_boxes = [[[] for _ in range(225)]
                 for _ in range(num_classes)]
    for r in range(0,h-win_size,stride):
        for c in range(0,w-win_size,stride):
            max_per_image = 300
            thresh = 0.5
            i = int(r//128*15)+int(c//128)
            tmp = img[r:r+win_size, c:c+win_size]
            tmp = cv2.resize(tmp,(512,512))
            x = Variable(transforms(tmp).unsqueeze(0), volatile = True).cuda()
            out = net(x=x,test=True)
            boxes, scores = detector.forward(out, priors)
            boxes = boxes[0]
            scores = scores[0]
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            # scale each detection back up to the image
            scale = torch.Tensor([tmp.shape[1], tmp.shape[0],
                              tmp.shape[1], tmp.shape[0]]).cpu().numpy()
            boxes *= scale

            for j in range(1, num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                if len(inds) == 0:
                    #print(i)
                    #print(j)
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
                cpu = False
                keep = nms(c_dets, 0.45, force_cpu=cpu)
                keep = keep[:50]
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
    mask_ori = np.zeros([h_ori,w_ori])
    #print(mask_ori.shape)
    #pdb.set_trace()
    for x1 in range(1,7):
        for x2 in range(225):
            if len(all_boxes[x1][x2]):
                  for index_im in range(len(all_boxes[x1][x2])):
                      #for index_th in range(len(all_boxes[x1][x2])):
                      #    if index_th 
                      xmi,ymi,xma,yma,scores = all_boxes[x1][x2][index_im]
                      if xmi<=1 or ymi<=1 or xma <= 1 or yma<=1 or xmi>=511 or xma>=511 or ymi>=511 or yma>=511:
                          continue
                      zong = x2//15
                      heng = x2%15
                      xmi = int((xmi/2 + heng*128)*(w_ori/w))
                      ymi = int((ymi/2 + zong*128)*(h_ori/h))
                      xma = int((xma/2 + heng*128)*(w_ori/w))
                      yma = int((yma/2 + zong*128)*(h_ori/h))
                      xzh = (xma+xmi)//2
                      yzh = (yma+ymi)//2
                      if mask_ori[yzh][xzh]==0:
                        cv2.rectangle(img_ori,(xmi,ymi),(xma,yma),(0,255,0),2)
                        cv2.putText(img_ori, PCB_CLASSES[x1], (xmi,ymi), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0, 255, 0) )
                        #mask_ori[ymi:yma][xmi:xma] = 1
                        for he in range(xmi,xma):
                            for zo in range(ymi,yma):
                                mask_ori[zo][he]=1
                      else:
                        continue
     
    cv2.imwrite('/data/Datasets/luzs/PCB/result_60/result%d.jpg'%No,img_ori)
                      
                  
                  
                  
    
    

