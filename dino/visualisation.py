import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
model_checkpoint_path = "../logs/Sanity4/checkpoint_best_regular.pth" # change the path of the model checkpoint

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

args.dataset_file = 'coco'
args.coco_path = "../coco_1k" # the path of coco
args.fix_size = False

dataset_val = build_dataset(image_set='val', args=args)   
for i in range(0,len(dataset_val)):
    image, targets = dataset_val[i]

    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    thershold = 0.25 # set a thershold
    vslzr = COCOVisualizer()
    scores = output['scores']   
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    box_label = ["0" for item in labels[select_mask]]
    pred_dict = {
        'image_id' : targets['image_id'],
        'boxes': boxes[select_mask],
        'size': targets['size'],
        'box_label': box_label
    }
    vslzr.visualize(image, pred_dict, savedir='validation_visualisation')
