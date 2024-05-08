import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

from PIL import Image
import datasets.transforms as T


#Will remain the same
model_config_path = "config/DINO/DINO_4scale.py" 


#Put model path here.
model_checkpoint_path = "../../input/dino_trained/pytorch/dino/1/dino.pth" # change the path of the model checkpoint

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()


# dataset_val = build_dataset(image_set='val', args=args)

#Put images path here.   
coco_path = "../../input/cocodataset/coco_1k"

parent_dir_path = os.path.dirname(coco_path)

if not os.path.exists(os.path.join(parent_dir_path,"predictions")):
    os.makedirs(os.path.join(parent_dir_path,"predictions"),exist_ok=True)


for file_name in os.listdir(coco_path):
    if (not file_name.lower().endswith("png")):
        continue
    img_path = os.path.join(coco_path, file_name)
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image, _ = transform(img, None)
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    thershold = 0.05 # set a thershold
    # vslzr = COCOVisualizer()
    scores = output['scores']   
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    scrs = scores[select_mask]
    boxes = boxes[select_mask]
    out_txt = coco_path+"/../predictions/"+file_name[:-4]+"_preds.txt"
    content = ""
    for j in range(len(boxes)):
            f=True
            temp = str(boxes[j][0].item())+" "+str(boxes[j][1].item())+" "+str(boxes[j][2].item())+" "+str(boxes[j][3].item())+" "+str(scrs[j].item())+"\n"
            content = content+temp
    if (content!=""):
         with open(out_txt, 'w') as file:
            file.write(content)

