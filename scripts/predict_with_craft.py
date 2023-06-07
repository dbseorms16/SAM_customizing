# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import argparse
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from predict_utils import *

from predict_parser import parser_init, parser_craft
from craft.craft import CRAFT
import torch

def main(args: argparse.Namespace, c_args) -> None:
    print("Loading Craft")
    net = CRAFT()     # initialize
    net.load_state_dict(copyStateDict(torch.load('./craft_mlt_25k.pth')))
    net = net.cuda()
    
    
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    # output_mode = "binary_mask" 
    amg_kwargs = get_amg_kwargs(args)
    # generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    predictor = SamPredictor(sam)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        
        targets = [os.path.join(args.input, f) for f in targets]
    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        
        
        fname = t.split('\\')[-1].split('.')[0]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        box_based = True
        # box_based = False

        bboxes, polys, score_text =  Craft_inference(net, predictor, image, c_args, point_based)

        text = []
        label = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2).reshape((-1, 1, 2))
            if box_based:
                #box
                text.append([poly[0][0][0], poly[0][0][1], poly[2][0][0], poly[2][0][1]])
            else:
                #point
                text.append([int((poly[2][0][0] + poly[0][0][0]) // 2), int((poly[0][0][1] + poly[2][0][1]) // 2)])
                label.append(1)
        
        if box_based:
            #boxes
            input_boxes = torch.tensor(text, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        
        else:
            #point
            input_point = np.array(text)
            input_label = np.array(label)
        
        
        predictor.set_image(image)

        # masks = generator.generate(image)
        if box_based:
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes = transformed_boxes,
                multimask_output=False,
            )
        else:
            # point promt
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box = None,
                multimask_output=True,
            )
        
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output)
        os.makedirs(save_base, exist_ok=True)
        
        if box_based:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box in input_boxes:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.savefig(os.path.join(save_base, fname), bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else :
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_mask(mask, plt.gca())
                # show_box(input_box, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                filename = f"{fname}_{i}.png"
                plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
                plt.close()
            
    
    print("Done!")



if __name__ == "__main__":
    args = parser_init()
    c_args = parser_craft()
    
    main(args, c_args)