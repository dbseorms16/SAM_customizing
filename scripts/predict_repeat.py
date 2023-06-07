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
from predict_parser import parser_init
import sys
import winsound as sd
Craft_On = True

if Craft_On:
    from predict_parser import parser_init, parser_craft
    from craft.craft import CRAFT
    import torch


np.set_printoptions(threshold=sys.maxsize)
def main(args: argparse.Namespace, c_args) -> None:
    if Craft_On:
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
    
    
    
    ## 한 이미지에 여러 visual을 부여할때 결과
    # dict = {
    #     '3' : {
    #         # 가운데
    #         'input_point': np.array([[410, 220]]),
    #         # 'input_point': np.array([[410, 290]]),
    #         # 'input_point': np.array([[410, 365]]),
    #         'label' : np.array([1, 1, 1]),
    #         'input_box' : True,
    #         'craft' : True,
    #         'point_based' : False
    #         },
    #     }
    
    dict = {
        '1' : {
            # 'input_point': np.array([[352, 139]]),
            # 'input_point': np.array([[354, 392]]),
            # 'label' : np.array([1]),
            
            # 가운데
            'input_point': np.array([[352, 139], [354, 392]]),
            'label' : np.array([1, 1]),
            # 4개 꼭지점
            
            # 'input_point': np.array([[45, 99], [44,187], [651,103], [655,189],
            #                          [116,354], [637, 353], [115,430], [638, 427]
            #                          ]),
            # 'label' : np.array([1, 1, 1, 1, 1, 1, 1, 1]),

            # 4개 꼭지점 + 가운데
            # 'input_point': np.array([[45, 99], [44,187], [651,103], [655,189],
            #              [116,354], [637, 353], [115,430], [638, 427],      [352, 139], [354, 392]
            #              ]),
            # 'label' : np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'input_box' : None,
            'craft' : True,
            'point_based' : True
            
            },
        
        '2' : {
            'input_point': np.array([[172, 165], [170, 200], [168, 240], [168, 281], [165, 321],
                                     [485, 137], [484, 175], [485, 219], [483, 266], [485, 318]]),
            'label' : np.array([1,1,1,1,1,1,1,1,1,1]),
            'input_box' : None,
            'craft' : True,
            'point_based' : True
            
            # 'label' : np.array([1])
            },
        
        '3' : {
            # 'input_point': np.array([[684, 580]]),
            # 'label' : np.array([1]),
            'input_point': np.array([[681, 308], [684, 580]]),
            'label' : np.array([1, 1]),
            # 'input_point': np.array([[433, 221], [457, 372], [926, 247], [916,368],
            #                          [834, 488], [992, 473], [447, 602], [430, 575]]),
            # 'label' : np.array([1, 1,1,1,1,1,1,1]),
            
            # 'input_point': np.array([[433, 221], [457, 372], [926, 247], [916,368],
            #                          [834, 488], [992, 473], [447, 602], [430, 575], [681, 308], [684, 580]]),
            # 'label' : np.array([1,1,1,1,1,1,1,1,1,1]),
            
            'input_box' : None,
            'craft' : True,
            'point_based' : True
            
            },
        
        '4' : {
            'input_point': np.array([[716, 514]]),
            'label' : np.array([1]),
            # 'input_point': np.array([[501, 444], [487, 551], [1214, 816], [1198, 254]]),
            # 'label' : np.array([1, 1, 1, 1]),
            # 'input_point': np.array([[501, 444], [487, 551], [1214, 816], [1198, 254], [716, 514]]),
            # 'label' : np.array([1, 1, 1, 1, 1]),
            
            'input_box' : None,
            'craft' : True,
            'point_based' : True
            
            },
        
        '5' : {
            'input_point': np.array([[766, 524]]),
            'label' : np.array([1]),
            # 'input_point': np.array([[693, 437], [776, 446], [769, 656], [794, 581]]),
            # 'label' : np.array([ 1, 1, 1, 1]),
            
            # 'input_point': np.array([[693, 437], [776, 446], [769, 656], [794, 581], [766, 524]]),
            # 'label' : np.array([1, 1, 1, 1, 1]),
            
            'input_box' : None,
            'craft' : True,
            'point_based' : True
            
            }
    }
    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        
        
        filename = t.split('\\')[-1].split('.')[0]
        inputs = dict[filename]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        # masks = generator.generate(image)
         # box_based = True
        for r in range (1, 2):
            if inputs['craft']:
                point_based =  inputs['point_based']
                input_point, input_label, transformed_boxes, input_box = Craft_inference(net, predictor, image, c_args, point_based)
                box_based = True if point_based is False else False
            else:            
                input_point = inputs['input_point']
                # x = input_point[0][0]
                # y = input_point[0][1]
                # input_point = np.array([[x+ (r*20), y + int(r*1)], [x+ (r*20), y - int(r*1)], [x - (r*20), y - int(r*1)]]) 
                input_label = inputs['label']
                input_box = inputs['input_box']
                box_based = False
            
            if box_based:
                masks, scores, logits = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes = transformed_boxes,
                    multimask_output=False)
            else:
                # point promt
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box = None if input_box is None else input_box[None, :],
                    multimask_output=True)
                
            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(args.output)
            os.makedirs(save_base, exist_ok=True)
            gt_mask = np.load(f'./gt/gt_{filename}.npy')
            ## please fix here!
            # gt_mask = np.load(f'./mask/3_indivisual/1.npy')
            if box_based:
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box in input_box:
                    show_box(box.cpu().numpy(), plt.gca())
                plt.axis('off')
                plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
                plt.close()
                # plt.figure(figsize=(10,10))
                # plt.imshow(image)
                # pred_mask = masks.detach().cpu().numpy().astype(int)
                # iou, f_score, precision, recall = calculate_metrics(pred_mask, gt_mask)
                
                # for mask in masks:
                #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                # for box in input_box:
                #     show_box(box.cpu().numpy(), plt.gca())
                    
                # print(scores.shape)
                # score = score.detach().cpu().numpy()[0]

                # plt.title(f"Mask {1}, IOU: {iou:.3f}, F-score: {f_score:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, Confidence: {score:.3f}", fontsize=12)
                # plt.axis('off')
                # filename = f"{base}.png"
                # plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
                # plt.close()
            
            else:            
                # new_index = np.argmax(scores)
                # score = scores[new_index]
                # mask = masks[new_index]
                
                # plt.figure(figsize=(10,10))
                # plt.imshow(image)
                # ##mask save
                # pred_mask = mask.astype(int) 
                
                # iou, f_score, precision, recall = calculate_metrics(pred_mask, gt_mask)
                # # np.save(f'./mask/mask1_{i}', mask.astype(int))
                # show_mask(mask, plt.gca())
                # # show_box(input_box, plt.gca())
                # show_points(input_point, input_label, plt.gca())

                # plt.title(f"Mask {1}, IOU: {iou:.3f}, F-score: {f_score:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, Confidence: {score:.3f}", fontsize=12)
                # plt.axis('off')
                # filename = f"{base}_{r}.png"
                # plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
                # plt.close()

                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    ##mask save
                    pred_mask = mask.astype(int) 
                    
                    iou, f_score, precision, recall = calculate_metrics(pred_mask, gt_mask)
                    # np.save(f'./mask/mask1_{i}', mask.astype(int))
                    show_mask(mask, plt.gca())
                    # show_box(input_box, plt.gca())
                    show_points(input_point, input_label, plt.gca())
                    

                    plt.title(f"Mask {i+1}, IOU: {iou:.3f}, F-score: {f_score:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, Confidence: {score:.3f}", fontsize=12)
                    plt.axis('off')
                    filename = f"{base}_{r}_{i}.png"
                    plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
                    plt.close()

            plt.figure(figsize=(10,10))
            plt.imshow(image)
            ##mask save
            show_mask(gt_mask, plt.gca(), GT=True)
            # show_box(input_box, plt.gca())
            plt.title(f"Mask GT", fontsize=12)
            plt.axis('off')
            filename = f"{base}_gt.png"
            plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
            plt.close()
    
    fr = 1000    # range : 37 ~ 32767
    du = 800     # 1000 ms ==1second
    sd.Beep(fr, du) 
    print("Done!")



if __name__ == "__main__":
    args = parser_init()
    c_args = parser_craft()
    
    main(args, c_args)
