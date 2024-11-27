import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *

import cv2
import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log("Tester start ... ", logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger=None):

    base_model.eval()  # set model to eval mode
    target = "./vis"
    useful_cate = [
        "02691156",  # plane
        "04379243",  # table
        "03790512",  # motorbike
        "03948459",  # pistol
        "03642806",  # laptop
        "03467517",  # guitar
        "03261776",  # earphone
        "03001627",  # chair
        "02958343",  # car
        "04090263",  # rifle
        "03759954",  # microphone
        "random",  # zeroverse random shape
    ]
    
    running_chamfer_distance_l2_list = []
    len_examples = len(test_dataloader)
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if taxonomy_ids[0] not in useful_cate:
                continue
            if taxonomy_ids[0] == "02691156":
                a, b = 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45
            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            elif taxonomy_ids[0] == "random":
                a, b = 90, 135
            else:
                a, b = 0, 0

            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == "ShapeNet":
                points = data.cuda()
            elif "ZeroVerse" in dataset_name:
                points = data.cuda()
            else:
                raise NotImplementedError(f"Train phase do not support {dataset_name}")

            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers, chamfer_distance_l2 = base_model(points, vis=True)
            chamfer_distance_l2 = chamfer_distance_l2.item()
            running_chamfer_distance_l2_list.append(chamfer_distance_l2)
            final_image = []
            # data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            data_path = os.path.join(
                args.experiment_path, f"vis/{taxonomy_ids[0]}_{idx}"
            )
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            print_log(f"[data_path]] ${data_path}$", logger = 'viz_zeroverse')
            chamfer_distance_l2_path = os.path.join(data_path, "chamfer_distance_l2.txt")
            with open(chamfer_distance_l2_path, "a") as f:
                f.write(f"{chamfer_distance_l2}\n")

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, "gt.txt"), points, delimiter=";")

            points = misc.get_ptcloud_img(points, a, b)
            
            points_img_cropped = points[150:650, 150:675, :]
            cv2.imwrite(os.path.join(data_path, "gt.jpg"), points_img_cropped)
            
            final_image.append(points[150:650, 150:675, :])

            centers = centers.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')

            centers = misc.get_ptcloud_img(centers, a, b)
            
            centers_img_cropped = centers[150:650, 150:675, :]
            
            cv2.imwrite(os.path.join(data_path, "center.jpg"), centers_img_cropped)
            final_image.append(centers[150:650, 150:675, :])

            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, "vis.txt"), vis_points, delimiter=";")
            vis_points = misc.get_ptcloud_img(vis_points, a, b)
            vis_points_img_cropped = vis_points[150:650, 150:675, :]
            cv2.imwrite(os.path.join(data_path, "vis.jpg"), vis_points_img_cropped)
            
            final_image.append(vis_points[150:650, 150:675, :])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(
                os.path.join(data_path, "dense_points.txt"), dense_points, delimiter=";"
            )
            dense_points = misc.get_ptcloud_img(dense_points, a, b)
            
            dense_points_img_cropped = dense_points[150:650, 150:675, :]
            cv2.imwrite(os.path.join(data_path, "dense_points.jpg"), dense_points_img_cropped)

            final_image.append(dense_points[150:650, 150:675, :])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f"final_panel_plot.jpg")
            cv2.imwrite(img_path, img)

            if idx > 1500:
                break
        final_data_path = os.path.join(
                args.experiment_path, f"vis"
            )
        chamfer_distance_l2_path = os.path.join(final_data_path, "chamfer_distance_l2.txt")
        
        with open(chamfer_distance_l2_path, "a") as f:
            f.write(f"Mean Chamfer Distance: {np.array(running_chamfer_distance_l2_list).mean()}\n")
        return
