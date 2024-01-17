import csv
import pickle
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
from mmpose_utils import (concat, convert_instance_to_frame, frame_iter,
                          process_mmdet_results)
from tqdm import tqdm
from utils import (getMMposeAnatomicalCocoMarkerNames,
                   getMMposeAnatomicalCocoMarkerPairs,
                   getMMposeAnatomicalMarkerNames,
                   getMMposeAnatomicalMarkerPairs, getMMposeMarkerNames)
from virtualmarker.utils.coord_utils import pixel2cam

from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from VirtualMarker.main.inference import Simple3DMeshInferencer

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
import pickle
from typing import Optional, Sequence, Union

import numpy as np
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dataset import Compose, default_collate, pseudo_collate
from mmpose_constants import get_flip_pair_dict
from mmpose_inference import (init_pose_model, init_test_pipeline,
                              run_pose_inference, run_pose_tracking)
from mmpose_utils import _box2cs, _xyxy2xywh, frame_iter
from torch.utils.data import DataLoader, Dataset

# from mmpose.apis import vis_pose_tracking_result
# from mmpose.datasets import DatasetInfo

SMPL_VERTICES_PATH = "./resource/vertices_keypoints_corr_smpl.csv"

class CustomVideoDataset(Dataset):
    """Create custom video dataset for top down inference

    Args:
        video_path (str): Path to video file
        bbox_path (str): Path to bounding box file
                         (expects format to be xyxy [left, top, right, bottom])
        pipeline (list[dict | callable]): A sequence of data transforms
    """

    def __init__(
        self, video_path, bbox_path, bbox_threshold
    ):
        # load video
        self.capture = cv2.VideoCapture(video_path)
        assert self.capture.isOpened(), f"Failed to load video file {video_path}"
        self.frames = np.stack([x for x in frame_iter(self.capture)])

        # load bbox
        self.bboxs = pickle.load(open(bbox_path, "rb"))
        print(f"Loaded {len(self.bboxs)} frames from {video_path}")
        self.bbox_threshold = bbox_threshold

        # create instance to frame and frame to instance mapping
        self.instance_to_frame = []
        self.frame_to_instance = []
        for i, frame in enumerate(self.bboxs):
            self.frame_to_instance.append([])
            for j, instance in enumerate(frame):
                bbox = instance["bbox"]
                if bbox[4] >= bbox_threshold:
                    self.instance_to_frame.append([i, j])
                    self.frame_to_instance[-1].append(len(self.instance_to_frame) - 1)

    def __len__(self):
        return len(self.instance_to_frame)

    def __getitem__(self, idx):
        frame_num, detection_num = self.instance_to_frame[idx]
        # num_joints = self.cfg.data_cfg["num_joints"]
        bbox_xyxy = self.bboxs[frame_num][detection_num]["bbox"]
        bbox_xywh = _xyxy2xywh(bbox_xyxy)

        # joints_3d and joints_3d_visalble are place holders
        # but bbox in image file, image file is not used but we need bbox information later
        img_h, img_w, _ = self.frames[frame_num].shape
        data = {
            "img": self.frames[frame_num],
            "bbox": bbox_xyxy[None, :4],
            "img_h" : img_h,
            "img_w" : img_w,
            "bbox_score": np.ones(1, dtype=np.float32),
        }

        return data

def load_smp_indices():
    with open(SMPL_VERTICES_PATH, "r", encoding="utf-8-sig") as data:
        augmented_vertices_index = list(csv.DictReader(data))
        augmented_vertices_index_dict = {
            vertex["Name"]: int(vertex["Index"]) for vertex in augmented_vertices_index
        }

    return augmented_vertices_index_dict

def pose_inference_updated(
    model_config,
    model_ckpt,
    video_path,
    bbox_path,
    pkl_path,
    video_out_path,
    device="cuda:0",
    batch_size=8,
    bbox_thr=0.95,
    visualize=True,
    save_results=True,
    marker_set="Anatomical",
):
    """Run pose inference on custom video dataset"""

    # build dataset
    video_basename = video_path.split("/")[-1].split(".")[0]
    dataset = CustomVideoDataset(
        video_path=video_path,
        bbox_path=bbox_path,
        bbox_threshold=bbox_thr,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=pseudo_collate
    )

    ######## MODEL SPECIFIC ###########

    max_person = 1
    load_path_test = 'VirtualMarker/experiment/simple3dmesh_train/baseline_mix/final.pth.tar'
    smpl_indices = load_smp_indices()

    print("Building {} Custom Video Dataset".format(video_basename))

    # run pose inference
    print("Running pose inference...")
    results_instances = []
    for batch in tqdm(dataloader):
        detection_all = np.array([[i, batch["bbox"][i][0][0], batch["bbox"][i][0][1], batch["bbox"][i][0][2], batch["bbox"][i][0][3], 100, 100, 10000] for i in range(len(batch["bbox"]))])
        img_h = batch["img_h"]
        img_w = batch["img_w"]
        inferencer = Simple3DMeshInferencer(load_path=load_path_test, img_path_list=batch["img"], detection_all=detection_all, max_person=max_person)
        inferencer.model.eval()
        results = defaultdict(list)
        with torch.no_grad():
            # result = run_pose_inference(model, batch)
            for i, meta in enumerate(tqdm(inferencer.demo_dataloader, dynamic_ncols=True)):
                for k, _ in meta.items():
                    meta[k] = meta[k].cuda()

                imgs = meta['img'].cuda()
                inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                pose_root = meta['root_cam'].cuda()
                depth_factor = meta['depth_factor'].cuda()

                _, _, _, _, pred_mesh, _, pred_root_xy_img = inferencer.model(imgs, inv_trans, intrinsic_param, pose_root, depth_factor, flip_item=None, flip_mask=None)
                results['pred_mesh'].append(pred_mesh.detach().cpu().numpy())
                results['pose_root'].append(pose_root.detach().cpu().numpy())
                results['pred_root_xy_img'].append(pred_root_xy_img.squeeze(1).squeeze(-1).detach().cpu().numpy())
                results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())

        for term in results.keys():
            results[term] = np.concatenate(results[term])

        pred_mesh = results['pred_mesh']  # (N*T, V, 3)
        pred_root_xy_img = results['pred_root_xy_img']  # (N*T, J, 2)
        pose_root = results['pose_root']  # (N*T, 3)
        focal_l = results['focal_l']
        center_pt = results['center_pt']
        # root modification (differenct root definition betwee VM & VirtualPose)
        new_pose_root = []
        for root_xy, root_cam, focal, center in zip(pred_root_xy_img, pose_root, focal_l, center_pt):
            root_img = np.array([root_xy[0], root_xy[1], root_cam[-1]])
            new_root_cam = pixel2cam(root_img[None,:], center, focal)
            new_pose_root.append(new_root_cam)
        pose_root = np.array(new_pose_root)  # (N*T, 1, 3)
        pred_mesh = pred_mesh + pose_root
        data_samples = []

        print("len(batch['img']):", len(batch["img"]))
        for i in range(len(batch["img"])):
            chosen_mask = detection_all[:, 0] == i
            pred_mesh_T = pred_mesh[chosen_mask]  # (N, V, 3)
            focal_T = focal_l[chosen_mask]  # (N, ...)
            center_pt_T = center_pt[chosen_mask]  # (N, ...)
            intrinsic_matrix = torch.tensor(
                [
                    [focal_T[0][0], 0, center_pt_T[0][0]],
                    [0, focal_T[0][1], center_pt_T[0][1]],
                    [0, 0, 1],
                ]
            )
            single_frame = []
            if len(pred_mesh_T) != 1:
                print("!!!!!!!pred_mesh_T:", pred_mesh_T.shape)
            for person_idx in range(len(pred_mesh_T)):
                pred_vertices = pred_mesh_T[person_idx]
                anatomical_vertices = pred_vertices[
                    list(smpl_indices.values())
                ]
                projected_vertices = np.matmul(anatomical_vertices, intrinsic_matrix.cpu().detach().numpy().T)
                projected_vertices = projected_vertices[:, :2] / projected_vertices[:, 2:]
                projected_vertices = projected_vertices[:, :2]
                # print("projected_vertices:", projected_vertices)

                # concat with coco keypoints
                coco_kps = np.zeros((17, 2))
                projected_vertices = np.concatenate((coco_kps, projected_vertices), axis=0)
                # add confidence score to keypoints
                keypoints = np.concatenate((projected_vertices, np.ones((projected_vertices.shape[0], 1))), axis=1)
                # set coco keypoints to 0 confidence score
                keypoints[:17, 2] = 0
                person = {
                    "person_id": person_idx,
                    "keypoints": keypoints,
                    "pose_keypoints_2d": keypoints.flatten().tolist(),
                    "bbox": batch["bbox"][i][person_idx].tolist(),
                }
                # single_frame.append(person)
                results_instances.append(person)

    # convert to per frame format
    results_frame = []
    for idxs in dataset.frame_to_instance:
        results_frame.append([])
        for idx in idxs:
            result_instance = results_instances[idx]
            results_frame[-1].append(result_instance)
    ###################################

    # concat results and transform to per frame format
    # run pose tracking
    # results = run_pose_tracking(results)

    # save results
    if save_results:
        print("Saving Pose Results...")
        kpt_save_file = pkl_path
        with open(kpt_save_file, "wb") as f:
            pickle.dump(results_frame, f)

    # visualzize
    if visualize:
        print("Rendering Visualization...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_save_file = video_out_path
        videoWriter = cv2.VideoWriter(str(video_save_file), fourcc, fps, size)
        # markerPairs = getMMposeAnatomicalMarkerPairs()
        # markerNames = getMMposeAnatomicalMarkerNames()
        if marker_set == "Anatomical":
            markerPairs = getMMposeAnatomicalCocoMarkerPairs()
            markerNames = getMMposeAnatomicalCocoMarkerNames()
            for pose_results, img in tqdm(zip(results_frame, frame_iter(cap))):
                # display keypoints and bbox on frame
                for index_person in range(len(pose_results)):
                    bbox = pose_results[index_person]["bbox"]
                    kpts = pose_results[index_person]["keypoints"]
                    cv2.rectangle(
                        img,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0),
                        2,
                    )
                    for index_kpt in range(len(kpts)):
                        if index_kpt < 17:
                            color = (0, 0, 255)

                        else:
                            # if markerNames[index_kpt-17] in markerPairs.keys():
                            #     if markerNames[index_kpt-17][0] == "l":
                            if markerNames[index_kpt] in markerPairs.keys():
                                if markerNames[index_kpt][0] == "l":
                                    color = (255, 255, 0)
                                else:
                                    color = (255, 0, 255)
                            else:
                                color = (255, 0, 0)

                        cv2.circle(
                            img,
                            (int(kpts[index_kpt][0]), int(kpts[index_kpt][1])),
                            3,
                            color,
                            -1,
                        )
                videoWriter.write(img)
        elif marker_set == "Coco":
            for pose_results, img in tqdm(zip(results, frame_iter(cap))):
                # display keypoints and bbox on frame
                preds = pose_results[0]["pred_instances"]
                for index_person in range(len(preds["bboxes"])):
                    bbox = preds["bboxes"][index_person]
                    kpts = preds["keypoints"][index_person]
                    cv2.rectangle(
                        img,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0),
                        2,
                    )
                    for index_kpt in range(len(getMMposeMarkerNames())):
                        color = (0, 0, 255)
                        cv2.circle(
                            img,
                            (int(kpts[index_kpt][0]), int(kpts[index_kpt][1])),
                            3,
                            color,
                            -1,
                        )
                videoWriter.write(img)
        videoWriter.release()
