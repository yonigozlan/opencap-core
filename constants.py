import os

config_global = "sherlock"

constants = {}
if config_global == "local":
    constants = {
        "mmposeDirectory" : "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/mmpose",
        "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
        "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
        "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
        "model_ckpt_pose" : "pretrain/hrnet/best_infinity_AP_epoch_21.pth",
        "dataDir" : "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/OpenCap/data",
    }
    constants["model_ckpt_pose_absolute"] = os.path.join(constants["mmposeDirectory"], constants["model_ckpt_pose"])

if config_global == "sherlock":
    constants = {
        "mmposeDirectory" : "/home/users/yonigoz/RA/mmpose",
        "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
        "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
        "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
        "model_ckpt_pose" : "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_eval_bedlam/HRNet/w48_dark_pretrained/best_infinity_AP_epoch_18.pth",
        "dataDir" : "/scratch/users/yonigoz/OpenCap_data",
    }
    constants["model_ckpt_pose_absolute"] = constants["model_ckpt_pose"]

