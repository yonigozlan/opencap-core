import argparse
import copy
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import utilsDataman as dm
import wandb
from ReproducePaperResults.labValidationVideosToKinematicsAnatomical import \
    process_trials
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import getDataDirectory, storage2df

config_global = "sherlock" # "local" or "sherlock

config_base_local = {
    "mmposeDirectory" : "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/mmpose",
    "model_config_person" : "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    "model_ckpt_person" : "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    # "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
    # "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
    "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
    "model_ckpt_pose" : "pretrain/hrnet/best_infinity_AP_epoch_21.pth",
    "dataDir" : "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/OpenCap/data",
    "batch_size_det": 4,
    "batch_size_pose": 8
}
config_base_local["model_ckpt_pose_absolute"] = os.path.join(config_base_local["mmposeDirectory"], config_base_local["model_ckpt_pose"])


config_base_sherlock = {
    "mmposeDirectory" : "/home/users/yonigoz/RA/mmpose",
    "OutputBoxDirectory" : "/scratch/users/yonigoz/OpenCap_data/OutputBox",
    "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
    "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
    # "model_config_person" : "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    # "model_ckpt_person" : "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_ViTPose-huge_8xb64-210e_merge_bedlam_infinity_eval_bedlam-256x192.py",
    "model_ckpt_pose" : "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_eval_bedlam/ViT/huge/best_infinity_AP_epoch_10.pth",
    # "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
    # "model_ckpt_pose" : "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_eval_bedlam/HRNet/w48_dark_pretrained/best_infinity_AP_epoch_18.pth",
    "dataDir" : "/scratch/users/yonigoz/OpenCap_data",
    "batch_size_det": 32,
    "batch_size_pose": 4
}
config_base_sherlock["model_ckpt_pose_absolute"] = config_base_sherlock["model_ckpt_pose"]

config = {}
if config_global == "local":
    config = config_base_local
if config_global == "sherlock":
    config = config_base_sherlock

parser = argparse.ArgumentParser(description='Benchmark OpenCap')
parser.add_argument('--dataDir', type=str, default=config["dataDir"],
                    help='Directory where data is stored')
parser.add_argument('--model_config_person', type=str, default=config["model_config_person"],
                    help='Model config file for person detector')
parser.add_argument('--model_ckpt_person', type=str, default=config["model_ckpt_person"],
                    help='Model checkpoint file for person detector')
parser.add_argument('--model_config_pose', type=str, default=config["model_config_pose"],
                    help='Model config file for pose detector')
parser.add_argument('--model_ckpt_pose', type=str, default=config["model_ckpt_pose"],
                    help='Model checkpoint file for pose detector')
parser.add_argument('--batch_size_det', type=int, default=config["batch_size_det"],
                    help='Batch size for person detector')
parser.add_argument('--batch_size_pose', type=int, default=config["batch_size_pose"],
                    help='Batch size for pose detector')
parser.add_argument('--dataName', type=str, default="Data",
                    help='Name of data directory where predictions will be stored')
parser.add_argument('--subjects', type=str, default="all",
                    help='Subjects to process')
parser.add_argument('--sessions', type=str, default="all",
                    help='Sessions to process')
parser.add_argument('--cameraSetups', type=str, default="2-cameras",
                    help='Camera setups to process')
parser.add_argument('--process_trials', type=bool, default=True,
                    help='Process trials')


args = parser.parse_args()

# replace config with args
config["dataDir"] = args.dataDir
config["model_config_person"] = args.model_config_person
config["model_ckpt_person"] = args.model_ckpt_person
config["model_config_pose"] = args.model_config_pose
config["model_ckpt_pose"] = args.model_ckpt_pose
config["batch_size_det"] = args.batch_size_det
config["batch_size_pose"] = args.batch_size_pose
config["dataName"] = args.dataName
config["subjects"] = args.subjects
config["sessions"] = args.sessions
config["cameraSetups"] = args.cameraSetups
config["process_trials"] = args.process_trials


if config["process_trials"]:
    process_trials(config)


plots = False
saveAndOverwriteResults = True
overwriteResults = False

scriptDir = os.getcwd()
repoDir = os.path.dirname(scriptDir)
dataDir = config["dataDir"]
dataName = config["dataName"]
outputDir = os.path.join(dataDir, dataName, 'joints-metrics')
#create output directory if it doesn't exist
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


if config["subjects"] == "all":
    subjects = ["subject" + str(i) for i in range(2, 12)]
else:
    subjects = [config["subjects"]]
if config["sessions"] == "all":
    sessions = ['Session0', 'Session1']
else:
    sessions = [config["sessions"]]

poseDetectors = ['mmpose_0.8']
cameraSetups = [config["cameraSetups"]]
augmenterTypes = {
    "upsampling"
}
# setups_t = list(augmenterTypes.keys())
setups_t = ['Uhlrich et al. 2022', 'Latest']

# processingTypes = ['IK_IK', 'addB_addB', 'IK_addB', 'addB_IK']
processingTypes = ['IK_IK']
genericModel4ScalingName = 'LaiArnoldModified2017_poly_withArms_weldHand.osim'
coordinates = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_l', 'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r',
    'subtalar_angle_l', 'subtalar_angle_r',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
nCoordinates = len(coordinates)
# Bilateral coordinates
coordinates_bil = [
    'hip_flexion', 'hip_adduction', 'hip_rotation',
    'knee_angle', 'ankle_angle', 'subtalar_angle']
# coordinates without the right side, such that bilateral coordinates are
# combined later.
coordinates_lr = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
coordinates_lr_tr = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
coordinates_lr_rot = coordinates_lr.copy()
# Translational coordinates.
coordinates_tr = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
for coordinate in coordinates_lr_tr:
    coordinates_lr_rot.remove(coordinate)
motions = ['walking', 'DJ', 'squats', 'STS']

addBiomechanicsMocapModel = 'LaiArnold2107_OpenCapMocap'
addBiomechanicsVideoModel = 'LaiArnold2107_OpenCapVideo'

fixed_markers = False # False should be default (better results)


if not os.path.exists(os.path.join(outputDir, 'RMSEs.npy')):
    RMSEs = {}
    os.makedirs(outputDir)
else:
    RMSEs = np.load(os.path.join(outputDir, 'RMSEs.npy'), allow_pickle=True).item()
    RMSEs = {}
if not os.path.exists(os.path.join(outputDir, 'MAEs.npy')):
    MAEs = {}
else:
    MAEs = np.load(os.path.join(outputDir, 'MAEs.npy'), allow_pickle=True).item()
if not os.path.exists(os.path.join(outputDir, 'MEs.npy')):
    MEs = {}
else:
    MEs = np.load(os.path.join(outputDir, 'MEs.npy'), allow_pickle=True).item()

if not 'all' in RMSEs:
    RMSEs['all'] = {}
if not 'all' in MAEs:
    MAEs['all'] = {}
if not 'all' in MEs:
    MEs['all'] = {}

for motion in motions:
    if not motion in RMSEs:
        RMSEs[motion] = {}
    if not motion in MAEs:
        MAEs[motion] = {}
    if not motion in MEs:
        MEs[motion] = {}

for subjectName in subjects:
    for sessionName in sessions:
        if not subjectName in RMSEs:
            RMSEs[subjectName] = {}
        if not subjectName in MAEs:
            MAEs[subjectName] = {}
        if not subjectName in MEs:
            MEs[subjectName] = {}
        print('\nProcessing {}'.format(subjectName))
        osDir = os.path.join(dataDir, dataName, f"{subjectName}_{sessionName}", 'OpenSimData')
        markerDir = os.path.join(dataDir, dataName, f"{subjectName}_{sessionName}", 'MarkerData')

        # Hack
        mocapDirAll = os.path.join(dataDir, subjectName, 'OpenSimData', 'Mocap', 'IK')
        trials = []
        for trial in os.listdir(mocapDirAll):
            if not trial[-3:] == 'mot':
                continue
            trials.append(trial[:-4])
        count = 0
        for trial in os.listdir(mocapDirAll):
            if not trial[-3:] == 'mot':
                continue
            for poseDetector in poseDetectors:
                if not poseDetector in RMSEs[subjectName]:
                    RMSEs[subjectName][poseDetector] = {}
                if not poseDetector in RMSEs['all']:
                    RMSEs['all'][poseDetector] = {}
                for motion in motions:
                    if not poseDetector in RMSEs[motion]:
                        RMSEs[motion][poseDetector] = {}

                if not poseDetector in MAEs[subjectName]:
                    MAEs[subjectName][poseDetector] = {}
                if not poseDetector in MAEs['all']:
                    MAEs['all'][poseDetector] = {}
                for motion in motions:
                    if not poseDetector in MAEs[motion]:
                        MAEs[motion][poseDetector] = {}

                if not poseDetector in MEs[subjectName]:
                    MEs[subjectName][poseDetector] = {}
                if not poseDetector in MEs['all']:
                    MEs['all'][poseDetector] = {}
                for motion in motions:
                    if not poseDetector in MEs[motion]:
                        MEs[motion][poseDetector] = {}

                poseDetectorDir = os.path.join(osDir, poseDetector)
                poseDetectorMarkerDir = os.path.join(markerDir, poseDetector)
                for cameraSetup in cameraSetups:
                    if not cameraSetup in RMSEs[subjectName][poseDetector]:
                        RMSEs[subjectName][poseDetector][cameraSetup] = {}
                    if not cameraSetup in RMSEs['all'][poseDetector]:
                        RMSEs['all'][poseDetector][cameraSetup] = {}
                    for motion in motions:
                        if not cameraSetup in RMSEs[motion][poseDetector]:
                            RMSEs[motion][poseDetector][cameraSetup] = {}

                    if not cameraSetup in MAEs[subjectName][poseDetector]:
                        MAEs[subjectName][poseDetector][cameraSetup] = {}
                    if not cameraSetup in MAEs['all'][poseDetector]:
                        MAEs['all'][poseDetector][cameraSetup] = {}
                    for motion in motions:
                        if not cameraSetup in MAEs[motion][poseDetector]:
                            MAEs[motion][poseDetector][cameraSetup] = {}

                    if not cameraSetup in MEs[subjectName][poseDetector]:
                        MEs[subjectName][poseDetector][cameraSetup] = {}
                    if not cameraSetup in MEs['all'][poseDetector]:
                        MEs['all'][poseDetector][cameraSetup] = {}
                    for motion in motions:
                        if not cameraSetup in MEs[motion][poseDetector]:
                            MEs[motion][poseDetector][cameraSetup] = {}

                    for augmenterType in augmenterTypes:

                        if not augmenterType in RMSEs[subjectName][poseDetector][cameraSetup]:
                            RMSEs[subjectName][poseDetector][cameraSetup][augmenterType] = {}
                        if not augmenterType in RMSEs['all'][poseDetector][cameraSetup]:
                            RMSEs['all'][poseDetector][cameraSetup][augmenterType] = {}
                        for motion in motions:
                            if not augmenterType in RMSEs[motion][poseDetector][cameraSetup]:
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType] = {}

                        if not augmenterType in MAEs[subjectName][poseDetector][cameraSetup]:
                            MAEs[subjectName][poseDetector][cameraSetup][augmenterType] = {}
                        if not augmenterType in MAEs['all'][poseDetector][cameraSetup]:
                            MAEs['all'][poseDetector][cameraSetup][augmenterType] = {}
                        for motion in motions:
                            if not augmenterType in MAEs[motion][poseDetector][cameraSetup]:
                                MAEs[motion][poseDetector][cameraSetup][augmenterType] = {}

                        if not augmenterType in MEs[subjectName][poseDetector][cameraSetup]:
                            MEs[subjectName][poseDetector][cameraSetup][augmenterType] = {}
                        if not augmenterType in MEs['all'][poseDetector][cameraSetup]:
                            MEs['all'][poseDetector][cameraSetup][augmenterType] = {}
                        for motion in motions:
                            if not augmenterType in MEs[motion][poseDetector][cameraSetup]:
                                MEs[motion][poseDetector][cameraSetup][augmenterType] = {}

                        for processingType in processingTypes:

                            if processingType == 'IK_IK' or processingType == 'IK_addB':
                                addBiomechanicsMocap = False
                            else:
                                addBiomechanicsMocap = True


                            if processingType == 'IK_IK' or processingType == 'addB_IK':
                                addBiomechanicsVideo = False
                            else:
                                addBiomechanicsVideo = True

                            # if augmenterType in RMSEs[subjectName][poseDetector][cameraSetup] and augmenterType in MEs[subjectName][poseDetector][cameraSetup] and not overwriteResults:
                            #     continue

                            if not processingType in RMSEs[subjectName][poseDetector][cameraSetup][augmenterType]:
                                RMSEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates, index=trials)
                            if not processingType in RMSEs['all'][poseDetector][cameraSetup][augmenterType]:
                                RMSEs['all'][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                            for motion in motions:
                                if not processingType in RMSEs[motion][poseDetector][cameraSetup][augmenterType]:
                                    RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)

                            if not processingType in MAEs[subjectName][poseDetector][cameraSetup][augmenterType]:
                                MAEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates, index=trials)
                            if not processingType in MAEs['all'][poseDetector][cameraSetup][augmenterType]:
                                MAEs['all'][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                            for motion in motions:
                                if not processingType in MAEs[motion][poseDetector][cameraSetup][augmenterType]:
                                    MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)

                            if not processingType in MEs[subjectName][poseDetector][cameraSetup][augmenterType]:
                                MEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates, index=trials)
                            if not processingType in MEs['all'][poseDetector][cameraSetup][augmenterType]:
                                MEs['all'][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)
                            for motion in motions:
                                if not processingType in MEs[motion][poseDetector][cameraSetup][augmenterType]:
                                    MEs[motion][poseDetector][cameraSetup][augmenterType][processingType] = pd.DataFrame(columns=coordinates)


                            if addBiomechanicsMocap:
                                mocapDir = os.path.join(osDir, 'Mocap', 'AddBiomechanics', 'IK', addBiomechanicsMocapModel)
                            else:
                                mocapDir = os.path.join(dataDir, subjectName, 'OpenSimData', 'Mocap', 'IK')

                            pathTrial = os.path.join(mocapDir, trial)
                            trial_mocap_df = storage2df(pathTrial, coordinates)

                            if addBiomechanicsVideo:
                                cameraSetupDir = os.path.join(
                                    poseDetectorDir, cameraSetup, augmenterType, 'AddBiomechanics', 'IK',
                                    addBiomechanicsVideoModel)
                            else:
                                cameraSetupDir = os.path.join(
                                    poseDetectorDir, cameraSetup, 'Kinematics')

                            # Used to use same augmenter for all for offset, not sure why
                            cameraSetupMarkerDir = os.path.join(
                                poseDetectorMarkerDir, cameraSetup, "PreAugmentation")

                            trial_video = trial[:-4] + '.mot'
                            trial_marker = trial[:-4] + '.trc'
                            pathTrial_video = os.path.join(cameraSetupDir, trial_video)
                            pathTrial_marker = os.path.join(cameraSetupMarkerDir, trial_marker)

                            if not os.path.exists(pathTrial_video):
                                continue

                            trial_video_df = storage2df(pathTrial_video, coordinates)

                            # Convert to numpy
                            trial_mocap_np = trial_mocap_df.to_numpy()
                            trial_video_np = trial_video_df.to_numpy()

                            # Extract start and end time from video
                            time_start_video = trial_video_np[0, 0]
                            time_end_video = trial_video_np[-1, 0]

                            # Extract start and end time from moca[]
                            time_start_mocap = trial_mocap_np[0, 0]
                            time_end_mocap = trial_mocap_np[-1, 0]

                            # Pick the time vector that is the most restrictive
                            time_start = max(time_start_video, time_start_mocap)
                            time_end = min(time_end_video, time_end_mocap)

                            # For DJ trials, we trimmed the OpenSim IK ones, but not the addb ones
                            # Let's load a reference OpenSim IK one to get the same time vector
                            if addBiomechanicsMocap and addBiomechanicsVideo and 'DJ' in trial:
                                mocapDir_temp = os.path.join(osDir, 'Mocap', 'IK', genericModel4ScalingName[:-5])
                                pathTrial_temp = os.path.join(mocapDir_temp, trial)
                                trial_mocap_df_temp = storage2df(pathTrial_temp, coordinates)
                                trial_mocap_np_temp = trial_mocap_df_temp.to_numpy()
                                time_start_mocap_temp = trial_mocap_np_temp[0, 0]
                                time_end_mocap_temp = trial_mocap_np_temp[-1, 0]
                                time_start = max(time_start, time_start_mocap_temp)
                                time_end = min(time_end, time_end_mocap_temp)

                            # Find corresponding indices in mocap data
                            min_len = min(trial_mocap_np.shape[0], trial_video_np.shape[0])
                            idx_start_mocap = np.argwhere(trial_mocap_np[:, 0] == time_start)[0][0]
                            idx_end_mocap = np.argwhere(trial_mocap_np[:, 0] == time_end)[0][0]
                            # Select mocap data based on video-based time vector
                            trial_mocap_np_adj = trial_mocap_np[idx_start_mocap:idx_end_mocap+1, :]

                            # Find corresponding indices in video data
                            idx_start_video = np.argwhere(trial_video_np[:, 0] == time_start)[0][0]
                            idx_end_video = np.argwhere(trial_video_np[:, 0] == time_end)[0][0]
                            # Select video data based on video-based time vector
                            trial_video_np_adj = trial_video_np[idx_start_video:idx_end_video+1, :]

                            # Compute RMSEs and MEs
                            y_true = trial_mocap_np_adj[:, 1:]
                            y_pred = trial_video_np_adj[:, 1:]

                            c_rmse = []
                            c_mae = []
                            c_me = []

                            # If translational degree of freedom, adjust for offset
                            # Compute offset from trc file.
                            # c_trc = dm.TRCFile(pathTrial_marker)
                            # c_trc_m1 = c_trc.marker('Neck')
                            # c_trc_m1_offsetRemoved = c_trc.marker('Neck_offsetRemoved')
                            # # Verify same offset for different markers.
                            # c_trc_m2 = c_trc.marker('L_knee_study')
                            # c_trc_m2_offsetRemoved = c_trc.marker('L_knee_study_offsetRemoved')
                            # c_trc_m1_offset = np.mean(c_trc_m1-c_trc_m1_offsetRemoved, axis=0)
                            # c_trc_m2_offset = np.mean(c_trc_m2-c_trc_m2_offsetRemoved, axis=0)
                            # assert (np.all(np.round(c_trc_m1_offset,2)==np.round(c_trc_m2_offset,2))), 'Problem offset'

                            for count1 in range(y_true.shape[1]):
                                c_coord = coordinates[count1]
                                # If translational degree of freedom, adjust for offset
                                # if c_coord in coordinates_tr:
                                #     # continue
                                #     if c_coord == 'pelvis_tx':
                                #         y_pred_adj = y_pred[:, count1] - c_trc_m1_offset[0]/1000
                                #     elif c_coord == 'pelvis_ty':
                                #         y_pred_adj = y_pred[:, count1] - c_trc_m1_offset[1]/1000
                                #     elif c_coord == 'pelvis_tz':
                                #         y_pred_adj = y_pred[:, count1] - c_trc_m1_offset[2]/1000
                                #     y_true_adj = y_true[:, count1]
                                # else:
                                # Convert to degrees if addBiomechanics
                                if addBiomechanicsMocap:
                                    y_true_adj = np.rad2deg(y_true[:, count1])
                                else:
                                    y_true_adj = y_true[:, count1]
                                if addBiomechanicsVideo:
                                    y_pred_adj = np.rad2deg(y_pred[:, count1])
                                else:
                                    y_pred_adj = y_pred[:, count1]

                                # RMSE
                                value = mean_squared_error(
                                    y_true_adj, y_pred_adj, squared=False)
                                RMSEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType].loc[trials[count], c_coord] = value
                                c_rmse.append(value)
                                # MAE
                                value2 = mean_absolute_error(
                                    y_true_adj, y_pred_adj)
                                MAEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType].loc[trials[count], c_coord] = value2
                                c_mae.append(value2)
                                # ME
                                # COmpute mean error between y_true_adj and y_pred_adj
                                value3 = np.mean(y_true_adj - y_pred_adj)
                                MEs[subjectName][poseDetector][cameraSetup][augmenterType][processingType].loc[trials[count], c_coord] = value3
                                c_me.append(value3)

                                if value > 20:
                                    print('{} - {} - {} - {} - {} - {} - {} had RMSE of {}'.format(subjectName, poseDetector, cameraSetup, augmenterType, processingType, trial[:-4], c_coord, value))

                            c_name = subjectName + '_' +  trial[:-4]
                            RMSEs['all'][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_rmse
                            MAEs['all'][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_mae
                            MEs['all'][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_me

                            for motion in motions:
                                if motion in trial:
                                    RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_rmse
                                    MAEs[motion][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_mae
                                    MEs[motion][poseDetector][cameraSetup][augmenterType][processingType].loc[c_name] = c_me

            count += 1
if saveAndOverwriteResults:
    np.save(os.path.join(outputDir, 'RMSEs.npy'), RMSEs)
    np.save(os.path.join(outputDir, 'MAEs.npy'), MAEs)
    np.save(os.path.join(outputDir, 'MEs.npy'), MEs)

all_motions = ['all'] + motions
bps, means_RMSEs, medians_RMSEs, stds_RMSEs = {}, {}, {}, {}
for motion in all_motions:
    bps[motion], means_RMSEs[motion], medians_RMSEs[motion], stds_RMSEs[motion] = {}, {}, {}, {}
    if plots:
        fig, axs = plt.subplots(5, 3, sharex=True)
        fig.suptitle(motion)
    for count, coordinate in enumerate(coordinates_lr):
        c_data = {}
        for processingType in processingTypes:
            for augmenterType in augmenterTypes:
                for poseDetector in poseDetectors:
                    for cameraSetup in cameraSetups:
                        if coordinate[-2:] == '_l':
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType] = (
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist() +
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate[:-2] + '_r'].tolist())
                            coordinate_title = coordinate[:-2]
                        else:
                            c_data[poseDetector + '_' + cameraSetup + '_' + augmenterType+ '_' + processingType] = (
                                RMSEs[motion][poseDetector][cameraSetup][augmenterType][processingType][coordinate].tolist())
                            coordinate_title = coordinate
        if plots:
            ax = axs.flat[count]
            bps[motion][coordinate] = ax.boxplot(c_data.values())
            ax.set_title(coordinate_title)
            xticks = list(range(1, len(cameraSetups)*len(poseDetectors)*len(augmenterTypes)*len(processingTypes)+1))
            ax.set_xticks(xticks)
            ax.set_xticklabels(c_data.keys(), rotation = 90)
            ax.set_ylim(0, 20)
            ax.axhline(y=5, color='r', linestyle='--')
            ax.axhline(y=10, color='b', linestyle='--')

        means_RMSEs[motion][coordinate] = [np.mean(c_data[a]) for a in c_data]
        medians_RMSEs[motion][coordinate] = [np.median(c_data[a]) for a in c_data]
        stds_RMSEs[motion][coordinate] = [np.std(c_data[a]) for a in c_data]

setups = []
for processingType in processingTypes:
    for augmenterType in augmenterTypes:
        for poseDetector in poseDetectors:
            for cameraSetup in cameraSetups:
                setups.append(poseDetector + '_' + cameraSetup + '_' + augmenterType + '_' + processingType)

suffixRMSE = ''
if fixed_markers:
    suffixRMSE = '_fixed'

with open(os.path.join(outputDir,'RMSEs{}_means.csv'.format(suffixRMSE)), 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    topRow = ['motion-type', '', 'setup']
    for label in coordinates_lr:

        if label[-2:] == '_l':
            label_adj = label[:-2]
        else:
            label_adj = label

        topRow.extend([label_adj,''])
    _ = csvWriter.writerow(topRow)
    secondRow = ['', '', '']
    secondRow.extend(["mean-RMSE",''] * len(coordinates_lr))
    secondRow.extend(["min-rot","max-rot","mean-rot","std-rot","","min-tr","max-tr","mean-tr","std-tr"])
    _ = csvWriter.writerow(secondRow)
    means_RMSE_summary, mins_RMSE_summary, maxs_RMSE_summary = {}, {}, {}
    for idxMotion, motion in enumerate(all_motions):
        means_RMSE_summary[motion], mins_RMSE_summary[motion], maxs_RMSE_summary[motion] = {}, {}, {}
        c_bp = means_RMSEs[motion]
        for idxSetup, setup in enumerate(setups):
            means_RMSE_summary[motion][setup] = {}
            mins_RMSE_summary[motion][setup] = {}
            maxs_RMSE_summary[motion][setup] = {}
            if idxSetup == 0:
                RMSErow = [motion, '', setup]
            else:
                RMSErow = ['', '', setup]
            temp_med_rot = np.zeros(len(coordinates_lr_rot) + len(coordinates_bil),)
            temp_med_tr = np.zeros(len(coordinates_lr_tr),)
            c_rot = 0
            c_tr = 0
            for coordinate in coordinates_lr:
                c_coord = c_bp[coordinate]
                RMSErow.extend(['%.2f' %c_coord[idxSetup], ''])
                if coordinate in coordinates_lr_rot:
                    # We want to include twice the bilateral coordinates and
                    # once the unilateral coordinates such as to make sure the
                    # unilateral coordinates are not overweighted when
                    # computing means.
                    if coordinate[:-2] in coordinates_bil:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        temp_med_rot[c_rot+1,] = c_coord[idxSetup]
                        c_rot += 2
                    else:
                        temp_med_rot[c_rot,] = c_coord[idxSetup]
                        c_rot += 1
                elif coordinate in coordinates_lr_tr:
                    temp_med_tr[c_tr,] = c_coord[idxSetup]
                    c_tr += 1
            # Add min, max, mean
            RMSErow.extend(['%.2f' %np.round(np.min(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.max(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.mean(temp_med_rot),1)])
            RMSErow.extend(['%.2f' %np.round(np.std(temp_med_rot),1), ''])
            RMSErow.extend(['%.2f' %np.round(np.min(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.max(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.mean(temp_med_tr*1000),1)])
            RMSErow.extend(['%.2f' %np.round(np.std(temp_med_tr*1000),1)])
            _ = csvWriter.writerow(RMSErow)
            means_RMSE_summary[motion][setup]['rotation'] = np.round(np.mean(temp_med_rot),1)
            means_RMSE_summary[motion][setup]['translation'] = np.round(np.mean(temp_med_tr*1000),1)

            mins_RMSE_summary[motion][setup]['rotation'] = np.round(np.min(temp_med_rot),1)
            mins_RMSE_summary[motion][setup]['translation'] = np.round(np.min(temp_med_tr*1000),1)

            maxs_RMSE_summary[motion][setup]['rotation'] = np.round(np.max(temp_med_rot),1)
            maxs_RMSE_summary[motion][setup]['translation'] = np.round(np.max(temp_med_tr*1000),1)


# Add wandb logging
mean_RMSEs_df = pd.DataFrame(means_RMSEs).transpose().applymap(lambda x: x[0]).reset_index()
print(mean_RMSEs_df)
wandb.init(project="opencap_bench",
            entity="yonigoz",
            name="subject4_local",)
mean_RMSEs_table = wandb.Table(dataframe=mean_RMSEs_df)
wandb.log({"Mean RMSEs": mean_RMSEs_table})
# wandb.log(means_RMSEs)
