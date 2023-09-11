import copy
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import utilsDataman as dm
from constants import constants
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import getDataDirectory, storage2df

plots = False
saveAndOverwriteResults = True
overwriteResults = False

scriptDir = os.getcwd()
repoDir = os.path.dirname(scriptDir)
mainDir = getDataDirectory(False)
dataDir = constants["dataDir"]
dataName = 'Data'
outputDir = os.path.join(dataDir, 'Results-paper-augmenterV2')

# %% User inputs.
subjects = ['subject4']
sessions = ['Session0', 'Session1']
poseDetectors = ['mmpose_0.8']
cameraSetups = ['2-cameras']
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
else:
    RMSEs = np.load(os.path.join(outputDir, 'RMSEs.npy'), allow_pickle=True).item()
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

print(RMSEs['all'][poseDetector][cameraSetup][augmenterType][processingType])
