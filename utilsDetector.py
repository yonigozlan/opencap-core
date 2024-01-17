import json
import os
import pickle
import shutil
import sys
import time

import cv2
import numpy as np
from decouple import config
from utils import (getMMposeAnatomicalCocoMarkerNames,
                   getMMposeAnatomicalMarkerNames, getMMposeMarkerNames,
                   getOpenPoseMarkerNames, getVideoExtension)
from utilsChecker import getVideoRotation


# %%
def runPoseDetector(
    config_benchmark,
    CameraDirectories,
    trialRelativePath,
    pathPoseDetector,
    trialName,
    CamParamDict=None,
    resolutionPoseDetection="default",
    generateVideo=True,
    cams2Use=["all"],
    poseDetector="OpenPose",
    bbox_thr=0.8,
):
    # Create list of cameras.
    if cams2Use[0] == "all":
        cameras2Use = list(CameraDirectories.keys())
    else:
        cameras2Use = cams2Use

    CameraDirectories_selectedCams = {}
    CamParamList_selectedCams = []
    for cam in cameras2Use:
        CameraDirectories_selectedCams[cam] = CameraDirectories[cam]
        CamParamList_selectedCams.append(CamParamDict[cam])

    # Get/add video extension.
    cameraDirectory = CameraDirectories_selectedCams[cameras2Use[0]]
    pathVideoWithoutExtension = os.path.join(cameraDirectory, trialRelativePath)
    extension = getVideoExtension(pathVideoWithoutExtension)
    trialRelativePath += extension

    for camName in CameraDirectories_selectedCams:
        cameraDirectory = CameraDirectories_selectedCams[camName]
        print("Running {} for {}".format(poseDetector, camName))
        if poseDetector == "OpenPose":
            runOpenPoseVideo(
                cameraDirectory,
                trialRelativePath,
                pathPoseDetector,
                trialName,
                resolutionPoseDetection=resolutionPoseDetection,
                generateVideo=generateVideo,
            )
        elif poseDetector == "mmpose":
            runMMposeVideo(
                config_benchmark,
                cameraDirectory,
                trialRelativePath,
                pathPoseDetector,
                trialName,
                generateVideo=generateVideo,
                bbox_thr=bbox_thr,
            )

    return extension


# %%
def runOpenPoseVideo(
    cameraDirectory,
    fileName,
    pathOpenPose,
    trialName,
    resolutionPoseDetection="default",
    generateVideo=True,
):
    trialPrefix, _ = os.path.splitext(os.path.basename(fileName))
    videoFullPath = os.path.normpath(os.path.join(cameraDirectory, fileName))

    if not os.path.exists(videoFullPath):
        exception = "Video upload failed. Make sure all devices are connected to Internet and that your connection is stable."
        raise Exception(exception, exception)

    outputMediaFolder = "OutputMedia_" + resolutionPoseDetection
    outputJsonFolder = "OutputJsons_" + resolutionPoseDetection
    outputPklFolder = "OutputPkl_" + resolutionPoseDetection

    pathOutputVideo = os.path.join(cameraDirectory, outputMediaFolder, trialName)

    openposeJsonDir = os.path.join(outputJsonFolder, trialName)
    pathOutputJsons = os.path.join(cameraDirectory, openposeJsonDir)
    pathJsonDir = os.path.join(cameraDirectory, outputJsonFolder)

    openposePklDir = os.path.join(outputPklFolder, trialName)
    pathOutputPkl = os.path.join(cameraDirectory, openposePklDir)

    os.makedirs(pathOutputVideo, exist_ok=True)
    os.makedirs(pathOutputJsons, exist_ok=True)
    os.makedirs(pathOutputPkl, exist_ok=True)

    # Get number of frames.
    thisVideo = cv2.VideoCapture(videoFullPath)
    nFrameIn = int(thisVideo.get(cv2.CAP_PROP_FRAME_COUNT))

    # The video is rewritten, unrotated, and downsampled. There is no
    # need to do anything specific for the rotation, just rewriting the video
    # unrotates it.
    trialPath, _ = os.path.splitext(fileName)
    fileName = trialPath + "_rotated.avi"
    pathVideoRot = os.path.normpath(os.path.join(cameraDirectory, fileName))
    cmd_fr = " "
    # frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))
    # if frameRate > 60.0: # previously downsampled for efficiency
    #     cmd_fr = ' -r 60 '
    #     frameRate = 60.0
    CMD = "ffmpeg -loglevel error -y -i {}{}-q 0 {}".format(
        videoFullPath, cmd_fr, pathVideoRot
    )

    videoFullPath = pathVideoRot
    trialPrefix = trialPrefix + "_rotated"

    if not os.path.exists(pathVideoRot):
        os.system(CMD)

    # Run OpenPose if this file doesn't exist in outputs
    ppPklPath = os.path.join(pathOutputPkl, trialPrefix + "_pp.pkl")
    if not os.path.exists(ppPklPath):
        c_path = os.getcwd()
        command = runOpenPoseCMD(
            pathOpenPose,
            resolutionPoseDetection,
            cameraDirectory,
            fileName,
            openposeJsonDir,
            pathOutputVideo,
            trialPrefix,
            generateVideo,
            videoFullPath,
            pathOutputJsons,
        )

        if not pathOpenPose == "docker":
            os.chdir(c_path)
        # Get number of frames output video. We count the number of jsons, as
        # videos are not written on server.
        nFrameOut = len([f for f in os.listdir(pathOutputJsons) if f.endswith(".json")])
        # At high resolution, sometimes OpenPose does not process the full
        # video, let's check here and try max 5 times. If still bad, then raise
        # an exception.
        checknFrames = False
        if not resolutionPoseDetection == "default" and checknFrames:
            countFrames = 0
            while nFrameIn != nFrameOut:
                # Need to get command again, as there is os.chdir(pathOpenPose)
                # in the function.
                command = runOpenPoseCMD(
                    pathOpenPose,
                    resolutionPoseDetection,
                    cameraDirectory,
                    fileName,
                    openposeJsonDir,
                    pathOutputVideo,
                    trialPrefix,
                    generateVideo,
                    videoFullPath,
                    pathOutputJsons,
                )

                if not pathOpenPose == "docker":
                    os.chdir(c_path)
                nFrameOut = len(
                    [f for f in os.listdir(pathOutputJsons) if f.endswith(".json")]
                )
                if countFrames > 4:
                    print(
                        "# frames in {} - # frames out {}".format(nFrameIn, nFrameOut)
                    )
                    raise ValueError("OpenPose did not process the full video")
                countFrames += 1

        # Gather data from jsons in pkl file.
        saveJsonsAsPkl(pathOutputJsons, ppPklPath, trialPrefix)

        # Delete jsons
        shutil.rmtree(pathJsonDir)

    return


# %%
def runOpenPoseCMD(
    pathOpenPose,
    resolutionPoseDetection,
    cameraDirectory,
    fileName,
    openposeJsonDir,
    pathOutputVideo,
    trialPrefix,
    generateVideo,
    videoFullPath,
    pathOutputJsons,
):
    rotation = getVideoRotation(videoFullPath)
    if rotation in [0, 180]:
        horizontal = True
    else:
        horizontal = False

    command = None
    if resolutionPoseDetection == "default":
        cmd_hr = " "
    elif resolutionPoseDetection == "1x1008_4scales":
        if horizontal:
            cmd_hr = ' --net_resolution "1008x-1" --scale_number 4 --scale_gap 0.25 '
        else:
            cmd_hr = ' --net_resolution "-1x1008" --scale_number 4 --scale_gap 0.25 '
    elif resolutionPoseDetection == "1x736":
        if horizontal:
            cmd_hr = ' --net_resolution "736x-1" '
        else:
            cmd_hr = ' --net_resolution "-1x736" '
    elif resolutionPoseDetection == "1x736_2scales":
        if horizontal:
            cmd_hr = ' --net_resolution "-1x736" --scale_number 2 --scale_gap 0.75 '
        else:
            cmd_hr = ' --net_resolution "736x-1" --scale_number 2 --scale_gap 0.75 '

    if config("DOCKERCOMPOSE", cast=bool, default=False):
        vid_path_tmp = "/data/tmp-video.mov"
        vid_path = "/data/video_openpose.mov"

        # copy the video to vid_path_tmp
        shutil.copy(f"{cameraDirectory}/{fileName}", vid_path_tmp)

        # rename the video to vid_path
        os.rename(vid_path_tmp, vid_path)

        try:
            # wait until the video is processed (i.e. until the video is removed -- then json should be ready)
            start = time.time()
            while True:
                if not os.path.isfile(vid_path):
                    break

                if start + 60 * 30 < time.time():
                    raise Exception("OpenPose processing timeout")

                time.sleep(0.1)

            # copy /data/output to openposeJsonDir
            os.system(
                "cp /data/output_openpose/* {cameraDirectory}/{openposeJsonDir}/".format(
                    cameraDirectory=cameraDirectory, openposeJsonDir=openposeJsonDir
                )
            )
        except:
            exception = "Pose detection failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
            raise Exception(exception, exception)

    elif pathOpenPose == "docker":
        command = "docker run --gpus=1 -v {}:/openpose/data stanfordnmbl/openpose-gpu\
            /openpose/build/examples/openpose/openpose.bin\
            --video /openpose/data/{}\
            --display 0\
            --write_json /openpose/data/{}\
            --render_pose 0{}".format(
            cameraDirectory, fileName, openposeJsonDir, cmd_hr
        )
    else:
        os.chdir(pathOpenPose)
        pathVideoOut = os.path.join(pathOutputVideo, trialPrefix + "withKeypoints.avi")
        if not generateVideo:
            command = "bin\OpenPoseDemo.exe --video {} --write_json {} --render_threshold 0.5 --display 0 --render_pose 0{}".format(
                videoFullPath, pathOutputJsons, cmd_hr
            )
        else:
            command = "bin\OpenPoseDemo.exe --video {} --write_json {} --render_threshold 0.5 --display 0{}--write_video {}".format(
                videoFullPath, pathOutputJsons, cmd_hr, pathVideoOut
            )

    if command:
        os.system(command)

    return


# %%
def runMMposeVideo(
    config_benchmark,
    cameraDirectory,
    fileName,
    pathMMpose,
    trialName,
    generateVideo=True,
    bbox_thr=0.8,

):
    model_config_person = config_benchmark["model_config_person"]
    model_ckpt_person = config_benchmark["model_ckpt_person"]
    # model_config_person="demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    # model_ckpt_person="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    model_config_pose=config_benchmark["model_config_pose"]
    trialPrefix, _ = os.path.splitext(os.path.basename(fileName))
    videoFullPath = os.path.normpath(os.path.join(cameraDirectory, fileName))

    pathOutputVideo = os.path.join(
        cameraDirectory, "OutputMedia_mmpose_" + str(bbox_thr), trialName
    )

    # mmposeBoxDir = os.path.join("OutputBox_mmpose", trialName)
    pathOutputBox = os.path.join(cameraDirectory.replace(config_benchmark["dataName"], config_benchmark["OutputBoxDirectory"]), trialName)
    # pathOutputBox = os.path.join(cameraDirectory, mmposeBoxDir)

    mmposePklDir = os.path.join("OutputPkl_mmpose_" + str(bbox_thr), trialName)
    pathOutputPkl = os.path.join(cameraDirectory, mmposePklDir)

    os.makedirs(pathOutputVideo, exist_ok=True)
    os.makedirs(pathOutputPkl, exist_ok=True)

    # Get frame rate.
    thisVideo = cv2.VideoCapture(videoFullPath)
    # frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))

    # The video is rewritten, unrotated, and downsampled. There is no
    # need to do anything specific for the rotation, just rewriting the video
    # unrotates it.
    trialPath, _ = os.path.splitext(fileName)
    fileName = trialPath + "_rotated.avi"
    pathVideoRot = os.path.normpath(os.path.join(cameraDirectory, fileName))
    cmd_fr = " "
    # if frameRate > 60.0:
    #     cmd_fr = ' -r 60 '
    #     frameRate = 60.0
    CMD = "ffmpeg -loglevel error -y -i {}{}-q 0 {}".format(
        videoFullPath, cmd_fr, pathVideoRot
    )

    videoFullPath = pathVideoRot
    trialPrefix = trialPrefix + "_rotated"

    if not os.path.exists(pathVideoRot):
        os.system(CMD)

    pklPath = os.path.join(pathOutputPkl, trialPrefix + ".pkl")
    ppPklPath = os.path.join(pathOutputPkl, trialPrefix + "_pp.pkl")
    # Run pose detector if this file doesn't exist in outputs
    if not os.path.exists(ppPklPath):
        if config("DOCKERCOMPOSE", cast=bool, default=False):
            vid_path_tmp = "/data/tmp-video.mov"
            vid_path = "/data/video_mmpose.mov"

            # copy the video to vid_path_tmp
            shutil.copy(f"{cameraDirectory}/{fileName}", vid_path_tmp)

            # rename the video to vid_path
            os.rename(vid_path_tmp, vid_path)

            try:
                # wait until the video is processed (i.e. until the video is removed -- then json should be ready)
                start = time.time()
                while True:
                    if not os.path.isfile(vid_path):
                        break

                    if start + 60 * 30 < time.time():
                        raise Exception("mmpose processing timeout")

                    time.sleep(0.1)

                # copy /data/output to pathOutputPkl
                os.system(
                    "cp /data/output_mmpose/* {pathOutputPkl}/".format(
                        pathOutputPkl=pathOutputPkl
                    )
                )
                pkl_path_tmp = os.path.join(pathOutputPkl, "human.pkl")
                os.rename(pkl_path_tmp, pklPath)
            except:
                exception = "Pose detection failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
                raise Exception(exception, exception)

        else:
            c_path = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(os.path.join(c_path, "mmpose"))
            if config_benchmark["alt_model"] is "VirtualMarker":
                from utilsMMpose import detection_inference
                from utilsPose import pose_inference_updated
            else:
                from utilsMMpose import (detection_inference,
                                         pose_inference_updated)

            # Run human detection.
            bboxPath = os.path.join(pathOutputBox, trialPrefix + ".pkl")
            if not os.path.exists(bboxPath):
                os.makedirs(pathOutputBox, exist_ok=True)
                full_model_config_person = os.path.join(pathMMpose, model_config_person)
                detection_inference(
                    full_model_config_person, model_ckpt_person, videoFullPath, bboxPath, batch_size=config_benchmark["batch_size_det"]
                )

            # Run pose detection.
            pathModelCkptPose = config_benchmark["model_ckpt_pose_absolute"]
            videoOutPath = os.path.join(
                pathOutputVideo, trialPrefix + "withKeypoints.mp4"
            )
            full_model_config_pose = os.path.join(pathMMpose, model_config_pose)
            pose_inference_updated(
                full_model_config_pose,
                pathModelCkptPose,
                videoFullPath,
                bboxPath,
                pklPath,
                videoOutPath,
                batch_size=config_benchmark["batch_size_pose"],
                bbox_thr=bbox_thr,
                visualize=generateVideo,
                marker_set=config_benchmark["marker_set"],
            )

        # Post-process data to have OpenPose-like file structure.
        # arrangeMMposePkl(pklPath, ppPklPath)
        if config_benchmark["alt_model"] is None:
            if config_benchmark["marker_set"] == "Anatomical":
                arrangeMMposeAnatomicalPkl(pklPath, ppPklPath)
            elif config_benchmark["marker_set"] == "Coco":
                arrangeMMposePkl(pklPath, ppPklPath)
        else:
            # copy pklPath to ppPklPath
            shutil.copy(pklPath, ppPklPath)

# %%
def arrangeMMposePkl(poseInferencePklPath, outputPklPath):
    open_file = open(poseInferencePklPath, "rb")
    frames = pickle.load(open_file)
    open_file.close()

    markersMMpose = getMMposeMarkerNames()
    markersOpenPose = getOpenPoseMarkerNames()

    data4pkl = []
    for c_frame, frame in enumerate(frames):
        data4people = []
        for c, person in enumerate(frame):
            # coordinates = person["preds_with_flip"].tolist()
            coordinates = person["pred_instances"]["keypoints"][0, :, :]
            confidence = person["pred_instances"]["keypoint_scores"][0, :]
            # stack confidence with coordinates
            coordinates = np.column_stack((coordinates, confidence))
            c_coord_out = np.zeros((25 * 3,))
            for c_m, marker in enumerate(markersOpenPose):
                if marker == "midHip":
                    leftHip = coordinates[markersMMpose.index("LHip")]
                    rightHip = coordinates[markersMMpose.index("RHip")]
                    c_coord = []
                    # Mid point between both hips
                    c_coord.append((leftHip[0] + rightHip[0]) / 2)
                    c_coord.append((leftHip[1] + rightHip[1]) / 2)
                    # Lowest confidence
                    c_coord.append(np.min([leftHip[2], rightHip[2]]))
                elif marker == "Neck":
                    leftShoulder = coordinates[markersMMpose.index("LShoulder")]
                    rightShoulder = coordinates[markersMMpose.index("RShoulder")]
                    c_coord = []
                    # Mid point between both shoulders
                    c_coord.append((leftShoulder[0] + rightShoulder[0]) / 2)
                    c_coord.append((leftShoulder[1] + rightShoulder[1]) / 2)
                    # Lowest confidence
                    c_coord.append(np.min([leftShoulder[2], rightShoulder[2]]))
                else:
                    c_coord = coordinates[markersMMpose.index(marker)]
                idx_out = np.arange(c_m * 3, c_m * 3 + 3)
                c_coord_out[idx_out,] = c_coord
            c_dict = {}
            c_dict["person_id"] = [c]
            c_dict["pose_keypoints_2d"] = c_coord_out.tolist()
            data4people.append(c_dict)
        data4pkl.append(data4people)

    with open(outputPklPath, "wb") as f:
        pickle.dump(data4pkl, f)

    return


# %%
def arrangeMMposeAnatomicalPkl(poseInferencePklPath, outputPklPath):
    open_file = open(poseInferencePklPath, "rb")
    frames = pickle.load(open_file)
    open_file.close()

    markersMMposeAnatomical = getMMposeAnatomicalCocoMarkerNames()
    # markersMMposeAnatomical = getMMposeAnatomicalMarkerNames()
    nb_markers = len(markersMMposeAnatomical)
    data4pkl = []
    for c_frame, frame in enumerate(frames):
        data4people = []
        for c, person in enumerate(frame):
            # coordinates_anatomical = person["pred_instances"]["keypoints"][0, 17:, :]
            # coordinates_knees = person["pred_instances"]["keypoints"][0, 13:15, :]
            # coordinates = np.concatenate((coordinates_anatomical, coordinates_knees), axis=0)
            # c_coord_out = np.zeros((53 * 3,))
            coordinates = person["pred_instances"]["keypoints"][0, :, :]
            c_coord_out = np.zeros((nb_markers * 3,))
            for c_m, marker in enumerate(markersMMposeAnatomical):
                c_coord = [coordinates[c_m][0], coordinates[c_m][1]]
                c_coord.append(person["pred_instances"]["keypoint_scores"][0][c_m])
                idx_out = np.arange(c_m * 3, c_m * 3 + 3)
                c_coord_out[idx_out,] = c_coord
            c_dict = {}
            c_dict["person_id"] = [c]
            c_dict["pose_keypoints_2d"] = c_coord_out.tolist()
            data4people.append(c_dict)
        data4pkl.append(data4people)

    with open(outputPklPath, "wb") as f:
        pickle.dump(data4pkl, f)

    return


# %%
def saveJsonsAsPkl(json_directory, outputPklPath, videoName):
    nFrames = 0
    for file in os.listdir(json_directory):
        if videoName + "_000" in file:  # not great
            nFrames += 1

    data4pkl = []
    for frame in sorted(os.listdir(json_directory)):
        image_json = os.path.join(json_directory, frame)

        if not os.path.isfile(image_json):
            break
        with open(image_json) as data_file:
            data = json.load(data_file)

        data4people = []
        for person_idx in range(len(data["people"])):
            person = data["people"][person_idx]
            keypoints = person["pose_keypoints_2d"]

            c_dict = {}
            c_dict["person_id"] = [person_idx]
            c_dict["pose_keypoints_2d"] = keypoints
            data4people.append(c_dict)
        data4pkl.append(data4people)

    with open(outputPklPath, "wb") as f:
        pickle.dump(data4pkl, f)

    return
