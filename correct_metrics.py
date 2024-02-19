import argparse
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

# GT_PATH = "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/OpenCap/data/"
GT_PATH = "/scratch/users/yonigoz/OpenCap_data"

def mot_to_df(motPath):
    # parse the mocap motion file
    with open(motPath, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith('endheader'):
                break
        data = lines[i+1:]

    # parse data into table
    data = [x.split() for x in data]
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.set_index('time')

    return df

def build_exercises_dict(gt_path, pred_path):
    # get all mot files in the directory
    subjects_dict = {}

    subjects_dir = glob.glob(gt_path + "/subject*")
    for dir in subjects_dir:
        motFiles = glob.glob(dir + "/OpenSimData/Mocap/IK/*.mot")
        subject_key = dir.split("/")[-1]
        for motFile in motFiles:
            if subject_key not in subjects_dict:
                subjects_dict[subject_key] = {motFile.split("/")[-1]: {"gt":motFile}}
            else:
                subjects_dict[subject_key][motFile.split("/")[-1]] = {"gt":motFile}

    predicted_subjects_dir = glob.glob(pred_path + "/subject*")
    for dir in predicted_subjects_dir:
        motFiles = glob.glob(dir + "/OpenSimData/mmpose_0.8/2-cameras/Kinematics/*.mot")
        subject_key = dir.split("/")[-1].split("_")[0]
        for motFile in motFiles:
                if motFile.split("/")[-1] in subjects_dict[subject_key]:
                    subjects_dict[subject_key][motFile.split("/")[-1]]["predicted"] = motFile

    return subjects_dict

def get_best_shifts_for_metric(exercises_dict, metric):
    best_shifts = {subject: {} for subject in exercises_dict}
    df_dict = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        df_dict[subject] = {motFile: {} for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            # print(subject, motFile)
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)

                df_dict[subject][motFile]["gt"] = df_gt
                df_dict[subject][motFile]["predicted"] = df

                lowest_mse = 100000
                best_shift = 0
                for shift in range(-50, 50):
                    df_shifted = df.shift(shift)
                    df_diff = df_shifted - df_gt
                    df_diff = df_diff.dropna()
                    mse = df_diff[metric].abs().mean()
                    if mse < lowest_mse:
                        lowest_mse = mse
                        best_shift = shift
                # print(lowest_mse, best_shift)
                best_shifts[subject][motFile] = (best_shift, lowest_mse)

    return best_shifts

def compute_median_shifts(best_shifts_list):
    median_shifts = {subject: {} for subject in best_shifts_list[0]}
    for subject in best_shifts_list[0]:
        for motFile in best_shifts_list[0][subject]:
            shifts = [best_shift[subject][motFile][0] for best_shift in best_shifts_list]
            median_shifts[subject][motFile] = int(sum(shifts) / len(shifts))

    return median_shifts



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correct metrics for OpenSim')
    parser.add_argument('pred_dir', type=str, help='Path to the prediction directory')
    parser.add_argument('--output', type=str, help='Path to the output directory', default=None)

    args = parser.parse_args()
    if args.output is None:
        args.output = args.pred_dir

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)



    exercises_dict = build_exercises_dict(GT_PATH, args.pred_dir)
    best_shifts_r = get_best_shifts_for_metric(exercises_dict, "knee_angle_r")
    best_shifts_l = get_best_shifts_for_metric(exercises_dict, "knee_angle_l")
    best_shifts = [best_shifts_r, best_shifts_l]
    best_shifts = [best_shifts_r, best_shifts_l]
    median_shifts = compute_median_shifts(best_shifts)

    with open(os.path.join(args.output, 'shifts.json'), 'w') as file:
        # format json
        json.dump(median_shifts, file, indent=4)


    # compute mean rmse for each metric
    rmses = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        rmses[subject] = {motFile: None for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)
                df_shifted = df.shift(median_shifts[subject][motFile])
                df_diff_squared = (df_shifted - df_gt).pow(2)
                df_diff_squared = df_diff_squared.dropna()
                df_diff_squared = df_diff_squared[df_diff_squared.columns[:df_diff_squared.columns.get_loc('lumbar_rotation')]]
                df_diff_squared = df_diff_squared.drop(['pelvis_tx', 'pelvis_ty', 'pelvis_tz', "knee_angle_l_beta", "knee_angle_r_beta", "mtp_angle_l", "mtp_angle_r"], axis=1)
                rmses[subject][motFile] = df_diff_squared.mean().pow(0.5)


    # compute mean df for all rmses
    list_of_rmses = [rmses[subject][motFile] for subject in rmses for motFile in rmses[subject] if rmses[subject][motFile] is not None]
    # print(list_of_rmses)
    concatenated_rmses = pd.concat(list_of_rmses)
    grouped = concatenated_rmses.groupby(level=0)
    mean_rmses = grouped.mean()
    # only keep 3 significant digits
    mean_rmses = mean_rmses.round(3)
    print("mean rmses: ", mean_rmses)
    # export to csv
    mean_rmses.to_csv(os.path.join(args.output, 'mean_rmses.csv'))


    # same but without the median shift
    # compute mean rmse for each metric
    rmses = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        rmses[subject] = {motFile: None for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)
                df_diff_squared = (df - df_gt).pow(2)
                df_diff_squared = df_diff_squared.dropna()
                df_diff_squared = df_diff_squared[df_diff_squared.columns[:df_diff_squared.columns.get_loc('lumbar_rotation')]]
                df_diff_squared = df_diff_squared.drop(['pelvis_tx', 'pelvis_ty', 'pelvis_tz', "knee_angle_l_beta", "knee_angle_r_beta", "mtp_angle_l", "mtp_angle_r"], axis=1)
                rmses[subject][motFile] = df_diff_squared.mean().pow(0.5)


    # compute mean df for all rmses
    list_of_rmses = [rmses[subject][motFile] for subject in rmses for motFile in rmses[subject] if rmses[subject][motFile] is not None]
    concatenated_rmses = pd.concat(list_of_rmses)
    grouped = concatenated_rmses.groupby(level=0)
    mean_rmses = grouped.mean()
    # only keep 3 significant digits
    mean_rmses = mean_rmses.round(3)
    print("mean_rmses_no_shift: ", mean_rmses)
    # export to csv
    mean_rmses.to_csv(os.path.join(args.output, 'mean_rmses_no_shift.csv'))



