
# from spatial import *

from shapely.geometry.polygon import Polygon
import numpy as np
from scipy.spatial import distance as dist
import os
import cv2
import sys
from pathlib import Path
import csv
import time
from joblib import load
from keras.models import load_model
from feature_extraction import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_sparse_opticalflow(prev_gray, frame, tracks, track_len, lk_params, feature_params):

    vis = frame.copy()
    mm1 = 0

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)

    vel=0

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray

        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)


        tracks = new_tracks

        ptn1 = 0
        for tr in tracks:
            ptn1 += 1
            mm1 += dist.euclidean(tr[0], tr[1])
            vel = mm1/ptn1

        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

    mask = np.zeros_like(frame_gray)
    mask[:] = 255
    for x, y in [np.int32(tr[-1]) for tr in tracks]:
        cv2.circle(mask, (x, y), 3, 0, -1)
    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

    if p is not None:
        for x, y in np.float32(p).reshape(-1, 2):
            tracks.append([(x, y)])
    return vis, tracks, vel


def handle_left_click (event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (255,255,255), -1)
    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 255, 255), thickness=2)
    return frame

def focus_on_region(source):
    points=[]
    cap = cv2.VideoCapture(source)
    while True:
        ret, first_frame = cap.read()

        first_frame = draw_polygon(first_frame, points)

        if  cv2.waitKey(0) & 0xFF == ord('d'): 
            points.append(points[0])

            cal_mask = np.zeros_like(first_frame[:, :, 0])
            view_polygon = np.array(points)

            cv2.fillConvexPoly(cal_mask, view_polygon, 1)

            cv2.destroyAllWindows()
            return points, cal_mask

        cv2.imshow("Frame", first_frame)
        cv2.setMouseCallback("Frame", handle_left_click, points)

def save_csv(csv_file, header, data):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    f.close()


def cal_ROI(poly):
    polygon = Polygon(poly)
    return polygon.area


def run(
        source=ROOT / 'data/images',
        clf_weights = ROOT / 'svm/lgbm.sav',
        transform = ROOT / 'transform/lgbm_scaler.sav',
        lk_params = dict( winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        feature_params = dict(maxCorners = 200,
            qualityLevel = 0.6,
            minDistance = 5,
            blockSize = 4),
):
    points, view_mask = focus_on_region(source)
    density_label = ["Level 1", "Level 2", "Level 3", "Level 4", 'Level 5']
    areaROI = cal_ROI(points)
    get_prv_frame = False
    video = cv2.VideoCapture(source)
    density = "Loading..."

    clf = load_model(clf_weights)
    sc = load(transform)
################################################
################################################

    start = time.time()
    interval = 0


    while True:
        
        ret, img = video.read()
        if not ret:
            break
        display = img.copy()

        if not get_prv_frame:
            frame_rate = '___/s'
            frame_id = 1
            frame_time = time.time()
            prev_frame = to_hsv(img)
            get_prv_frame = True
            init_flag = True
            data = []
            track_opf = []
            continue


        if int(time.time() - frame_time) == 1:
            frame_time = time.time()
            frame_rate = f'{frame_id}/s'
            frame_id = 1


        img = cv2.bitwise_and(img, img, mask=view_mask)
        
        img = to_hsv(img)
        hist, entr = cal_hist(img)
        dissimilarity, correlation, homogeneity, energy, contrast, ASM = GLCM(img)
        opf, track_opf, vel = get_sparse_opticalflow(prev_frame, img, track_opf, 4, lk_params, feature_params)
        vel = round(vel, 3)
        # old_track_opf, old_vel = track_opf.copy(), vel
        po = np.array([p[-1] for p in track_opf])
        dispersion_x = np.std(po[:,0])
        dispersion_y = np.std(po[:,1])
        data.extend([vel, len(track_opf), hist[0], dispersion_x, dispersion_y, entr, areaROI, dissimilarity[0][0], correlation[0][0], homogeneity[0][0], energy[0][0], contrast[0][0], ASM[0][0], len(track_opf)/areaROI])
        cv2.putText(opf, f'Points: {len(track_opf)}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        prev_frame = img.copy()

        # print(f"TIME: {int(end - start + 1) % 15 == 0} - Flag: {init_flag} - Data shape: {np.array(data).shape}")
        if init_flag:
            if int(time.time() - start + 1) % 8 == 0:
                s = time.time()
                init_flag = False

        else:
            if len(data) >= 75*14:
                data = np.array([data[-75*14:]])
                # print(f'Data shape: {data.shape}')
                # exit()
                data = sc.transform(data)
                data = data.reshape(1, 75, 14)
                # [vel, len(track_opf), hist[0], dispersion_x, dispersion_y, entr, areaROI, dissimilarity[0][0], correlation[0][0], homogeneity[0][0], energy[0][0], contrast[0][0], ASM[0][0], len(track_opf)/areaROI]
                pred = clf.predict(data)
                pred = np.argmax(pred).tolist()
                density = density_label[pred]
                data = []
                interval = str(int(time.time() - s))
                s = time.time()


                

########################################
        # if density != "Loading..." and not velocity:
        #     if pred in velocity_level:
        #         velocity_level[pred].append(vel)
        #     else:
        #         velocity_level[pred] = [vel]
#########################################

        cv2.putText(opf, f'Density: {str(density)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(opf, f'Init flag: {str(init_flag)}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(opf, f'Length of data: {str(len(data))}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(opf, f'Frame rate: {frame_rate}', (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 80), 1)
        
        
        cv2.imshow("Monitor", opf)
        frame_id += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()



import glob
def get_last_file_name(dir_):
    try:
        dir_ = dir_.replace("/","/")
        list_of_files = glob.glob(dir_+"/*") # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        return "".join((char if char.isalnum() else " ") for char in latest_file).split()[-2]
    except:
        return '0'
    
def data_prepare(
        source=ROOT / 'data/images',
        lk_params = dict( winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        feature_params = dict(maxCorners = 200,
            qualityLevel = 0.6,
            minDistance = 5,
            blockSize = 4),
        save_f=None
):
    points, view_mask = focus_on_region(source)
    areaROI = cal_ROI(points)
    get_prv_frame = False
    track_opf = []
    video = cv2.VideoCapture(source)

    frame_num = 0
    init = True
    while True:
        ret, img = video.read()
        if not ret:
            break
        
        s = time.time()
        while init:
            if not get_prv_frame:
                prev_frame = img.copy()
                get_prv_frame = True
                continue

            img = cv2.bitwise_and(img, img, mask=view_mask)
            img = to_hsv(img)
            hist, entr = cal_hist(img)

            dissimilarity, correlation, homogeneity, energy, contrast, ASM = GLCM(img)
            opf, track_opf, vel = get_sparse_opticalflow(prev_frame, img, track_opf, 4, lk_params, feature_params)
            
            prev_frame = img.copy()
            cv2.imshow('Init frames', opf)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

            if time.time() - s > 8:
                init = False
                cv2.destroyAllWindows()

        img = cv2.bitwise_and(img, img, mask=view_mask)
        img = to_hsv(img)
        hist, entr = cal_hist(img)

        dissimilarity, correlation, homogeneity, energy, contrast, ASM = GLCM(img)
        opf, track_opf, vel = get_sparse_opticalflow(prev_frame, img, track_opf, 4, lk_params, feature_params)
        vel = round(vel, 3)
        # old_track_opf, old_vel = track_opf.copy(), vel
        po = np.array([p[-1] for p in track_opf])
        dispersion_x = np.std(po[:,0])
        dispersion_y = np.std(po[:,1])

        keypoints = [vel, len(track_opf), hist[0], dispersion_x, dispersion_y, entr, areaROI, dissimilarity[0][0], correlation[0][0], homogeneity[0][0], energy[0][0], contrast[0][0], ASM[0][0], len(track_opf)/areaROI]
        npy_path = os.path.join(save_f, str(frame_num))
        np.save(npy_path, keypoints)

        frame_num += 1
        prev_frame = img.copy()

        cv2.imshow('Actual frames', opf)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    video.release()
    cv2.destroyAllWindows()


import shutil
def extract_features(folder):
    try:
        for label_folder in os.listdir(folder): # f1, f2 ... f5
            s = os.path.join(folder, f'Feature_{label_folder}')
            for i, vid in enumerate(os.listdir(os.path.join(folder, label_folder))): # f1/mp4_1.mp4
                # feature_folder = os.path.join(folder, label_folder, str(i))
                feature_folder = os.path.join(s, str(i))
                # if 'mp4' in vid:
                if os.path.isdir(feature_folder):
                    if len(os.listdir(feature_folder)) < 150:
                        shutil.rmtree(feature_folder)
                    else:
                        continue
                os.makedirs(feature_folder, exist_ok=True)
                data_prepare(source=os.path.join(folder, label_folder, vid), save_f=feature_folder, clf_weights=r'D:\NCKH\tempo\yolov5\svm\lgbm.sav', transform=r'D:\NCKH\tempo\yolov5\transform\lgbm_scaler.bin')
                os.remove(os.path.join(folder, label_folder, vid))
    except:
        print(f'Proccessing for ./{label_folder}/{vid} is stoped +.+')

# if __name__ == "__main__":
#     # extract_features(r"D:\NCKH\tempo\yolov5\vids\orin_cut_vids")
#     # run(source="D:/NCKH/tempo/local_backup/vids/cro_me.mp4", clf_weights=r'D:\NCKH\working\transform_&_weight\acc_85_75_frames.h5', transform=r'D:\NCKH\working\transform_&_weight\lstm_scaler_2.bin', vid_stride=1)
#     run(source=0, clf_weights=r'D:\NCKH\working\transform_&_weight\acc_85_75_frames.h5', transform=r'D:\NCKH\working\transform_&_weight\lstm_scaler_2.bin')