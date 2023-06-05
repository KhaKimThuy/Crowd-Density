import cv2
import glob
import os
import numpy as np
import shutil
import time




def get_last_file_name(dir_):
    try:
        dir_ = dir_.replace("/","/")
        list_of_files = glob.glob(dir_+"/*") # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        return "".join((char if char.isalnum() else " ") for char in latest_file).split()[-2]
    except:
        return '0'

def cut_video(vid_path, frame_limit, frame_rate):

    orin = vid_path.split('.')[-1]

    path = 'D:/NCKH/tempo/yolov5/vids/orin_cut_vids'
    # if 
    vid_num = int(get_last_file_name(path))
    video = cv2.VideoCapture(vid_path)
    frame_num = frame_limit + 1
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        if frame_num > frame_limit:
            # Prepare for new cut video
            frame_num = 1
            vid_num += 1

            video_name = os.path.join(path,orin+'_'+str(vid_num)+'.mp4')
            video_save_path = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

        ret, img = video.read()
        if not ret:
            print(f'Terminate at: {frame_num}')
            print('Done!')
            break

        video_save_path.write(img)
        frame_num += 1

def optical_flow(vid):
    # chuỗi hình là một danh sách các khung hình
    video = cv2.VideoCapture(vid)
    ret, frame1 = video.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    s = time.time()
    while True:
        ret, frame2 = video.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # tính toán hướng x và y của dòng chảy quang học (optical flow)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        print(f'---------- Magnitude: {mag} - Shape: {ang.shape}')
        print(f'---------- Angle: {ang} - Shape: {ang.shape}')
        cv2.imshow('optical flow',hsv[...,1])
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        # chuỗi khung hình tiếp theo sẽ trở thành chuỗi khung hình trước
        prvs = next
    print(f'Complete in {time.time() - s} s')

def label_vid(folder):
    stop = ''
    try:
        for vid in os.listdir(folder):
            if 'mp4' in vid:
                stop = vid
                video = cv2.VideoCapture(os.path.join(folder,vid))
                fps = int(round(video.get(cv2.CAP_PROP_FPS)))
                s = time.time()
                while True:
                    ret, img = video.read()
                    if not ret:
                        break
                    cv2.imshow("Monitor", img)
                    if cv2.waitKey(1000//fps) & 0xFF == ord('q'):
                        exit()
                        
                if time.time() - s - 2 < 4:
                    continue
                
                video.release()
                cv2.destroyAllWindows()
                label = input(f'>>>>>> Label for {stop}: ')

                source = os.path.join(folder,vid)
                destination = os.path.join(folder, label)
                shutil.move(source, destination)
    except:
        print(f"\nNot label for --------- {stop}")



# for i in os.listdir('D:/NCKH/tempo/yolov5/vids'):
#     if 'mp4' in i:
#         cut_video(f'D:/NCKH/tempo/yolov5/vids/{i}', 150, 30)
#         print(f'--------------------- Video {i} ---------------------')



# print(get_last_file_name('D:/NCKH/tempo/yolov5/vids/1'))


# train(r'D:\NCKH\tempo\yolov5\vids\orin_cut_vids')



# optical_flow(r'D:\NCKH\tempo\yolov5\vids\cut_vids\mp4_1.mp4')

# label_vid(r'D:\NCKH\tempo\yolov5\vids\orin_cut_vids')
#   cut_vids
#      |-vid1.mp4
#      |-vid2.mp4
#      |-vid3.mp4
#      |-Label_folder_1
#      |-Label_folder_2
