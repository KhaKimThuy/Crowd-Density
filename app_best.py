from flask_socketio import SocketIO
from flask import Flask, render_template, request
from joblib import load
import base64
import eventlet
import cv2
import sys

###############################
import os
import time
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from new_idea import *
############################################


global ip
ip = input('Enter ip address local machine: ')
app = Flask(__name__)
socketio = SocketIO(app)

greelet_dict = {}

source_folder = 'D:/NCKH/tempo/local_backup/vids'

monitored_area = {}


#######################################

def proc(
        source = None,
        clf_weights = ROOT / 'transform_&_weight/acc_85_75_frames.h5',
        transform = ROOT / 'transform_&_weight/lstm_scaler_2.bin',
        lk_params = dict( winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        feature_params = dict(maxCorners = 200,
            qualityLevel = 0.6,
            minDistance = 5,
            blockSize = 4),
        sid = None,
        monitor = None
):
#########################################

    if source==None:
        print('=========== Please choose a video ==============')
        sys.exit()


    print("====================== SOURCE+++: ", source)
    points, view_mask = focus_on_region(source)
    density_label = ["Level 1", "Level 2", "Level 3", "Level 4", 'Level 5']
    areaROI = cal_ROI(points)
    get_prv_frame = False
    video = cv2.VideoCapture(source)
    density = "Loading..."

    clf = load_model(clf_weights)
    sc = load(transform)

#########################################
    start = time.time()
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
                init_flag = False

        else:
            if len(data) >= 75*14:
                data = np.array([data[-75*14:]])
                data = sc.transform(data)
                data = data.reshape(1, 75, 14)
                pred = clf.predict(data)
                pred = np.argmax(pred).tolist()
                density = density_label[pred]
                data = []

        cv2.putText(opf, f'Density: {str(density)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(opf, f'Init flag: {str(init_flag)}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(opf, f'Length of data: {str(len(data))}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(opf, f'Frame rate: {frame_rate}', (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 80), 1)
        frame_id += 1

        monitored_area[sid] = [monitor, density]

        opf = cv2.resize(opf, (0,0), fx=0.5, fy=0.5)
        frame = cv2.imencode('.jpg', opf)[1].tobytes()
        frame = base64.encodebytes(frame).decode("utf-8")
        socketio.emit('estimate', {'dens': density,'obs': frame, 'sid':sid, 'name_m':monitor}, namespace='/admin')
        eventlet.sleep(0.001)

def transmit(sid):
    global result
    params = result
    if params['source'] != "0":
        params['monitor'] = params['source'].split("\\")[-1]
        params['source'] = os.path.join(source_folder, params['source'])
    else:
        params['monitor'] = params['source']
        params['source'] = int(params['source'])

    socketio.emit('monitor_request', {"sid":sid, 'monitor': params['monitor']}, namespace='/admin')


    # params['clf_weights'] = r'D:\DL\torch\acc_85_75_frames.h5'
    # params['transform'] = r'D:\DL\torch\lstm_scaler_2.bin'
    params['sid'] = sid
    return proc(**params)


@app.route('/')
def index():
    domain = "http://"+ip+":9099/moderator"
    files = os.listdir(source_folder)
    return render_template("index_loop.html", domain=domain, src=source_folder, files=files)

@app.route('/moderator', methods=['POST', 'GET'])
def res():
    global result
    global ip
    if request.method=='POST':
        result = request.form.to_dict()
        return render_template("loop.html", ip=ip)


@app.route('/client', methods=['GET'])
def client_side():
    return render_template("index3_1_get.html", areas=monitored_area, broadcast=True)

@socketio.on('connect', namespace='/admin')
def res_connect():
    global greelet_dict
    sid = request.sid
    # create new Greenlet and start long-running task
    gevent = eventlet.spawn(transmit, sid)
    # save Greenlet instance in dictionary (key = session ID)
    greelet_dict[sid] = gevent

@socketio.on('cancel_task', namespace='/admin')
def handle_cancel_task(msg):
    # get session ID of client where the event starts
    sid = msg['client_id'][:-4]
    # retrieve Greenlet instance corresponding to the session ID
    gevent = greelet_dict.get(sid)
    if gevent:
        greelet_dict.pop(sid)
        monitored_area.pop(sid)
        gevent.kill(eventlet.greenlet.GreenletExit)
        socketio.emit('task_cancelled', {'sid_end':sid}, namespace='/admin')

@socketio.on('disconnect', namespace='/admin')
def res_disconnect():
    sid = request.sid
    print(f'Client {sid} disconnected')

if __name__ == "__main__":
    socketio.run(app, host=ip, port=9099)

# cmd: ipconfig -> kéo xuống cuối cùng gán ip = IPv4_address (ví dụ: ip = 192.168.1.77)
# Chạy trên web: http:// + ip + :9099/ (Chạy video 1, chạy video 2 vẫn nhập lại domain này)
# Chọn Video (chỉ có 2 cái để test) -> xem các video có sãn trong folder vids
# Để ý task bar, chương trình sẽ hiện ảnh lên yêu cầu người dùng chọn vùng
# -> Click chuột khoanh vùng cần detect và nhấn "d" để chạy chương trình
# Sau khi web chạy, trong android, ta sửa địa chỉ ip thành "http:// + ip + :9099/client"

# 10.0.134.9
