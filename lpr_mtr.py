import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from utils.datasets import letterbox
import numpy as np

import yaml
from queue import Queue
from threading import Thread
import warnings
import http.client


def similariity(a, b):
    if a is None and b is None:
        return 1.0
    elif a is None or b is None:
        return 0
    else:
        return SequenceMatcher(None, a, b).ratio()

def get_webservertime(host):
    conn=http.client.HTTPConnection(host)
    conn.request("GET", "/")
    r =conn.getresponse()
    ts =  r.getheader('date') # get the data pharse in http header 
    # change GMT time to Beijing time.
    ltime = time.strptime(ts[5:25], "%d %b %Y %H:%M:%S")
    print(ltime)
    # localtime
    # ttime =time.localtime(time.mktime(ltime)+8*60*60)
    # print(ttime)
    # dat="date %u-%02u-%02u"%(ttime.tm_year,ttime.tm_mon,ttime.tm_mday)
    # tm="time %02u:%02u:%02u"%(ttime.tm_hour,ttime.tm_min,ttime.tm_sec)
    # print (dat,tm)
    return ltime

# TODO:
def check_lp_lines_type(det, lp_lines_type, img_lp, img_lp0):
    gn = torch.tensor(img_lp0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if lp_lines_type == 0:
        """                      
        if actually two lines lp
        |   A B   |  
        | 1 2 3 4 |
        , then averaged y of '1' '4' should be quite different from averaged y of other characters or digits  
        """
        if det is not None and len(det)>2:
            xywh_list = []
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_lp.shape[2:], det[:, :4], img_lp0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                xywh_list.append(xywh)
            sorted_xywh_list = [x for x in sorted(xywh_list, key=lambda xywh_list: xywh_list[0])]
            ave_h = np.average(sorted_xywh_list, 0)[3]
            ave_y_outer = (sorted_xywh_list[0][1] + sorted_xywh_list[-1][1])/2
            ave_y_inner = np.average(sorted_xywh_list[1:-1], 0)[1]
            if abs(ave_y_outer - ave_y_inner) >  ave_h/3:
                lp_lines_type = lp_lines_type+1

    return lp_lines_type


# TODO: make sure that the criteria to separate the characters in line1/line2 is valid
def sort_characters(det, lp_lines_type, img_lp, img_lp0, names_recog):
    gn = torch.tensor(img_lp0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    line1_xywhc_list = []
    line2_xywhc_list = []
    sorted_line1_xywhc_list = []
    sorted_line2_xywhc_list = []
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img_lp.shape[2:], det[:, :4], img_lp0.shape).round()
        img_lp0_y_center = img_lp0.shape[0]
        for *xyxy, conf, cls in reversed(det):
            cls = int(cls.data.tolist())
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            if lp_lines_type == 0 or xywh[1]<0.5:
                line1_xywhc_list.append(xywh+[cls])
            else:
                line2_xywhc_list.append(xywh+[cls])
        sorted_line1_xywhc_list = [x for x in sorted(line1_xywhc_list, key=lambda line1_xywhc_list: line1_xywhc_list[0])]
        if len(line2_xywhc_list) > 0:
            sorted_line2_xywhc_list = [x for x in sorted(line2_xywhc_list, key=lambda line2_xywhc_list: line2_xywhc_list[0])]
    line1_license_str = ''.join([names_recog[xywhc[4]] for xywhc in sorted_line1_xywhc_list])
    line2_license_str = ''.join([names_recog[xywhc[4]] for xywhc in sorted_line2_xywhc_list])
    return line1_license_str + line2_license_str


class pred_post_proce_class():
    """
    added for mtr video post processing
    note that there the four hyper-parameters are quite important
    C&G;8&B
    """
    def __init__(self, k=15, day_light=False):
        if day_light:
            day_light_para = 5
        else:
            day_light_para = 0
        self.k = int(k-day_light_para)     # k same predictions then get a valid output
        self.queue = [None] * self.k
        self.label_returned = None
        self.high_min_limit = 30 - day_light_para
        self.weidth_min_limit = 75 - day_light_para
        self.bl = 5     # boundary min limit
        self.add_limit = 10     # for C&G;8&B
        self.hard_char = ['C', 'G', '8', 'B']
        self.last_label_returned = None
        self.label_returned_count = 0

    def _reset(self):
        for i in range(self.k):
            self.queue[i] = None
            self.label_returned = None
        return None

    def _update_label_returned(self):
        if self.label_returned is None:
            if self.queue[0] is None:
                return
            elif self.queue.count(self.queue[0]) == self.k:
                """
                When the cars are near the bar, the result may not be that accurate due to distortion. 
                Simi is added for that, can be removed once the model trained is better.
                """
                simi = similariity(self.label_returned, self.queue[0])
                if simi == 1.0 or simi < 0.8:
                    simi = similariity(self.last_label_returned, self.queue[0])
                    if simi == 1.0 or simi < 0.8:
                        self.label_returned = self.queue[0]
                        if self.last_label_returned != self.label_returned: # last_label_returned need to be updated later
                            self.label_returned_count = self.label_returned_count+1
        elif self.label_returned not in self.queue:
            self.label_returned = None

    def update_empty_frame(self):
        self.queue.pop(0)
        self.queue.append(None)
        self._update_label_returned()

    def update_nonempty_frame(self, lp_xyxy, img_size, label_str):
        # if the lp detected is too small, call reset
        hard_add_limit = 0
        for i in self.hard_char:
            if i in label_str:
                hard_add_limit = self.add_limit
                break
        if lp_xyxy[3]-lp_xyxy[1] < self.high_min_limit+hard_add_limit or lp_xyxy[2]-lp_xyxy[0] < self.weidth_min_limit+hard_add_limit:
            # Xiaodong: if the bbox is small, ignoring this frame, waitting the bbox large enough to get the accurate recognition.
            # label_str = None
            return self._reset()
        # if the boundin box is near the boundary, call reset()
        if lp_xyxy[0] < self.bl or lp_xyxy[1] < self.bl or lp_xyxy[2]+self.bl > img_size[1] or lp_xyxy[3]+self.bl > img_size[0]:
            return self._reset()

        self.queue.pop(0)
        self.queue.append(label_str)
        self._update_label_returned()

        return self.label_returned


def detect_recog(params, output_queue=None, save_img=False):
    dir_name = params['dir_name']
    if not dir_name.startswith(('rtsp://', 'rtmp://', 'http://')):
        source = params['input_dir']+params['dir_name']
    out = (params['output_dir']+params['dir_name']).replace('.', '_', 1)
    weights_detect, weights_recog, view_img, save_txt, imgsz_detect, imgsz_recog = \
        params['weights_file_detect'], params['weights_file_recog'],\
        params['view_img'], params['save_txt'], params['img_size_detect'], params['img_size_recog']
    if params['start_timming']:
        start_timming = time.strptime(params['start_timming'], r"%Y-%m-%d %H:%M:%S")
    else:
        start_timming = get_webservertime('www.google.com')  # TODO: need to consider the time synchornization problem.
    default_fps = params['default_fps']
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    device = select_device(params['device'])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_detect = attempt_load(weights_detect, map_location=device)  # load FP32 model
    model_recog = attempt_load(weights_recog, map_location=device)  # load FP32 model
    imgsz_detect = check_img_size(imgsz_detect, s=model_detect.stride.max())  # check img_size
    imgsz_recog = check_img_size(imgsz_recog, s=model_recog.stride.max())  # check img_size
    if half:
        model_detect.half()  # to FP16
        model_recog.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz_detect)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz_detect)

    # Get names and colors
    names_detect = model_detect.module.names if hasattr(model_detect, 'module') else model_detect.names
    names_recog = model_detect.module.names if hasattr(model_recog, 'module') else model_recog.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_detect))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz_detect, imgsz_detect), device=device)  # init img
    img_lp = torch.zeros((1, 3, imgsz_recog, imgsz_recog), device=device)  # init img
    _ = model_detect(img.half() if half else img) if device.type != 'cpu' else None  # run once
    _ = model_recog(img_lp.half() if half else img_lp) if device.type != 'cpu' else None  # run once

    pred_post_proce = pred_post_proce_class(day_light=params['day_light'])

    frameID = -1
    for path, img, im0s, vid_cap in dataset:
        frameID += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model_detect(img, augment=params['augment'])[0]
        # Apply NMS
        pred = non_max_suppression(pred, params['conf_detect'], params['iou_detect'], classes=params['classes_detect'], agnostic=params['agnostic_nms'])
        t2 = time_synchronized()
        all_t2_t1 = t2-t1
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is None or len(det)==0:
                pred_post_proce.update_empty_frame()
            elif det is not None and len(det):
                if len(det) > 1:  # only one license plate can occur in the entrance/exit at a time
                    warnings.warn("Warning: two bounding box coexist in the same frame!", UserWarning)
                    # print("Warning: two bounding box coexist in the same frame!")
                    continue
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names_detect[int(c)])  # add to string

                # Recognition and then Write results
                for *xyxy, conf, cls in reversed(det):

                    ''' Recognition '''
                    # Retrieve original resolution for each lp
                    img_lp0 = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]   # BGR
                    # Padded resize
                    img_lp = letterbox(img_lp0, new_shape=imgsz_recog)[0]
                    # Convert
                    img_lp = img_lp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img_lp = np.ascontiguousarray(img_lp)
                    img_lp = torch.from_numpy(img_lp).to(device)
                    img_lp = img_lp.half() if half else img_lp.float()  # uint8 to fp16/32
                    img_lp /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img_lp.ndimension() == 3:
                        img_lp = img_lp.unsqueeze(0)
                    t1 = time_synchronized()
                    # Inference
                    pred_lp = model_recog(img_lp, augment=params['augment'])[0]
                    # Apply NMS
                    pred_lp = non_max_suppression(pred_lp, params['conf_recog'], params['iou_recog'],
                                                  classes=params['classes_recog'], agnostic=params['agnostic_nms'])
                    t2 = time_synchronized()
                    all_t2_t1 = all_t2_t1 + t2 - t1
                    # Apply Classifier
                    if classify:
                        pred_lp = apply_classifier(pred_lp, modelc, img_lp, img_lp0)
                    # check_lp_lines_type
                    cls = check_lp_lines_type(pred_lp[0], cls, img_lp, img_lp0)
                    # Sort characters based on pred_lp
                    license_str = sort_characters(pred_lp[0], cls, img_lp, img_lp0, names_recog)
                    # *Note: if the mode is 'image', do not add the constraint. 
                    if dataset.mode != 'images':
                        license_str = pred_post_proce.update_nonempty_frame(xyxy, im0.shape[:2], license_str)
                    if license_str is None or len(license_str) == 0:
                        continue

                    if output_queue:  # *Note: put recognized lp and timestamp to the output queue.
                        _t = 10
                        while output_queue.full() and _t:
                            warnings.warn("The output queue buffer is full now, please clear or get items out. After" + str(_t) + " s will clear the queue.", UserWarning)
                            _t -= 1
                        if _t == 0: # *Clean the queue.
                            warnings.warn("Please check the status of update.", RuntimeWarning)
                        
                        timestamp = int(frameID/default_fps) + int(time.mktime(start_timming))
                        output_queue.put((timestamp, frameID, license_str))
                        print((timestamp, frameID, license_str))
                        # TODO: The code should be rewrote to support the multi LP in one frame. According to the requirement to give the output

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (license_str)
                        # label = '%s %.2f' % (license_str, conf)
                        line_thickness = 2 if im0.shape[0] < 500 else 3
                        xyxy_larger = [xyxy[0]-8, xyxy[0]-8, xyxy[0]+8, xyxy[0]+8]
                        plot_one_box(xyxy, im0, label=label, color=[200,100,20], line_thickness=line_thickness)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, all_t2_t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                # TODO: support RTSP stream
                else: # !only support video in this branch.
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

                    # save frame and txt for newly recognized LP
                    if pred_post_proce.last_label_returned != pred_post_proce.label_returned \
                            and pred_post_proce.label_returned is not None:
                        pred_post_proce.last_label_returned = pred_post_proce.label_returned
                        dir, filename = os.path.split(save_path)
                        frame_dir = dir + os.path.sep + os.path.splitext(filename)[0]
                        if not os.path.exists(frame_dir):
                            os.makedirs(frame_dir)
                        frame_path = frame_dir + os.path.sep + os.path.splitext(filename)[0] + "_frame%04d.jpg" % (int(pred_post_proce.label_returned_count))
                        if not os.path.isfile(frame_path):
                            cv2.imwrite(frame_path, im0)

                            with open(dir + os. path.sep + os.path.splitext(filename)[0] + '.txt', 'a') as ff:
                                ff.write(('%04d  %s \n' + '') % (int(pred_post_proce.label_returned_count), license_str))  # label format

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        # if platform.system() == 'Darwin' and not opt.update:  # MacOS
        #     os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


class lpr_mtr():
    def __init__(self, yaml_path, queueSize=300):
        self.yaml_path = yaml_path
        self.queueSize = queueSize
        f = open(self.yaml_path)
        self.params = yaml.load(f)
        self.queue = Queue(maxsize=self.queueSize)
    
    def run_worker(self, target):
        p = Thread(target=target, args=(), kwargs={})
        p.start()
        return p

    def run(self):
        self.lpr_mtr_worker = self.run_worker(self.run_video_lpr)
        return self
    
    def stop(self):
        self.lpr_mtr_worker.join()
        self.clear_queues()
    
    def clear_queues(self):
        self.clear(self.queue)
    
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)
    
    def wait_and_get(self, queue):
        return queue.get()
    
    def read_result(self):
        if not self.queue.empty():
            return self.wait_and_get(self.queue)
        else:
            return (None, None, None)
    
    def run_video_lpr(self, **kwargs):
        with torch.no_grad():
            detect_recog(self.params, self.queue)
        # print(self.read_result())


if __name__ == '__main__':
    LPR_MTR = lpr_mtr(yaml_path='./lpr_mtr.yaml')
    LPR_MTR.run() # *Call LPR_MTR.stop() when you want to stop the thread.
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights_detect', nargs='+', type=str, default='weights/yolov5s_detect.pt', help='model.pt path(s)')
    # parser.add_argument('--weights_recog', nargs='+', type=str, default='weights/yolov5s_recog.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/input', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    # parser.add_argument('--img-size_detect', type=int, default=1600, help='inference size (pixels)')
    # parser.add_argument('--img-size_recog', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--conf-thres_detect', type=float, default=0.4, help='object confidence threshold')
    # parser.add_argument('--conf-thres_recog', type=float, default=0.6, help='object confidence threshold')
    # parser.add_argument('--iou-thres_detect', type=float, default=0.3, help='IOU threshold for NMS')
    # parser.add_argument('--iou-thres_recog', type=float, default=0.3, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--classes_detect', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--classes_recog', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--day_light', action='store_true', help='day light')
    # opt = parser.parse_args()
    # print(opt)