# import os
#
# dataset_yaml = "data/lp_recognition.yaml"
# # n_gpu = 1
# # device = '0'
#
# n_gpu = 2
# device = '0,1'
# # img_size_train = 416
# # img_size_train = 320
# img_size_train = 640
# batch_size_per_gpu = 16
#
# img_size_test = img_size_train
# epochs = 100
# # model_yaml = 'yolov5s_lprecog'
# # model_yaml = 'yolov5m_lprecog'
# model_yaml = 'yolov5x_lprecog'
# # weights_file = 'weights/yolov5s.pt'
# batch_size = n_gpu * batch_size_per_gpu
# hyp_yaml = 'hyp.lp_recognition.yaml'
# workers = 0
#
# os.system(f"python3 train.py --multi-scale --img-size {img_size_train} {img_size_test}"
#           f" --batch-size {batch_size} --epochs {epochs} --data {dataset_yaml} "
#           f"--cfg ./models/{model_yaml}.yaml --multi-scale --hyp {hyp_yaml} "
#           f"--name {model_yaml}_results --device={device} --cache-images --workers={workers}")
#           # f"--name {model_yaml}_results --device={device} --cache-images --workers={workers} --noautoanchor")


"""for mtr lpr with front lp only
#  Error analysis: C&G;J&U;X&Y;C&0;8&B
#
#
"""
import os

dataset_yaml = "data/lp_recognition.yaml"
n_gpu = 1
device = '1'

img_size_train = 416
batch_size_per_gpu = 16

img_size_test = img_size_train
epochs = 500
model_yaml = 'yolov5s_lprecog'
weights_file = 'weights/yolov5s_recog.pt'
# weights_file = 'runs/exp41_yolov5s_lprecog_results/weights/best.pt'
batch_size = n_gpu * batch_size_per_gpu
hyp_yaml = 'hyp.lp_recognition_mtr.yaml'

os.system(f"python3 train.py --multi-scale --img-size {img_size_train} {img_size_test}"
          f" --batch-size {batch_size} --epochs {epochs} --data {dataset_yaml} "
          f"--cfg ./models/{model_yaml}.yaml --multi-scale --hyp {hyp_yaml} "
          # f"--name {model_yaml}_results --device={device} --cache-images")
          f"--name {model_yaml}_results --device={device} --weights {weights_file} --cache-images")
