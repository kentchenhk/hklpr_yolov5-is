import os

dataset_yaml = "data/lp_recognition.yaml"
# n_gpu = 1
# device = '1'

n_gpu = 2
device = '0,1'
img_size_train = 416
batch_size_per_gpu = 16

img_size_test = img_size_train
epochs = 200
model_yaml = 'yolov5s_lprecog'
# weights_file = 'weights/yolov5s.pt'
batch_size = n_gpu * batch_size_per_gpu
hyp_yaml = 'hyp.lp_recognition.yaml'

os.system(f"python3 train.py --multi-scale --img-size {img_size_train} {img_size_test}"
          f" --batch-size {batch_size} --epochs {epochs} --data {dataset_yaml} "
          f"--cfg ./models/{model_yaml}.yaml --multi-scale --hyp {hyp_yaml} "
          f"--name {model_yaml}_results --device={device} --cache-images")
