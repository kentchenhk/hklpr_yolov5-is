import os

input_imgs = 'inference/input'
output_dir = 'inference/output'
iou = 0.3
conf = 0.5
img_size = 1600
weights_file = 'weights/yolov5s_detect.pt'

# Detect
os.system(f"python3 detect.py --img-size  {img_size} --conf-thres {conf}"
          f" --iou-thres {iou} --weights '{weights_file}' "
          f"--source '{input_imgs}' --output '{output_dir}'")
# os.system(f"python3 detect.py --img-size  {img_size} --conf-thres {conf} --iou-thres {iou} --weights '{weights_file}' --source '{input_imgs}' --output '{output_dir}' --device cpu")
