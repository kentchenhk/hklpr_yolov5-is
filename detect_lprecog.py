# import os
#
# input_imgs = 'inference/input'
# output_dir = 'inference/output'
# iou = 0.3
# conf = 0.5
# img_size = 416
# weights_file = 'weights/yolov5s_recog.pt'
#
# # Detect
# os.system(f"python3 detect.py --img-size {img_size} --conf-thres {conf} --iou-thres {iou} --weights '{weights_file}' --source '{input_imgs}' --output '{output_dir}'")
# os.system(f"python3 detect.py --img-size {img_size} --conf-thres {conf} --iou-thres {iou} --weights '{weights_file}' --source '{input_imgs}' --output '{output_dir}' --device cpu")


"""run the detect,py to pre-label the LP and save the .txt file"""
import os

device = '0'
iou = 0.3
conf = 0.5
img_size = 416
weights_file = 'weights/yolov5s_mtr_recog_416.pt'
# paths = ["train", "valid", "test"]
# for i in paths:
#     input_imgs = "/home/jrchan/data/LPR_datasets/hklp_mtr_recog_phesudo/" + i + "/images"
#
#     output_imgs_dir = "/home/jrchan/data/LPR_datasets/hklp_mtr_recog_phesudo/" + i + "/labels"
#
#     # Detect
#     os.system(f"python3 detect.py --img-size  {img_size} --conf-thres {conf} "
#               f"--iou-thres {iou} --weights '{weights_file}' "
#               f"--device={device} --source {input_imgs} --output {output_imgs_dir} --save-txt")

input_imgs = '/home/jrchan/data/LPR_datasets/misc_crop/all_1'
output_imgs_dir = '/home/jrchan/data/LPR_datasets/misc_crop/all_1_labels'

# Detect
os.system(f"python3 detect.py --img-size  {img_size} --conf-thres {conf} "
          f"--iou-thres {iou} --weights '{weights_file}' "
          f"--device={device} --source {input_imgs} --output {output_imgs_dir} --save-txt")
