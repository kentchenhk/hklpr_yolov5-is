# import os
#
# input_imgs = 'inference/input'
# output_dir = 'inference/output'
# device = '0,1'
#
# iou_detect = 0.3
# conf_detect = 0.5
# img_size_detect = 1600
# weights_file_detect = 'weights/yolov5s_detect.pt'
#
# iou_recog = 0.3
# conf_recog = 0.5
# img_size_recog = 416
# weights_file_recog = 'weights/yolov5s_recog.pt'
#
# # Detect and recog
# os.system(f"python3 lp_detect_recog.py --img-size_detect  {img_size_detect} --img-size_recog  {img_size_recog}"
#           f" --conf-thres_detect {conf_detect} --conf-thres_recog {conf_recog}"
#           f" --iou-thres_detect {iou_recog} --iou-thres_detect {iou_recog}"
#           f" --weights_detect '{weights_file_detect}' --weights_recog '{weights_file_recog}'"
#           f" --source '{input_imgs}' --output '{output_dir}' --device={device}")
# # os.system(f"python3 detect.py --img-size  {img_size} --conf-thres {conf} --iou-thres {iou} --weights '{weights_file}' --source '{input_imgs}' --output '{output_dir}' --device cpu")


"""for mtr lpr with front lp only"""
import os

# dir_name = 'iphone1_exit'
# dir_name = 'iphone0_entrance'
# dir_name = 'samsung0_entrance'
# dir_name = 'official_test'
# dir_name = 'official_test/20210508_1_edit.mp4'
# dir_name = 'official_test/20210508_2_edit.mp4'
# dir_name = 'official_test/20210508_3_edit.mp4'
# dir_name = 'official_test/20210508_3_edit_edit.mp4'
# dir_name = 'official_test/20210508_4_edit.mp4'
# dir_name = 'official_test/20210509_1_edit.mp4'
# dir_name = 'official_test/20210509_2_edit.mp4'
dir_name = 'official_test/20210510_1_edit.mp4'
# dir_name = 'official_test/20210508_all.mp4'
# dir_name = 'official_test/20210509_all.mp4'

# dir_name = 'single'
input_videos = '/home/jrchan/data/LPR_datasets/mtr/input/' + dir_name
output_dir = '/home/jrchan/data/LPR_datasets/mtr/output/' + dir_name
# device = '0,1'
# device = '1'
device = '0'

iou_detect = 0.3
conf_detect = 0.7   # Should be higher
img_size_detect = 640
weights_file_detect = 'weights/yolov5s_mtr_detect_416.pt'

iou_recog = 0.2     # Should be lower
# conf_recog = 0.65    # Should be higher
conf_recog = 0.65    # Should be higher
img_size_recog = 416
# weights_file_recog = 'weights/yolov5s_mtr_recog_416.pt'
# weights_file_recog = 'runs/exp53_yolov5s_lprecog_results/weights/best.pt'
weights_file_recog = 'runs/exp75_yolov5s_lprecog_results/weights/best.pt'


# Detect and recog
os.system(f"python3 lp_detect_recog_mtr_videos.py --img-size_detect  {img_size_detect} --img-size_recog  {img_size_recog}"
          f" --conf-thres_detect {conf_detect} --conf-thres_recog {conf_recog}"
          f" --iou-thres_detect {iou_recog} --iou-thres_detect {iou_recog}"
          f" --weights_detect '{weights_file_detect}' --weights_recog '{weights_file_recog}'"
          # f" --source '{input_videos}' --output '{output_dir}' --device={device}")
          f" --source '{input_videos}' --output '{output_dir}' --device={device} --day_light")

