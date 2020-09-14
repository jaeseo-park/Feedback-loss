# Feedback-loss
**Feedback-based object detection loss function for 2stage pose estimator.**


This code is designed to optimize the object detector to the pose estimator.

This code is currently implemented for HRNet with yolov3 as object detector.

We will additionally optimize for another 2stage pose estimator.





# Usage

- git clone https://github.com/jaeseo-park/simple-HRNet
- And, in this simple-HRNet, replace the 'models/detectors/yolo' folder with the git code below.
- git clone https://github.com/jaeseo-park/PyTorch-YOLOv3


**Train**


In 'models/detectors/yolo'

```
$ python train_feedback.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74
```


**Try**

- Follow the Simple-HRNet readme(https://github.com/jaeseo-park/simple-HRNet/blob/master/README.md)
