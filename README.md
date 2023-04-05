# OpenCV-Motion-Measure
A Python project use OpenCV and GoPro camera to measure/calibrate object motion.  

## An example

## Installation 

## Run Script

### Set Image ROI

To determine the coordinates of ROI, run script with input video file and pass “--ROI-only”:
> python ruler_recog.py “.\demo\demo.mp4” --ROI-only

In the opened window, adjust ROI as example shows. 
- ROI should be in the middle of image as possible for avoiding distortion. 
- ROI should cover 2 – 4 major ticks.
- Make sure all of ROI be within the ruler tape to have consistent background color.  

<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/roi_selector.png" alt="roi_selector" width="700">

### Run Analysis

Run script with input video file and ROI values:
> python CalibrateLWT.py “.\demo\demo.mp4” --ROI  520 440 240 70

It can also take an input of the start tick, 
which make no difference to analysis but let results looks better:
> python CalibrateLWT.py “.\demo\demo.mp4” --ROI  520 440 240 70   -start-tick 21

When it is done, there will be 

## An Example:

Input Video (/demo/demo.mp4):

<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/camera_view.png" alt="camera_view" width="700">


Recognition results:  

<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/demo_result1.jpg" alt="demo_result1" width="600">
<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/demo_result2.jpg" alt="demo_result2" width="600">

The table of summary results generated from script:

<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/motion_results.png" alt="motion_results" width="700">

All frames of recognition results can be combined back into video for a better look

> python ./Scripts/gen_demo_video.py 
