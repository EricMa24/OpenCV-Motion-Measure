# OpenCV-Motion-Measure
A Python project use OpenCV and GoPro camera to measure/calibrate object motion.  

## An Example:

Input Video (/demo/demo.mp4):

<img src="/img/demo.gif" alt="demo.mp4" autoplay="true">

Recognition Results:  

<img src="/img/result.gif" alt="result" autoplay="true">

Processing Flowchart:

<img src="/img/flowchart.png" alt="flowchart" autoplay="true">

## Installation 
Install Python libraries
> numpy~=1.23.3 
> 
> pandas~=1.4.4 
> 
> opencv-python==4.5.5.62 

## Run Script

Run script with input video file:
> python run.py “.\demo\demo.mp4”

### Set Image ROI

In the opened window, adjust ROI as example shows. 
- ROI should be in the middle of image as possible for avoiding distortion. 
- ROI should cover 2 – 4 major ticks.
- Make sure all of ROI be within the ruler tape to have consistent background color.  

<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/roi_selector.png" alt="roi_selector" width="700">

### Run Analysis

Press "K" to start analysis

When done, a Excel sheet of summary results would be generated:

<img src="https://github.com/EricMa24/OpenCV-Motion-Measure/blob/master/img/motion_results.png" alt="motion_results" width="700">

## run.py Usages
usage: run.py [-h] [--output [OUTPUT-MODE]] [--start [START]] [--end [END]] [--step [STEP]] VIDEO-FILE

positional arguments:
  VIDEO-FILE            video file name

options:
  -h, --help            show this help message and exit
  --output [OUTPUT-MODE]
                        ['result_only', 'details']
  --start [START]       Start frame (0-Indexed)
  --end [END]           End frame (0-Indexed)
  --step [STEP]         Step of reading frame
