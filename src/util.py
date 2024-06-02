# my
# v1 - 201015

import cv2 as cv
import numpy as np
import os


def path_combine(root, path):
    return root + '\\' + path


# def capture_frame(Cap, sep=1, start_num=0, end_num=0):
#     fps, frame_num = Cap.get(cv.CAP_PROP_FPS), Cap.get(cv.CAP_PROP_FRAME_COUNT)
#
#     print('FPS: %d\nTotal frames: %d' % (fps, int(frame_num)))
#
#     for i in range(int(frame_num)):
#         ret, frame = Cap.read()
#         if i < start_num-1:
#             continue
#         elif end_num and i > end_num-1:
#             return
#         elif (i+1) % sep:
#             continue
#         else:
#             yield i+1, frame


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
