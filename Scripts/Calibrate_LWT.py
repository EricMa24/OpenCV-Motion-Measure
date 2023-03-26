"""
Parse a video and label frames. Related settings are listed in <configs.py>
"""
# my
# v2 -201023

import sys
import argparse

from src.RulerParser import RulerParser

# video_file = None
video_path = r"E:\OneDrive - Louisiana State University\000\201005_LWT_openCV\22_data\raw_video\230207_GOPR0539.MP4"


if __name__ == '__main__':
    if not video_path:
        try:
            video_path = sys.argv[1]
        except IndexError:
            raise ValueError('No video path!')

    # print(sys.argv[1])
    calibrator = RulerParser()
    calibrator.parse_from_video(video_path,
                                end=None,
                                output_img='result_only')  # ['result_only', 'details']
