import argparse
from .configs import CONFIG

arg_parser = argparse.ArgumentParser(prog='run.py',
                                     description="Process analysis or select ROI, take an input video filename")

arg_parser.add_argument('video_file', metavar='VIDEO-FILE', type=str, default=None,
                        help="video file name")

# arg_parser.add_argument('first_major', metavar='FIRST-MAJOR', type=str, default=None,
#                         help="first major of ROI")

arg_parser.add_argument('--output', metavar='OUTPUT-MODE', type=str, nargs='?',
                        choices=['result-only', 'details'], default='result-only',
                        help="['result_only', 'details']")


# running subset of video frames
arg_parser.add_argument('--start', type=int, default=0, nargs='?', help="Start frame (0-Indexed)")
arg_parser.add_argument('--end', type=int, default=-1, nargs='?', help="End frame (0-Indexed)")
arg_parser.add_argument('--step', type=int, default=1, nargs='?', help="Step of reading frame")

if __name__ == '__main__':
    arg_parser.print_help()
