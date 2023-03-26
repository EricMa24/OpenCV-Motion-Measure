import argparse
from .configs import CONFIG

arg_parser = argparse.ArgumentParser(prog='CalibrateLWT.py',
                                     description="Process analysis or select ROI, take an input video filename")

arg_parser.add_argument('--ROI', metavar=('x', 'y', 'w', 'h'), type=int, nargs=4, default=CONFIG.ROI_RANGE,
                        help="x, y, width, height")
arg_parser.add_argument('--output', metavar='OUTPUT-MODE', type=str, nargs='?',
                        choices=['result-only', 'details'], default='result-only',
                        help="['result_only', 'details']")
arg_parser.add_argument('video_file', metavar='VIDEO-FILE', type=str,
                        help="video file name")
arg_parser.add_argument('--ROI-only', action=argparse.BooleanOptionalAction, default=False,
                        help="Open ROI Selector (without run analysis)")

# running subset of video frames
arg_parser.add_argument('--start', type=int, nargs='?', help="Start frame (0-Indexed)")
arg_parser.add_argument('--end', type=int, nargs='?', help="End frame (0-Indexed)")
arg_parser.add_argument('--step', type=int, nargs='?', help="Step of reading frame")

if __name__ == '__main__':
    arg_parser.print_help()
