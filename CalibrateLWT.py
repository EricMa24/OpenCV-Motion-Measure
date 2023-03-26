# Eric MA y1ma9494@gmail.com
# v3 -230208
"""
usage: CalibrateLWT.py [-h] [--ROI x y w h] [--output [OUTPUT-MODE]]
                       [--ROI-only | --no-ROI-only] [--start [START]]
                       [--end [END]] [--step [STEP]]
                       VIDEO-FILE

Process analysis or select ROI, take an input video filename

positional arguments:
  VIDEO-FILE            video file name

options:
  -h, --help            show this help message and exit
  --ROI x y w h         x, y, width, height
  --output [output_mode]
                        ['result-only', 'details']
  --ROI-only, --no-ROI-only
                        Open ROI Selector (without run analysis) (default:
                        False)
  --start [START]       Start frame (0-Indexed)
  --end [END]           End frame (0-Indexed)
  --step [STEP]         Step of reading frame

Process finished with exit code 0

"""

from src.argParser import arg_parser
from src.RulerParser import RulerParser


if __name__ == '__main__':
    args = arg_parser.parse_args()
    # ArgParser.print_help()

    calibrator = RulerParser()
    calibrator.parse_from_video(args.video_file,
                                ROI=args.ROI,
                                start=args.start,
                                step=args.step,
                                end=args.end,
                                output_img=args.output)
