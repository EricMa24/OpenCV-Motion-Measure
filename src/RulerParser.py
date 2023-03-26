"""
xxx
"""
from typing import *
import numpy as np
import os.path as opath

import pandas as pd
import cv2.cv2 as cv

from .TickMarkReader import TickMarkReader
from .util import mkdir

# Type hints
ColorImage = np.ndarray(shape=(Any, Any, 3), dtype=int)
ROI = Tuple[int, int, int, int]  # x, y, width, height
Tick = Tuple[int, float]  # Tick[0]: index; Tick[1]: x-location (horizontal) in pixel


class RulerParser:
    def __init__(self):
        pass

    def parse_from_video(self, video_path, output_img: Union[str, bool] = 'details', **kwargs):
        roi = kwargs.pop('ROI')
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        step = kwargs.pop('step', None)
        output_path = kwargs.pop('output_path', None)

        print(f'Start parsing\n  output image mode: {output_img}')

        sum_res = {}
        prev_ticks: List[Tick] = []
        tick_reader = TickMarkReader()
        for i, frame in self.capture_frame(video_path,
                                           start=start, end=end, step=step):
            frame_name = "Frame%04d" % i
            res, img_log, ticks = tick_reader.run(frame_name, frame, roi=roi, prev_ticks=prev_ticks)  # <===============
            prev_ticks = ticks

            msg = f'Parsed {frame_name}: major: {res["reading_major"]} minor: {res["reading_minor"]}'
            print('\x08' * (len(msg) + 12), end='')
            print(msg, end='', flush=True)

            sum_res.update({frame_name: res})

            # save img results
            if not output_path:
                output_path = r'.%s%s_%s' % (opath.sep, 'results', video_path.split(opath.sep)[-1][:-4])
                mkdir(output_path)
            if not output_img:
                continue
            elif output_img == 'result-only':
                self.save_result_img(frame_name, img_log, output_path)
            elif output_img == 'details':
                self.save_details_img(frame_name, img_log, output_path)
            else:
                raise ValueError(f'Not such output image mode: {output_img}')

        # results
        res_df = pd.DataFrame(sum_res).T
        excel_name = 'Results_%s.xlsx' % video_path.split("\\")[-1]
        res_df.to_excel(excel_name)
        print(f'\nResults saved to {excel_name}')

    @staticmethod
    def capture_frame(video_path: str, start=0, end=-1, step=1)\
            -> Generator[Tuple[int, ColorImage], None, None]:

        cap = cv.VideoCapture()
        if not cap.open(video_path):
            raise ValueError(f'Fail to open {video_path}')
        print(f'Read video file: {video_path}')

        fps, frame_num = cap.get(cv.CAP_PROP_FPS), cap.get(cv.CAP_PROP_FRAME_COUNT)
        print('    FPS: %d Total frames: %d' % (fps, int(frame_num)))

        if end == -1:
            end = int(frame_num)
        for i in range(int(frame_num)):
            if (start <= i < end and (i % step) == 0) or (i == end - 1):
                retval, frame = cap.read()
                if retval:
                    yield i + 1, frame
                else:
                    print(f'    Fail to read frame: {i + 1} error code: {retval}')

    @staticmethod
    def save_result_img(name, img_log, output_path):
        try:
            img = img_log['results_plot']
            cv.imwrite(opath.sep.join([output_path, 'results_plot_' + name + '.jpg']), img)
        except KeyError:
            pass

    @staticmethod
    def save_details_img(name, img_log, output_path):
        for i, action_name in enumerate(TickMarkReader.IMAGE_PROCESS_SERIES):
            prefix = "%02d_" % i
            try:
                img = img_log[action_name]
                cv.imwrite(opath.sep.join([output_path, prefix + action_name + '_' + name + '.jpg']), img)
            except KeyError:
                continue
