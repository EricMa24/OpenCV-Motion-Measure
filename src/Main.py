"""
xxx
"""
import os.path
from typing import *
import os.path

import pandas as pd
import cv2 as cv

from src.TickMarkReader import TickMarkReader
from src.VideoParser import VideoParser
from src.util import mkdir
from src.configs import CONFIG


class Main:
    def __init__(self):
        pass

    def main(self, video_path,
             output_flag='details',
             show_original_window=True,
             show_labeled_window=True,
             **kwargs):

        roi = kwargs.get('ROI')
        first_major = kwargs.get('FirstMajor', CONFIG.FIRST_MAJOR)
        start, end, step = [kwargs.get(key) for key in ['start', 'end', 'step']]
        user_output_path = kwargs.get('output_path', None)

        # Parse Video
        parser = VideoParser()
        if parser.load_video(fp=video_path) < 0:
            return
        print(f'Start parsing\n  output image mode: {output_flag}')

        # Analysis
        sum_res = {}
        config = CONFIG(ROI=roi, FIRST_MAJOR=first_major)
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        try:  # Ensure release source
            last_majors = None
            # Loop frames
            for i, frame in parser.capture_frame(start=start, end=end, step=step):
                frame_name = "Frame%04d" % i
                res, img_log, last_majors = self.read_single_frame(frame_name, frame,  # <==============================
                                                                   config=config,
                                                                   last_majors=last_majors)

                sum_res.update({frame_name: res})

                # Show original frame window
                if show_original_window:
                    self.update_original_frame(frame_name, frame)
                # Show labeled frame window
                if show_labeled_window:
                    self.update_labeled_frame(frame_name, img_log)
                # Export labeled frame
                self.export_single_frame(output_flag, frame_name, img_log,
                                         video_filename=video_filename,
                                         user_output_path=user_output_path)
        finally:
            parser.release()

        # Export Results
        res_df = pd.DataFrame(sum_res).T
        excel_name = 'Results_%s.xlsx' % video_filename
        self.export_df_safe(res_df, excel_name, root_dir=user_output_path)

    @staticmethod
    def read_single_frame(frame_name, frame, config, last_majors):
        tick_reader = TickMarkReader(cfg=config)
        res, img_log, last_majors = tick_reader.run(frame_name, frame, last_majors=last_majors)  # <====================

        msg = f'Parsed {frame_name}: major: {res["reading_major"]} minor: {res["reading_minor"]}'
        print('\x08' * (len(msg) + 12), end='')
        print(msg, end='', flush=True)

        return res, img_log, last_majors

    def export_single_frame(self, output_flag, frame_name, img_log,
                            video_filename, user_output_path):
        if not output_flag:
            return

        # Create dir
        if not user_output_path:
            img_out_path = r'.%s%s_%s' % (os.path.sep, 'results', video_filename)
        elif not os.path.exists(user_output_path):
            img_out_path = r'.%s%s_%s' % (os.path.sep, 'results', video_filename)
            print(f"{user_output_path} not exist, use {img_out_path} instead")
        else:
            img_out_path = r'%s.%s%s_%s' % (user_output_path, os.path.sep, 'results', video_filename)

        # Export
        os.mkdir(img_out_path) if not os.path.exists(img_out_path) else None
        self.save_result_img(frame_name, img_log, img_out_path)
        if output_flag == 'result-only':
            pass
        elif output_flag == 'details':
            output_detail_path = img_out_path + '_detail'
            os.mkdir(output_detail_path) if not os.path.exists(output_detail_path) else None
            self.save_details_img(frame_name, img_log, output_detail_path, keys=TickMarkReader.PROCESSES)
        else:
            raise ValueError(f'Not such output image mode: {output_flag}')

    @staticmethod
    def export_df_safe(res_df, excel_name, root_dir):
        if not root_dir or not os.path.exists(root_dir):
            root_dir = ".%s" % os.path.sep

        excel_name = os.path.join(root_dir, excel_name)
        suffix = 0
        while True:
            excel_name = excel_name[:-5] + "(" + str(suffix) + ")" + ".xlsx" if suffix else excel_name
            try:
                res_df.to_excel(excel_name)
            except PermissionError:
                suffix += 1
                continue
            break
        print(f'\nResults saved to {excel_name}')

    @staticmethod
    def save_result_img(name, img_log, output_path):
        try:
            img = img_log['results']
            cv.imwrite(os.path.sep.join([output_path, 'results_plot_' + name + '.jpg']), img)
        except KeyError:
            pass

    @staticmethod
    def save_details_img(name, img_log, output_path, keys):
        for k, v in img_log.items():
            try:
                index = keys.index(k)
                fname = os.path.sep.join([output_path, '%02d_%s_%s.jpg' % (index, k, name)])
                cv.imwrite(fname, v)
            except Exception as e:
                pass

    def update_original_frame(self, frame_name, frame):
        cv.imshow("Original", frame)

    def update_labeled_frame(self, frame_name, img_log):
        if 'results' not in img_log:
            return

        img = img_log['results']
        cv.imshow("Labeled Results", img)
        cv.waitKey(50)
