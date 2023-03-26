"""
TickMarkReader, the key part of this project.
"""
# Eric MA
# y1ma9494@Gmail.com
# v1 - 2020.10.21

from typing import *
import numpy as np
from collections import namedtuple

import cv2.cv2 as cv

from .configs import CONFIG
from .RulerParser import ColorImage, ROI, Tick


class TickReadException(Exception):
    def __init__(self, proc_name: str, img_log: Dict[str, ColorImage]):
        super(TickReadException, self).__init__()
        self.proc_name, self.img_log = proc_name, img_log


RUNTIME_VARS_KEY = ['roi', 'prev_ticks', 'cur_ticks', 'img_shape', 'img_shape_minor',
                    'major_lines', 'minor_lines']
RuntimeVarsNT = namedtuple('runtime_vars', field_names=RUNTIME_VARS_KEY, defaults=[None, ] * len(RUNTIME_VARS_KEY))


class ImgLogger:
    def __init__(self):
        self.log_count = 0
        
    def __call__(self, name, log):
        def _decor(proc_func):
            def inner(*args, **kwargs):
                ret = proc_func(args, kwargs)
                log.update({str(self.log_count) + '_' + name: ret})
                self.log_count += 1
                return ret
            return inner
        return _decor


class OpencvProcBase:
    def __init__(self, rvs: RuntimeVarsNT, cfg, img_log: Dict[str, ColorImage]):
        self.rvs = rvs
        self.cfg = cfg
        self.img_log = img_log

    def get_proc(self, proc_name: str):
        return getattr(self, 'proc_' + proc_name)

    def proc_pre_process(self, img: ColorImage):
        x, y, w, h = self.rvs.roi
        img_roi = img[y:y + h, x:x + w]

        img_roi = cv.pyrUp(img_roi)
        img_roi = cv.pyrUp(img_roi)

        self.rvs.img_shape = img_roi.shape
        self.rvs.img_roi = img_roi
        return img_roi

    @staticmethod
    def proc_gray(img: ColorImage):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    def proc_binary_inv(self, img: ColorImage):
        block_size, c = self.cfg.PROC_THERSH_BLOCK_SIZE, self.cfg.PROC_THERSH_C
        img_binary_inv = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                              block_size, c)
        return img_binary_inv

    def proc_major_morph_transform(self, img: ColorImage):
        open_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MAJOR_OPEN_KERNEL_RECT)
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MAJOR_ERODE_KERNEL_RECT)

        erode = cv.erode(img, open_kernel, iterations=1)
        dilate = cv.dilate(erode, open_kernel, iterations=1)
        erode = cv.erode(dilate, open_kernel, iterations=1)
        dilate = cv.dilate(erode, open_kernel, iterations=1)
        erode = cv.erode(dilate, erode_kernel, iterations=1)
        return erode

    def proc_major_canny_edges(self, img: ColorImage):
        min_val, max_val = self.cfg.PROC_CANNY_MIN_VAL, self.cfg.PROC_CANNY_MAX_VAL
        img_major_canny_edges = cv.Canny(img, min_val, max_val, apertureSize=3)
        return img_major_canny_edges

    def proc_major_hough_line_filter(self, img: ColorImage):
        threshold = self.cfg.PROC_HOUGH_MAJOR_THRESHOLD
        lines = cv.HoughLines(img, 1, np.pi / 180, threshold)
        if lines is None or len(lines) < 2:
            raise TickReadException('major_hough_line_filter', self.img_log)

        # filter vertical lines, line[0][1] is the theta
        lines = [line for line in lines if abs(line[0][1]) < np.pi * self.cfg.PROC_HOUGH_MAJOR_THETA_FILTER_FACTOR]
        # covert lines from polar to cartesian
        lines = self._convert_cartesian(lines, self.rvs.img_shape)
        # plot
        img_major_hough_line_filtered = self._plot_line(lines, self.rvs.roi.copy())

        self.rvs.major_lines = lines
        return img_major_hough_line_filtered

    def proc_minor_morph_transform(self, img: ColorImage):
        # take half roi-img for minor ticks---------------------------------
        height, width, color, = self.rvs.img_shape
        img_binary_inv = img[0: height // 2, :]
        self.rvs.img_shape_minor = (height // 2, width, color)
        # ------------------------------------------------------------------
        open_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MINOR_OPEN_KERNEL_RECT)
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MINOR_ERODE_KERNEL_RECT)

        erode = cv.erode(img_binary_inv, open_kernel, iterations=1)
        dilate = cv.dilate(erode, open_kernel, iterations=1)
        erode = cv.erode(dilate, erode_kernel, iterations=1)
        return erode

    def proc_minor_canny_edges(self, img: ColorImage):
        min_val, max_val = self.cfg.PROC_CANNY_MIN_VAL, self.cfg.PROC_CANNY_MAX_VAL

        img_minor_canny_edges = cv.Canny(img, min_val, max_val, apertureSize=3)
        return img_minor_canny_edges

    def proc_minor_hough_line_filter(self, img: ColorImage):
        threshold = self.cfg.PROC_HOUGH_MINOR_THRESHOLD

        lines = cv.HoughLines(img, 1, np.pi / 180, threshold)
        if lines is None:
            raise TickReadException('minor_hough_line_filter', self.img_log)
        lines = [line for line in lines if abs(line[0][1]) < np.pi * self.cfg.PROC_HOUGH_MINOR_THETA_FILTER_FACTOR]
        lines = self._convert_cartesian(lines, self.rvs.img_shape_minor)
        img_minor_hough_line_filtered = self._plot_line(lines, self.rvs.roi.copy())

        self.rvs.minor_lines = lines
        return img_minor_hough_line_filtered

    @staticmethod
    def _convert_cartesian(lines, img_shape):
        """
        Convert lines in polar coordinate system to lines in Cartesian coordinate system
        """
        res = list()
        height, width = img_shape[0], img_shape[1]

        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, x2 = int(x0 + width * (-b)), int(x0 - width * (-b))
            y1, y2 = int(y0 + height * a), int(y0 - height * a)
            res.append((x1, y1, x2, y2))

        return res

    @staticmethod
    def _plot_line(lines, bg_img, line_bgr=(0, 0, 255), line_thickness=1):
        for line in lines:
            x1, y1, x2, y2 = line
            cv.line(bg_img, (x1, y1), (x2, y2), line_bgr, line_thickness)

        return bg_img


# noinspection PyArgumentList
class TickMarkReader(OpencvProcBase):
    PRE_PROCESS_SERIES = ['pre_process', 'gray', 'binary_inv']
    MAJOR_TICK_PROCESS_SERIES = ['major_morph_transform', 'major_canny_edges', 'major_hough_line_filter']
    MINOR_TICK_PROCESS_SERIES = ['minor_morph_transform', 'minor_canny_edges', 'minor_hough_line_filter']
    TICK_RECOG_ANALYSIS_SERIES = ['major_tick_recog', 'minor_tick_recog', 'results_plot']

    def __init__(self):
        super(TickMarkReader, self).__init__(rvs=RuntimeVarsNT(), cfg=CONFIG, img_log=dict())

    def run(self, name, image: ColorImage, roi: ROI, prev_ticks=None, match_line=0.5) \
            -> Tuple[dict, Dict[str, ColorImage], List[Tick]]:
        """
        Take an input image and ROI, recognize tick marks of ruler tape within ROI,
        and return the reading where the midline of ROI is.
        """
        try:
            res, img_log, major_ticks = self._run(name, image, roi, prev_ticks, match_line)
        except TickReadException as e:
            res = {'name': name, 'reading_major': 'NaN', 'reading_minor': 'NaN',
                   'description': f'Error at {e.proc_name}'}
            img_log = e.img_log
            major_ticks = prev_ticks
            
        return res, img_log, major_ticks

    def _run(self, name, image, roi, prev_ticks, match_line) -> Tuple[dict, Dict[str, ColorImage], List[Tick]]:
        self.rvs = RuntimeVarsNT()  # reset
        self.rvs.roi = roi
        self.rvs.prev_ticks = prev_ticks
        self.img_log = dict()
        res = dict()
    
        img_logger = ImgLogger()
        # pre-process
        for proc_name in self.PRE_PROCESS_SERIES:
            process = img_logger(proc_name, self.img_log)(self.get_proc(proc_name))
            image = process(image)
        binary_inv = image.copy()  # reserve for minor image process

        # major tick image process
        for proc_name in self.MAJOR_TICK_PROCESS_SERIES:
            process = img_logger(proc_name, self.img_log)(self.get_proc(proc_name))
            image = process(image)

        # minor tick image process
        image = binary_inv
        for proc_name in self.MINOR_TICK_PROCESS_SERIES:
            process = img_logger(proc_name, self.img_log)(self.get_proc(proc_name))
            image = process(image)

        # tick recognition and tracking
        major_ticks = self.major_tick_recog(self.rvs.major_lines)
        minor_ticks, updated_major_ticks = self.minor_tick_recog(self.rvs.minor_lines, major_ticks)

        match_line_loc = int(self.rvs.img_shape[1] * match_line)
        reading_major, reading_minor, description = self._read_reading(
            match_line_loc, updated_major_ticks.copy(), minor_ticks.copy(), self.cfg.PROC_TICK_MINOR_HIT_ZONE)
        res['reading_major'] = reading_major
        res['reading_minor'] = reading_minor
        res['description'] = description
        return res, self.img_log, major_ticks

    def major_tick_recog(self, major_lines) -> List[Tick]:
        """Read major tick and track fading/emerging major ticks based on previous ticks"""
        max_spacing = self.cfg.PROC_TICK_MAJOR_MAX_SPACING
        start_tick = self.cfg.PROC_TICK_START_MAJOR_TICK

        # a. RAW READING
        x_locations = [(line[0] + line[2]) // 2 for line in major_lines]  # x1, y1, x2, y2 = line
        tick_groups = self._group_values(x_locations, max_spacing)  # group edge lines within threshold,
        # and consider each group is the edge lines of single tick mark, where the average x is the location of tick
        major_locations = [int(round(np.average(x))) for x in tick_groups]
        major_locations.sort()
        raw_major_ticks: List[Tick] = [(i, loc) for i, loc in enumerate(major_locations)]  # sort and numbering ticks

        if (len(raw_major_ticks) < 2) or (len(raw_major_ticks) > 4):
            raise TickReadException('major_tick_recog', self.img_log)

        # b. TRACK READING BY prev_ticks
        rm, pm = raw_major_ticks, self.rvs.prev_ticks
        if pm is None:
            major_ticks = [(i + start_tick, loc) for i, loc in rm]
            return major_ticks

        # Create extended ruler with pseudo last tick and pseudo next tick
        spacing = pm[1][1] - pm[0][1]  # estimate spacing between two major ticks
        last, next_ = (pm[0][0] - 1, pm[0][1] - spacing), (pm[-1][0] + 1, pm[-1][1] + spacing)
        pm1 = [last, ] + pm + [next_, ]
        # Match extended ruler
        major_ticks = []
        for _, loc in rm:
            matches = list(filter(lambda _, p_loc: abs(loc - p_loc) < spacing / 2, pm1))
            if len(matches) != 1:
                raise TickReadException('major_tick_recog', self.img_log)
            major_ticks.append((matches[0][0], loc))

        return major_ticks

    def minor_tick_recog(self, minor_lines, major_ticks):
        max_spacing = self.cfg.PROC_TICK_MINOR_MAX_SPACING
        ruler_base = self.cfg.MARK_RULER_BASE

        x_locations = [(line[0] + line[2]) // 2 for line in minor_lines]  # x1, y1, x2, y2 = line

        tick_groups = self._group_values(x_locations, max_spacing)
        tick_marks = [int(round(np.average(x))) for x in tick_groups]
        tick_marks.sort()

        # delete major ticks
        tick_marks = list(filter(lambda loc: any([abs(loc-m_loc) <= max_spacing for _, m_loc in major_ticks]),
                                 tick_marks))

        minor_ticks, updated_major_ticks = self._plot_minor_ticks(tick_marks, major_ticks.copy(), ruler_base)

        return minor_ticks, updated_major_ticks

    def results_plot(self, _):
        # used pre_process img instead of argument img

        # Define runtime_variables
        bg_img = self.img_log['pre_process'].copy()
        major_ticks, minor_ticks = self.rvs.major_ticks, self.rvs.minor_ticks
        height, width, color, = self.rvs.img_shape

        # Text alignment
        row1, row2, row3, row4 = [height // 10 * i for i in range(1, 5)]
        h_offset = self.cfg.PROC_TICK_MINOR_MAX_SPACING // 2

        # plot major ticks
        for major_tick in major_ticks:
            cv.line(bg_img, (major_tick[1], 0), (major_tick[1], height), (0, 0, 255), thickness=3)
            text = '#' + str(major_tick[0])
            cv.putText(bg_img, text, (major_tick[1] + h_offset, row1), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # plot minor ticks
        for minor_tick in minor_ticks:
            cv.line(bg_img, (minor_tick[2], 0), (minor_tick[2], height), (0, 0, 255), thickness=1)

        # plot mid line
        cv.line(bg_img, (width // 2, 0), (width // 2, height), (255, 0, 0), thickness=3)

        # plot labels
        text = self.name
        cv.putText(bg_img, text, (width // 2 + h_offset, row2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        text = 'Reading: major ' + str(self.res['reading_major']) + ' minor ' + str(self.res['reading_minor'])
        cv.putText(bg_img, text, (width // 2 + h_offset, row3), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        text = self.res['description']
        cv.putText(bg_img, text, (width // 2 + h_offset, row4), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        return bg_img

    @staticmethod
    def _group_values(values: List[int], max_spacing: int) -> List[List[int]]:
        res = []

        values.sort(reverse=True)
        while values:
            cur_value = values.pop(0)

            cur_value_group = [cur_value, ]
            while values:
                if abs(cur_value - values[0]) <= max_spacing:
                    cur_value = values.pop(0)
                    cur_value_group.append(cur_value)
                else:
                    break
            res.append(cur_value_group)

        return res

    @staticmethod
    def _plot_major_ticks(tick_marks, previous_major_ticks, width, start_idx):
        # res = []
        # pm = previous_major_ticks
        #
        # if not previous_major_ticks:
        #     for i, value in enumerate(tick_marks):
        #         res.append((i + start_idx, value))
        #     return res, res
        #
        # # first check
        # pm1 = pm.copy()
        #
        # for value in tick_marks:
        #     min_val, min_idx = width, 0
        #     for i, tick in enumerate(pm1):
        #         cur_distance = abs(value - tick[1])
        #         if cur_distance < min_val:
        #             min_val, min_idx = cur_distance, i
        #     res.append([pm1[min_idx][0], value])
        #
        # # second check
        # if len(res) == len(pm):
        #     return res, res
        # elif len(res) < len(pm):  #
        #     res_had_indexes = []
        #     for value in res:
        #         value = value[1]
        #         min_val, min_idx = width, 0
        #         for i, tick in enumerate(pm):
        #             cur_distance = abs(value - tick[1])
        #             if cur_distance < min_val:
        #                 min_val, min_idx = cur_distance, i
        #         res_had_indexes.append(min_idx)
        #         if 0 in res_had_indexes and (len(pm) - 1 in res_had_indexes):
        #             return res, pm
        #         else:
        #             return res, res
        # else:
        #     return res, res
        pass

    @staticmethod
    def _plot_minor_ticks(minor_ticks, major_ticks, ruler_base):
        res = []

        minor_ticks.sort()
        major_ticks.sort()

        left_ticks = []
        while len(minor_ticks) and (minor_ticks[0] < major_ticks[0][1]):
            left_ticks.append(minor_ticks.pop(0))
        left_ticks.sort(reverse=True)
        for i, tick in enumerate(left_ticks):
            res.append((major_ticks[0][0] - 1, ruler_base - 1 - i, tick))

        for i in range(len(major_ticks) - 1):
            cur_major, next_major = major_ticks[i], major_ticks[i + 1]
            count = 0
            while minor_ticks and (minor_ticks[0] > cur_major[1]) and (minor_ticks[0] < next_major[1]):
                count += 1
                res.append((cur_major[0], count, minor_ticks.pop(0)))
            else:
                if count != ruler_base - 1:
                    major_ticks[i] = major_ticks[i] + ['incomplete', ]

        last_major = major_ticks[-1]
        for i, tick in enumerate(minor_ticks):
            res.append((last_major[0], i + 1, tick))

        return res, major_ticks

    @staticmethod
    def _read_reading(mid_line, major_ticks, minor_ticks, hit_zone):
        def find_left(target, vals):
            for i, val in enumerate(vals):
                if val - target > 0:
                    return i - 1
            return len(vals) - 1

        def find_right(target, vals):
            for i, val in enumerate(vals):
                if target - val < 0:
                    return i
            return 0

        def interpolation(x, x_range, y_range):
            return y_range[0] + (x - x_range[0]) * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])

        major_vals = [val[1] for val in major_ticks]
        minor_vals = [val[2] for val in minor_ticks]

        reading_major = major_ticks[find_left(mid_line, major_vals)]
        reading_minor_left = minor_ticks[find_left(mid_line, minor_vals)]
        reading_minor_right = minor_ticks[find_right(mid_line, minor_vals)]

        if reading_minor_left[1] > reading_minor_right[1]:
            if mid_line - reading_major[1] < reading_minor_right[2] - reading_minor_left[2]:
                reading_minor_left = (reading_major[0], 0, reading_major[1])
            else:
                reading_minor_right = (major_ticks[find_right(mid_line, major_vals)][0], 16,
                                       major_ticks[find_right(mid_line, major_vals)][1])

        if 'incomplete' in reading_major:
            reading_minor = -1
            description = 'Incomplete minor ticks'
        elif abs(mid_line - reading_minor_left[2]) <= hit_zone:
            reading_minor = reading_minor_left[1]
            description = 'Hit line'
        elif abs(mid_line - reading_minor_right[2]) <= hit_zone:
            reading_minor = reading_minor_right[1]
            description = 'Hit line'
        else:
            reading_minor = np.round(interpolation(mid_line, (reading_minor_left[2], reading_minor_right[2]),
                                                   (reading_minor_left[1], reading_minor_right[1])), 1)
            description = 'Interpolation'

        return reading_major[0], reading_minor, description
