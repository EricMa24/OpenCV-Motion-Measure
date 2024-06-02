"""
TickMarkReader, the key part of this project.
"""
# Eric MA
# y1ma9494@Gmail.com
# v1 - 2020.10.21

from typing import *
import numpy as np

import cv2 as cv

from src.configs import CONFIG
from src.data_types import *


class Tick:
    __slots__ = ['major', 'minor', 'pos']

    def __init__(self, major=0, minor=0, pos=0):
        try:
            self.major, self.minor, self.pos = int(major), int(minor), int(pos)
        except TypeError as e:
            raise e

    def __repr__(self):
        return "<Tick major=%d minor=%.1f pos=%.1f>" % (self.major, self.minor, self.pos)


class ReaderContext(object):
    __slots__ = ['roi', 'img_shape', 'img_shape_minor', 'major_mask', 'minor_mask', 'major_ticks', 'minor_ticks']
    slot_types = {
        'roi': tuple,
        'prev_ticks': object,
        'cur_ticks': object,
        'img_shape': object,
        'img_shape_minor': object,
        'major_mask': np.ndarray,
        'minor_mask': np.ndarray
    }

    def __init__(self, **kwargs):
        for key in self.__slots__:
            if key in kwargs:
                if not isinstance(kwargs[key], self.slot_types.get(key, object)):
                    raise ValueError(f"{kwargs[key]} is not an instance of {self.slot_types[key]}")
                val = kwargs[key]
            else:
                val = None
            setattr(self, key, val)

    def __str__(self):
        import textwrap
        res = f"{self.__repr__()}: \n"
        for key in self.__slots__:
            # res += "    %s: %s\n" % (key, getattr(self, key).__repr__())
            res += "    %-16s \t%s\n" % \
                   (key + ":", textwrap.shorten(getattr(self, key).__repr__(), width=40, placeholder="..."))
        return res


class TickReadException(Exception):
    def __init__(self, proc_name: str, img_log: Dict[str, ColorImage]):
        super(TickReadException, self).__init__()
        self.proc_name, self.img_log = proc_name, img_log


class ImgLogger:
    def __init__(self, keys, log):
        self.keys = keys
        self.log = log

    def __call__(self, name):
        def _decor(proc_func):
            index = self.keys.index(name) if name in self.keys else -1

            def inner(*args, **kwargs):
                image: Image = proc_func(*args, **kwargs)
                # key = "%02d_%s" % (index, name)
                self.log[name] = image
                return image

            return inner

        return _decor


class OpencvProcMixin:
    def __init__(self):
        self.ctx: ReaderContext = NotImplemented
        self.cfg: CONFIG = NotImplemented
        self.img_log: Dict[str, ColorImage] = NotImplemented

    def get_proc(self, proc_name: str) -> Callable:
        return getattr(self, 'proc_%s' % proc_name)

    def proc_pre_process(self, img: Image):
        x, y, w, h = self.ctx.roi
        img_roi = img[y:y + h, x:x + w]

        while img_roi.shape[0] < 240 or img_roi.shape[1] < 960:
            img_roi = cv.pyrUp(img_roi)

        self.ctx.img_shape = img_roi.shape
        return img_roi

    @staticmethod
    def proc_gray(img: Image):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    def proc_binary_inv(self, img: GrayImage):
        block_size, c = self.cfg.PROC_THERSH_BLOCK_SIZE, self.cfg.PROC_THERSH_C
        img_binary_inv = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                              block_size, c)
        return img_binary_inv

    def proc_major_morph_transform(self, img: GrayImage):
        open_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MAJOR_OPEN_KERNEL_RECT)
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MAJOR_ERODE_KERNEL_RECT)

        for i in range(5):
            img = cv.erode(img, erode_kernel, iterations=1)
            img = cv.dilate(img, open_kernel, iterations=1)
        # img = cv.erode(img, erode_kernel, iterations=1)
        return img

    def proc_major_vertical_filter(self, img: GrayImage):
        threshold = self.cfg.PROC_VERTICAL_FILTER_MAJOR_THRESHOLD
        img = img.copy()
        for i in range(0, img.shape[1]):
            count = img[:, i].sum() // 255  # count of white pixel
            if count >= threshold:
                img[:, i] = 255
            else:
                img[:, i] = 0

        self.ctx.major_mask = (img[0, :] == 255)
        return img

    def proc_major_canny_edges(self, img: Image):
        min_val, max_val = self.cfg.PROC_CANNY_MIN_VAL, self.cfg.PROC_CANNY_MAX_VAL
        img_major_canny_edges = cv.Canny(img, min_val, max_val, apertureSize=3)
        return img_major_canny_edges

    def proc_major_hough_line_filter(self, img: Image):
        threshold = self.cfg.PROC_HOUGH_MAJOR_THRESHOLD
        lines = cv.HoughLines(img, 1, np.pi / 180, threshold)
        if lines is None or len(lines) < 2:
            raise TickReadException('major_hough_line_filter', self.img_log)

        # filter vertical lines, line[0][1] is the theta
        lines = [line for line in lines if abs(line[0][1]) < np.pi * self.cfg.PROC_HOUGH_MAJOR_THETA_FILTER_FACTOR]
        # covert lines from polar to cartesian
        lines = self._convert_cartesian(lines, self.ctx.img_shape)
        # plot
        bg_img = self.img_log['pre_process'].copy()
        img_major_hough_line_filtered = self._plot_line(lines, bg_img)

        self.ctx.major_lines = lines
        return img_major_hough_line_filtered

    def proc_minor_morph_transform(self, _: Image):
        # Use binary_inv Image
        img = self.img_log['binary_inv']

        # take half roi-img for minor ticks---------------------------------
        height, width, color, = self.ctx.img_shape
        img_half = img[0: height // 2, :]
        self.ctx.img_shape_minor = (height // 2, width, color)
        # ------------------------------------------------------------------

        open_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MINOR_OPEN_KERNEL_RECT)
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.cfg.PROC_MORPH_MINOR_ERODE_KERNEL_RECT)

        for i in range(10):
            img_half = cv.erode(img_half, erode_kernel, iterations=1)
            img_half = cv.dilate(img_half, open_kernel, iterations=1)
        # img_half = cv.erode(img_half, erode_kernel, iterations=2)
        return img_half

    def proc_minor_vertical_filter(self, img: GrayImage):
        threshold = self.cfg.PROC_VERTICAL_FILTER_MINOR_THRESHOLD
        img = img.copy()
        for i in range(0, img.shape[1]):
            count = img[:, i].sum() // 255  # count of white pixel
            if count >= threshold:
                img[:, i] = 255  # mark white
            else:
                img[:, i] = 0  # mark black
        self.ctx.minor_mask = (img[0, :] == 255)
        return img

    def proc_minor_canny_edges(self, img: Image):
        min_val, max_val = self.cfg.PROC_CANNY_MIN_VAL, self.cfg.PROC_CANNY_MAX_VAL

        img_minor_canny_edges = cv.Canny(img, min_val, max_val, apertureSize=3)
        return img_minor_canny_edges

    def proc_minor_hough_line_filter(self, img: Image):
        threshold = self.cfg.PROC_HOUGH_MINOR_THRESHOLD

        lines = cv.HoughLines(img, 1, np.pi / 360, threshold)
        if lines is None:
            raise TickReadException('minor_hough_line_filter', self.img_log)
        lines = [line for line in lines if abs(line[0][1]) < np.pi * self.cfg.PROC_HOUGH_MINOR_THETA_FILTER_FACTOR]
        lines = self._convert_cartesian(lines, self.ctx.img_shape_minor)

        bg_img = self.img_log['pre_process'].copy()
        img_minor_hough_line_filtered = self._plot_line(lines, bg_img)

        self.ctx.minor_lines = lines
        return img_minor_hough_line_filtered

    def proc_major_tick_recog(self, _: Image) -> ColorImage:
        """Recognize major tick"""
        bg_img = self.img_log['pre_process'].copy()
        mask = self.ctx.major_mask
        groups = self.group_by_distance(mask, threshold=self.cfg.PROC_VERTICAL_FILTER_MAJOR_THRESHOLD)

        # take each group's average pixel x-position as tick's position
        major_ticks = [Tick(major=0, minor=0, pos=int(np.mean(group)))
                       for group in groups]

        if len(major_ticks) < 2:
            raise TickReadException('major_tick_recog', self.img_log)

        self.ctx.major_ticks = major_ticks

        # Draw Ticks on bkg image
        y1, y2 = 0, self.ctx.img_shape[0]-1
        lines = [(tick.pos, y1, tick.pos, y2) for tick in major_ticks]
        img = self._plot_line(lines, bg_img, line_bgr=(0, 0, 255), line_thickness=3)
        return img

    def proc_minor_tick_recog(self, _: Image) -> ColorImage:
        """Recognize minor tick"""
        bg_img = self.img_log['pre_process'].copy()
        mask = self.ctx.minor_mask

        groups = self.group_by_distance(mask, threshold=self.cfg.PROC_VERTICAL_FILTER_MINOR_THRESHOLD)

        # take each group's average pixel x-position as tick's position
        minor_ticks = [Tick(major=0, minor=0, pos=int(np.mean(group)))
                       for group in groups]

        self.ctx.minor_ticks = minor_ticks

        # Draw Ticks on bkg image
        y1, y2 = 0, self.ctx.img_shape[0] - 1
        lines = [(int(tick.pos), y1, int(tick.pos), y2) for tick in minor_ticks]
        img = self._plot_line(lines, bg_img, line_bgr=(0, 0, 255), line_thickness=1)

        return img

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

    @staticmethod
    def group_by_distance(mask, threshold):
        # group masks by distance
        cur_group: List[int] = []
        groups: List[type(cur_group)] = []
        for el in filter(lambda el: el[1], enumerate(mask)):
            pos = el[0]
            if not cur_group:
                cur_group.append(pos)
            else:
                if pos - cur_group[-1] <= threshold:  # same group if two point is within distance threshold
                    cur_group.append(pos)
                else:  # next group
                    groups.append(cur_group)
                    cur_group = [pos, ]
        groups.append(cur_group)  # last group
        return groups


class TickMarkReader(OpencvProcMixin):
    PROCESSES = ['pre_process', 'gray', 'binary_inv',
                 'major_morph_transform', 'major_vertical_filter', 'minor_morph_transform', 'minor_vertical_filter',
                 'major_tick_recog', 'minor_tick_recog']

    def __init__(self, cfg):
        super(TickMarkReader, self).__init__()
        self.ctx = ReaderContext()
        self.cfg: CONFIG = cfg
        self.img_log: Dict[str, ColorImage] = dict()

    def run(self, name, image: ColorImage, last_majors=None, target_factor=0.5) \
            -> Tuple[dict, Dict[str, Image], List[Tick]]:
        """
        Take an input image and ROI, recognize tick marks of ruler tape within ROI,
        and return the reading where the midline of ROI is.
        """
        try:
            res, img_log, majors = self._run(name, image, last_majors, target_factor)
        except TickReadException as e:
            res = {'name': name,
                   'reading_major': 'NaN',
                   'reading_minor': 'NaN',
                   'description': f'Error at {e.proc_name}'}
            img_log = e.img_log
            majors = last_majors

        return res, img_log, majors

    def _run(self, name, image, last_majors, target_factor) -> Tuple[dict, Dict[str, Image], List[Tick]]:
        self.ctx.roi = self.cfg.ROI  # !!!
        img_logger = ImgLogger(keys=self.PROCESSES, log=self.img_log)

        # Run Processes
        for proc_name in self.PROCESSES:
            process = img_logger(proc_name)(self.get_proc(proc_name))
            image = process(image)

        # Track Major Ticks to handle fading/emerging major ticks
        self.ctx.major_ticks, self.ctx.minor_ticks = self.track_major_ticks(last_majors)

        # Ruler Readings, by default read mid-line reading
        reading_major, reading_minor, description = self.get_ruler_readings(target_factor=target_factor)

        # Format return res
        res = {'name': name,
               'reading_major': reading_major,
               'reading_minor': reading_minor,
               'description': description}

        # Plot results
        self.img_log['results'] = self.results_plot(self.ctx.major_ticks, self.ctx.minor_ticks, name, res)

        return res, self.img_log, self.ctx.major_ticks

    def track_major_ticks(self, last_majors) -> (List[Tick], List[Tick]):
        # Offset major ticks
        if last_majors is None:
            major_ticks = [Tick(major=ori.major+self.cfg.FIRST_MAJOR,
                                minor=ori.minor,
                                pos=ori.pos)
                           for ori in self.ctx.major_ticks]
        else:
            cm, pm = self.ctx.major_ticks, last_majors  # current majors, previous majors

            # Create extended ruler with pseudo last tick and pseudo next tick
            spacing = pm[1].pos - pm[0].pos  # estimate spacing between two major ticks
            last, next_ = Tick(pm[0].major-1, 0, pm[0].pos-spacing), Tick(pm[-1].major+1, 0, pm[-1].pos+spacing)
            pm_extend = [last, ] + pm + [next_, ]

            major_ticks = []
            for tick in cm:
                matches = list(filter(lambda pm_tick: abs(tick.pos - pm_tick.pos) < spacing / 2, pm_extend))
                if len(matches) != 1:
                    raise TickReadException('track_major_ticks', self.img_log)
                major_ticks.append(Tick(matches[0].major, 0, tick.pos))

        # assign minor ticks for each major
        minor_ticks = self.ctx.minor_ticks

        #   filter out major ticks from minors
        threshold = self.cfg.PROC_VERTICAL_FILTER_MINOR_THRESHOLD
        for major in major_ticks:
            minor_ticks = list(filter(lambda tick: abs(tick.pos-major.pos) > threshold, minor_ticks))

        sep_points = [0, ] + [tick.pos for tick in major_ticks] + [self.ctx.img_shape[1], ]
        first_major = major_ticks[0].major

        major_range_dict = {}  # dict to get major value by minor tick's pos
        for i in range(0, len(sep_points)-1):
            start, end = sep_points[i], sep_points[i+1]
            major_range_dict.update(dict([(k, first_major + i - 1) for k in range(start, end)]))  # open range

        cur_major, cur_minor = major_range_dict[0], 1
        for tick in minor_ticks:  # default minor_ticks[0] = Tick(0, 0, pos)
            tick.major = major_range_dict[tick.pos]  # assign major value
            if tick.major == cur_major:
                tick.minor = cur_minor  # assign minor value
                cur_minor = cur_minor + 1
            else:
                tick.minor = 1
                cur_major, cur_minor = tick.major, 1

        return major_ticks, minor_ticks

    def get_ruler_readings(self, target_factor=0.5) -> (float, float, str):
        #  For strict reading, only consider a valid reading when all minor ticks (number of ruler_base)
        #  were successfully recognized, otherwise raise Exception
        ruler_base = self.cfg.MARK_RULER_BASE
        target = int(self.ctx.img_shape[1] * target_factor)

        # major reading
        left_majors = list(filter(lambda tick: tick.pos <= target, self.ctx.major_ticks))
        if len(left_majors):
            major_res = left_majors[-1].major
        else:
            major_res = self.ctx.major_ticks[0].major - 1
            return major_res, np.nan, "Incomplete minor ticks"

        # minor reading
        # check if num of minors equal to ruler base
        relate_ticks = list(filter(lambda tick: tick.major == major_res, self.ctx.minor_ticks))
        if not len(relate_ticks) == ruler_base - 1:
            return major_res, np.nan, "Incomplete minor ticks"
        try:
            last_major = list(filter(lambda tick: (tick.major == major_res), self.ctx.major_ticks))[0]
            next_major = list(filter(lambda tick: (tick.major == major_res+1), self.ctx.major_ticks))[0]
        except IndexError:
            return major_res, np.nan, "Incomplete minor ticks"
        relate_ticks = [last_major, ] + relate_ticks + [next_major, ]

        # Interpolation of minor reading
        xs = [tick.pos for tick in relate_ticks]
        ys = list(range(len(relate_ticks)))
        minor_res = np.round(np.interp(target, xs, ys), 1)

        return major_res, minor_res, "Success"

    def results_plot(self, major_ticks, minor_ticks, name, res):
        # used pre_process img instead of argument img

        # Define runtime_variables
        bg_img = self.img_log['pre_process'].copy()
        height, width, color, = self.ctx.img_shape

        # Text alignment
        row1, row2, row3, row4 = [height // 10 * i for i in range(1, 5)]
        h_offset = self.cfg.PROC_TICK_MINOR_MAX_SPACING // 2

        # plot major ticks
        for major_tick in major_ticks:
            cv.line(bg_img, (major_tick.pos, 0), (major_tick.pos, height), (0, 0, 255), thickness=3)
            text = '#' + str(major_tick.major)
            cv.putText(bg_img, text, (major_tick.pos + h_offset, row1), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # plot minor ticks
        for minor_tick in minor_ticks:
            if minor_tick.minor == 0:
                continue
            cv.line(bg_img, (minor_tick.pos, 0), (minor_tick.pos, height), (0, 0, 255), thickness=1)

        # plot mid line
        cv.line(bg_img, (width // 2, 0), (width // 2, height), (255, 0, 0), thickness=3)

        # plot labels
        text = name
        cv.putText(bg_img, text, (width // 2 + h_offset, row2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        text = 'Reading: major ' + str(res['reading_major']) + ' minor ' + str(res['reading_minor'])
        cv.putText(bg_img, text, (width // 2 + h_offset, row3), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        text = res['description']
        cv.putText(bg_img, text, (width // 2 + h_offset, row4), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        return bg_img

