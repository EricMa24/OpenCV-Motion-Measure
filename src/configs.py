class CONFIG:
    # Ruler
    MARK_RULER_BASE = 16
    MARK_MAJOR_UNIT = 25.4
    MARK_MINOR_UNIT = 25.4 / 16

    # ROI
    ROI = (520, 440, 240, 70)  # default: x, y, width, height
    FIRST_MAJOR = 21

    #  Adaptive Threshold
    PROC_THERSH_BLOCK_SIZE = 15
    PROC_THERSH_C = 2

    #    Morph Transform
    PROC_MORPH_MAJOR_OPEN_KERNEL_RECT = (1, 100)
    PROC_MORPH_MAJOR_ERODE_KERNEL_RECT = (1, 100)
    PROC_MORPH_MINOR_OPEN_KERNEL_RECT = (1, 10)
    PROC_MORPH_MINOR_ERODE_KERNEL_RECT = (1, 10)

    #   Vertical Filter
    PROC_VERTICAL_FILTER_MAJOR_THRESHOLD = 50
    PROC_VERTICAL_FILTER_MINOR_THRESHOLD = 5

    #    CANNY EDGES DETECTION
    PROC_CANNY_MIN_VAL = 7
    PROC_CANNY_MAX_VAL = 50

    #    HOUGH LINES DETECTION
    PROC_HOUGH_MAJOR_THRESHOLD = 50
    PROC_HOUGH_MINOR_THRESHOLD = 5
    PROC_HOUGH_MAJOR_THETA_FILTER_FACTOR = 0.002
    PROC_HOUGH_MINOR_THETA_FILTER_FACTOR = 0.002

    #    TICK RECOGNIZE
    PROC_TICK_MAJOR_MAX_SPACING = 30
    PROC_TICK_MINOR_MAX_SPACING = 12
    PROC_TICK_MINOR_HIT_ZONE = 4
    PROC_TICK_START_MAJOR_TICK = 21

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
