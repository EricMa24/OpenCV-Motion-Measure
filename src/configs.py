class CONFIG:
    # Ruler
    MARK_RULER_BASE = 16
    MARK_MAJOR_UNIT = 25.4
    MARK_MINOR_UNIT = 25.4 / 16

    # IMAGE PROCESSING
    ROI_RANGE = (520, 440, 240, 70)  # x, y, width, height
    # ROI_RANGE = (330, 470, 110, 340)

    #  Adaptive Threshold
    PROC_THERSH_BLOCK_SIZE = 15
    PROC_THERSH_C = 2

    #    Morph Transform
    PROC_MORPH_MAJOR_OPEN_KERNEL_RECT = (1, 100)
    PROC_MORPH_MAJOR_ERODE_KERNEL_RECT = (3, 3)
    PROC_MORPH_MINOR_OPEN_KERNEL_RECT = (3, 15)
    PROC_MORPH_MINOR_ERODE_KERNEL_RECT = (1, 3)

    #    CANNY EDGES DETECTION
    PROC_CANNY_MIN_VAL = 5
    PROC_CANNY_MAX_VAL = 50

    #    HOUGH LINES DETECTION
    PROC_HOUGH_MAJOR_THRESHOLD = 50
    PROC_HOUGH_MINOR_THRESHOLD = 15
    PROC_HOUGH_MAJOR_THETA_FILTER_FACTOR = 0.003
    PROC_HOUGH_MINOR_THETA_FILTER_FACTOR = 0.003

    #    TICK RECOGNIZE
    PROC_TICK_MAJOR_MAX_SPACING = 30
    PROC_TICK_MINOR_MAX_SPACING = 12
    PROC_TICK_MINOR_HIT_ZONE = 4
    PROC_TICK_START_MAJOR_TICK = 21