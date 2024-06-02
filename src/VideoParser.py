from typing import *
import cv2 as cv

from src.data_types import *


class VideoParser:
    def __init__(self):
        self.cap = None
        self.fps, self.frame_num = None, None

    def load_video(self, fp):
        self.cap = cv.VideoCapture()
        if not self.cap.open(fp):
            print(f'Error: Fail to open {fp}')
            return -1

        print(f'Read video file: {fp}')

        self.fps, self.frame_num = self.cap.get(cv.CAP_PROP_FPS), self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        print('    FPS: %d Total frames: %d' % (self.fps, int(self.frame_num)))
        return 0

    def release(self):
        self.cap.release()

    def capture_frame(self, start=0, end=-1, step=1)\
            -> Generator[Tuple[int, ColorImage], None, None]:
        if end == -1:
            end = int(self.frame_num)
        for i in range(end):
            retval, frame = self.cap.read()
            if i in range(start, end, step):
                if retval:
                    yield i + 1, frame
                else:
                    print(f'Error:    Fail to read frame: {i + 1} error code: {retval}')
                    raise StopIteration()
        self.cap.release()
