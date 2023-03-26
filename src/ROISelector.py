import cv2 as cv


class ROISelector:
    def __init__(self):
        self.img = None
        pass

    def __call__(self, img, **kwargs):
        self.img = img
        self.height, self.width, self.color = img.shape
        x, y, w, h = (520, 440, 240, 70)  # default
        # main loop
        cv.namedWindow('ROI Selector')
        while True:
            k = cv.waitKey(10) & 0xff
            # events
            stepx, stepy = 2, 2
            if k == ord('w'):
                y -= stepy
            if k == ord('s'):
                y += stepy
            if k == ord('a'):
                x -= stepx
            if k == ord('d'):
                x += stepx
            if k == ord('W'):
                h -= stepy
            if k == ord('S'):
                h += stepy
            if k == ord('A'):
                w -= stepx
            if k == ord('D'):
                w += stepx
            if k == ord('r') or k == ord('R'):
                x, y, w, h = (520, 440, 240, 70)
            if k == 27 or k == ord('q') or k == ord('Q'):  # ESC
                break

            img = self.update_roi(x, y, w, h)
            cv.imshow('Frame 0', img)
        cv.destroyAllWindows()

    def update_roi(self, x, y, w, h):
        img = self.img.copy()
        self._display_instruction(img)
        self._display_roi(img, x, y, w, h)
        cv.line(img, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), thickness=1)
        cv.line(img, (0, self.height // 2), (self.width, self.height // 2), (255, 0, 0), thickness=1)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
        return img

    @staticmethod
    def _display_roi(img, x, y, w, h):
        text = f"ROI: {x} {y} {w} {h}"
        cv.putText(img, text, (x-30, y-30), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

    @staticmethod
    def _display_instruction(img):
        texts = ['Move ROI: W, S, A, D',
                 'Zoom ROI: Shift + [W, S, A, D]',
                 'Reset: r',
                 'Quit: Esc, q']
        y = 60
        for text in texts:
            cv.putText(img, text, (20, y), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
            y += 30


if __name__ == '__main__':
    video_file = r"E:\OneDrive - Louisiana State University\000\201005_LWT_openCV\22_data\raw_video\230207_GOPR0539.MP4"
    cap = cv.VideoCapture()
    cap.open(video_file)
    retval, frame = cap.read()

    ROISelector()(frame)
