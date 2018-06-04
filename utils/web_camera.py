import cv2
from threading import Thread


class VideoStream:
    def __init__(self, width=180, height=135, camera_num=0):
        # Initialize the video camera stream
        self.stream = cv2.VideoCapture(camera_num)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, self.frame = self.stream.read()
        self.RUN = True

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if not self.RUN:
                return
            (self.ret, self.frame) = self.stream.read()

    def get_img(self):
        return self.frame

    def finish(self):
        self.RUN = not self.RUN

    def stop(self):
        self.stream.release()

