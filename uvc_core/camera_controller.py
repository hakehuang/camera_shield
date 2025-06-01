import cv2
from typing import Optional, Any

class UVCCamera:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def show_frame(self, frame):
        cv2.imshow(f'Camera Preview (Device {self.device_id})', frame)
        cv2.waitKey(1)

    def auto_focus(self, enable: bool):
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enable else 0)

    def auto_white_balance(self, enable: bool):
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if enable else 0)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
