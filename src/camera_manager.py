import cv2
import threading
from queue import Queue

class CameraManager:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self._init_device()

    def _init_device(self):
        """初始化摄像头设备"""
        self.resolution = (
            self.config.get("camera_width", 1280),
            self.config.get("camera_height", 720)
        )

    def start_capture(self, camera_index=0):
        """启动摄像头捕获"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (索引: {camera_index})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.running = True
        threading.Thread(target=self._capture_thread, daemon=True).start()

    def _capture_thread(self):
        """摄像头捕获线程"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)

    def get_frame(self):
        """获取当前帧"""
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def stop(self):
        """停止捕获"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()