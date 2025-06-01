import cv2
import numpy as np
from uvc_core.plugin_base import DetectionPlugin

class BlackScreenDetector(DetectionPlugin):
    def __init__(self, config):
        super().__init__(config)
        self.threshold = {
            'variance': config.get('variance_threshold', 50),
            'histogram': config.get('histogram_threshold', 0.95),
            'edges': config.get('edge_threshold', 50),
            'brightness': config.get('brightness_threshold', 20)
        }

    def initialize(self):
        """初始化检测所需资源"""
        # 可以在这里初始化任何需要的资源
        pass

    def process_frame(self, frame):
        print("black frame check\n")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 多算法检测
        variance = np.var(gray)
        
        # 直方图分析（黑屏像素占比）
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        black_ratio = np.sum(hist[:10])/gray.size
        
        # 边缘检测（Canny边缘数量）
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.count_nonzero(edges)
        
        # 亮度均值检测
        mean_brightness = np.mean(gray)
        
        return {
            'is_black': (
                (variance < self.threshold['variance']) |
                (black_ratio > self.threshold['histogram']) |
                (mean_brightness < self.threshold['brightness'])
            ) & (edge_pixels < self.threshold['edges']),
            'variance': variance,
            'black_ratio': black_ratio,
            'edge_pixels': edge_pixels,
            'mean_brightness': mean_brightness
        }

    def shutdown(self):
        pass