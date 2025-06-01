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
        
        # 在帧上显示检测结果
        is_black = (
            (variance < self.threshold['variance']) |
            (black_ratio > self.threshold['histogram']) |
            (mean_brightness < self.threshold['brightness'])
        ) & (edge_pixels < self.threshold['edges'])
        
        # 显示检测状态
        status_text = "Black Screen Detected" if is_black else "Normal"
        color = (0, 0, 255) if is_black else (0, 255, 0)
        text_width = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(frame, status_text, (frame.shape[1] - text_width - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示详细指标
        cv2.putText(frame, f"Variance: {variance:.2f}", (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Black Ratio: {black_ratio:.2%}", (frame.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Edge Pixels: {edge_pixels}", (frame.shape[1] - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Brightness: {mean_brightness:.2f}", (frame.shape[1] - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return {
            'is_black': is_black,
            'variance': variance,
            'black_ratio': black_ratio,
            'edge_pixels': edge_pixels,
            'mean_brightness': mean_brightness,
            'frame': frame
        }

    def shutdown(self):
        pass