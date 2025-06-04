import cv2
import copy
import numpy as np
from uvc_core.plugin_base import DetectionPlugin


class PatternNoiseDetector(DetectionPlugin):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.thresholds = {
            "pattern_score": config.get("pattern_threshold", 0.5),
            "variance_threshold": config.get("variance_threshold", 50),
            "edge_pixels": config.get("edge_pixels", 5000)
        }
        self.frame_history = []
        self.history_size = config.get("history_size", 5)
        self.result_holding_frames = config.get("result_holding_frames", 5)

    def initialize(self):
        """初始化检测所需资源"""
        pass

    def _calculate_pattern_score(self, frame):
        """计算花屏模式得分"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpen the image before edge detection
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        edges = cv2.Canny(sharpened, 100, 200)
        edge_pixels = np.count_nonzero(edges)
        cv2.imwrite("edges.jpg", edges)

        # 1. 计算图像方差
        variance = np.var(gray)
        if variance < self.thresholds["variance_threshold"]:
            return 0.0

        # 2. FFT频谱分析
        fft = np.fft.fft2(gray.astype(float))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1e-5)

        # 3. 检测高频和低频能量分布
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2

        # 计算不同区域的能量分布
        low_freq = magnitude[
            center_y - 10 : center_y + 10, center_x - 10 : center_x + 10
        ].mean()
        high_freq = magnitude.mean() - low_freq

        # 4. 计算模式得分
        pattern_score = abs(high_freq) / (low_freq + 1e-5)
        return pattern_score, edge_pixels

    def process_frame(self, frame):
        """处理视频帧，检测花屏"""
        # 计算当前帧的模式得分
        current_score, edge_pixels = self._calculate_pattern_score(frame)

        # 维护帧历史记录
        self.frame_history.append(current_score)
        if len(self.frame_history) > self.history_size:
            self.frame_history.pop(0)

        # 判断是否为花屏
        is_pattern_noise = current_score > self.thresholds["pattern_score"] or \
                            edge_pixels > self.thresholds["edge_pixels"]


        return {
            "is_pattern_noise": is_pattern_noise,
            "pattern_score": current_score,
            "edge_pixels": edge_pixels,
            "frame": frame,
        }

    def handle_results(self, result, frame):
        # 在画面上显示检测结果

        if self.update_count > self.result_holding_frames:
            self.old_result = copy.deepcopy(result)

        if self.old_result:
            is_pattern_noise = self.old_result['is_pattern_noise']
            current_score = self.old_result['pattern_score']
            edge_pixels = self.old_result['edge_pixels']
        else:
            is_pattern_noise = result['is_pattern_noise']
            current_score = result['pattern_score']
            edge_pixels = result['edge_pixels']

        # 在右下角显示检测结果和得分
        cv2.putText(
            frame,
            f"edge: {edge_pixels:.2f}",
            (frame.shape[1] - 300, frame.shape[0] - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Pattern Noise: {'Detected' if is_pattern_noise else 'Normal'}",
            (frame.shape[1] - 300, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Score: {current_score:.2f}",
            (frame.shape[1] - 300, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )



    def shutdown(self):
        """释放插件资源"""
        pass
