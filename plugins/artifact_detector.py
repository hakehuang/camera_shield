# Copyright 2025 Hake Huang <hakehuang@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import copy
import numpy as np
from uvc_core.plugin_base import DetectionPlugin


class ArtifactDetector(DetectionPlugin):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.prev_frame = None
        self.alarm_duration = config.get("alarm_duration", 5)  # Alarm duration in frames
        self.current_alarms = 0
        self.thresholds = {
            "black_level": config.get("black_threshold", 15),
            "blur": config.get("blur_threshold", 100),
            "noise": config.get("noise_threshold", 20),
        }
        self.frame_counter = 0
        self.detection_interval = config.get("detection_interval", 30)
        self.result_holding_frames = config.get("result_holding_frames", 5)
        self.result = []

    def initialize(self):
        """Initialize detection resources"""
        # 可以在这里初始化任何需要的资源
        pass

    def _detect_black_screen(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.mean(gray)[0] < self.thresholds["black_level"]

    def _detect_motion_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian < self.thresholds["blur"]

    def _detect_image_noise(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std_dev = cv2.meanStdDev(gray)[1][0]
        return std_dev > self.thresholds["noise"]

    def _detect_static_frame(self, prev_frame, curr_frame):
        if prev_frame is None:
            return False
        diff = cv2.absdiff(prev_frame, curr_frame)
        non_zero = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
        return non_zero < 100

    def process_frame(self, frame):
        self.frame_counter += 1
        # Perform spectrum analysis on each frame (independent of detection logic)
        try:
            h, w = frame.shape[:2]
            display_size = min(h, w) // 3  # Dynamic display size

            # Color space conversion (camera output is RGB)
            analysis_roi = frame[10 : 10 + display_size, 10 : 10 + display_size]
            gray = cv2.cvtColor(analysis_roi, cv2.COLOR_RGB2GRAY)

            # Debug informatio
            print(f"[DEBUG] Grayscale range: {np.min(gray)}-{np.max(gray)}")

            # FFT calculation
            fft = np.fft.fft2(gray.astype(float))
            fft_shift = np.fft.fftshift(fft)
            magnitude = 20 * np.log(np.abs(fft_shift) + 1e-5)

            # Enhanced visualization
            norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        except Exception as e:
            print(f"[ERROR] Spectrum processing failed: {e}")
            import traceback

            traceback.print_exc()

        # Core detection logic
        alerts = []
        if self.frame_counter % self.detection_interval == 0:
            if self._detect_black_screen(frame):
                alerts.append("black_screen")
            if self._detect_motion_blur(frame):
                alerts.append("motion_blur")
            if self._detect_image_noise(frame):
                alerts.append("image_noise")

        return {
            "norm_mag": norm_mag,
            "alerts": alerts,
            "display_size": display_size,
            "frame": frame
        }

    def handle_results(self, result, frame):

        if self.update_count > self.result_holding_frames:
            self.old_result = copy.deepcopy(result)

        if self.old_result:
            alerts = self.old_result["alerts"]
            display_size = self.old_result["display_size"]
            norm_mag = self.old_result["norm_mag"]
        else:
            alerts = result["alerts"]
            display_size = result["display_size"]
            norm_mag = result["norm_mag"]

        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(norm_mag, alpha=1.5), cv2.COLORMAP_JET
        )

        # Overlay display
        overlay = cv2.resize(colored, (display_size, display_size))
        cv2.addWeighted(
            overlay,
            0.7,
            frame[10 : 10 + display_size, 10 : 10 + display_size],
            0.3,
            0,
            frame[10 : 10 + display_size, 10 : 10 + display_size],
        )

        # Draw green border
        cv2.rectangle(
            frame, (10, 10), (10 + display_size, 10 + display_size), (0, 255, 0), 2
        )
        if alerts:
            self.current_alarms = self.alarm_duration
            self.result = ['find_artifact']
            cv2.putText(
                frame,
                f"Alerts: {', '.join(alerts)}",
                (display_size + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Alarm state handling
        if self.current_alarms > 0:
            self.current_alarms -= 1
            frame = cv2.copyMakeBorder(
                frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 255)
            )

    def shutdown(self) -> list:
        """Release plugin resources"""
        return self.result
