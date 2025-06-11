# Copyright 2025 Hake Huang <hakehuang@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import copy
import numpy as np
from uvc_core.plugin_base import DetectionPlugin


class BlackScreenDetector(DetectionPlugin):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.threshold = {
            "variance": config.get("variance_threshold", 50),
            "histogram": config.get("histogram_threshold", 0.95),
            "edges": config.get("edge_threshold", 50),
            "brightness": config.get("brightness_threshold", 20),
        }
        self.result_holding_frames = config.get("result_holding_frames", 5)
        self.result = []

    def initialize(self):
        """Initialize detection resources"""
        # Can initialize any required resources here
        pass

    def process_frame(self, frame):
        print("black frame check\n")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Multi-algorithm detection
        variance = np.var(gray)

        # Histogram analysis (black pixel ratio)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        black_ratio = np.sum(hist[:20]) / gray.size

        # Preprocessing: Gaussian blur and thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresholded = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Edge detection (Canny edge count)
        edges = cv2.Canny(thresholded, 100, 200)
        edge_pixels = np.count_nonzero(edges)

        # Brightness mean detection
        mean_brightness = np.mean(gray)

        # Display detection results on frame
        # Voting mechanism: at least 3 conditions to determine black screen
        vote_count = (
            int(variance < self.threshold["variance"])
            + int(black_ratio > self.threshold["histogram"])
            + int(mean_brightness < self.threshold["brightness"])
            + int(edge_pixels < self.threshold["edges"])
        )
        is_black = vote_count >= 2

        return {
            "is_black": is_black,
            "variance": variance,
            "black_ratio": black_ratio,
            "edge_pixels": edge_pixels,
            "mean_brightness": mean_brightness,
            "frame": frame,
        }

    def handle_results(self, result, frame):
        # Display detection status
        if self.update_count > self.result_holding_frames:
            self.old_result = copy.deepcopy(result)

        if self.old_result:
            is_black = self.old_result["is_black"]
            variance = self.old_result["variance"]
            black_ratio = self.old_result["black_ratio"]
            edge_pixels = self.old_result["edge_pixels"]
            mean_brightness = self.old_result["mean_brightness"]
        else:
            is_black = result["is_black"]
            variance = result["variance"]
            black_ratio = result["black_ratio"]
            edge_pixels = result["edge_pixels"]
            mean_brightness = result["mean_brightness"]

        if is_black:
            self.result = ["black screen detected"]

        status_text = "Black Screen Detected" if is_black else "Normal"
        color = (0, 0, 255) if is_black else (0, 255, 0)
        text_width = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][
            0
        ]
        cv2.putText(
            frame,
            status_text,
            (frame.shape[1] - text_width - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # Display detailed metrics
        cv2.putText(
            frame,
            f"Variance: {variance:.2f}",
            (frame.shape[1] - 200, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Black Ratio: {black_ratio:.2%}",
            (frame.shape[1] - 200, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Edge Pixels: {edge_pixels}",
            (frame.shape[1] - 200, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Brightness: {mean_brightness:.2f}",
            (frame.shape[1] - 200, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def shutdown(self) -> list:
        """Release plugin resources"""
        return self.result
