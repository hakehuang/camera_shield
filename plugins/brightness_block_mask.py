# Copyright 2025 Hake Huang <hakehuang@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from camera_shield.uvc_core.plugin_base import DetectionPlugin
from datetime import datetime


class BrightnessBlockMaskPlugin(DetectionPlugin):
    """
    Plugin that generates block masks based on brightness analysis.
    Divides the image into blocks and creates masks based on brightness thresholds.
    """

    def __init__(self, name, config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Default configuration
        self.block_size = self.config.get("block_size", 32)
        self.brightness_threshold_low = self.config.get("brightness_threshold_low", 50)
        self.brightness_threshold_high = self.config.get(
            "brightness_threshold_high", 200
        )
        self.mask_mode = self.config.get(
            "mask_mode", "binary"
        )  # 'binary', 'gradient', 'adaptive'
        self.blur_kernel_size = self.config.get("blur_kernel_size", 5)
        self.enable_morphology = self.config.get("enable_morphology", True)
        self.morph_kernel_size = self.config.get("morph_kernel_size", 3)
        self.frame_processing_count = self.config.get("frame_processing_count", 20)
        self.frame_count = 0
        self.raw_mask = None
        self.block_target_name = self.config.get("block_target_name", "frdm_mcxn947")
        self.directory = self.config.get("directory", "./bbm")
        self.mode = self.config.get("mode", "generate")  # 'generate' or 'load'

    def initialize(self):
        """Initialize detection resources"""
        pass

    def preprocess_frame(self, frame):
        """Prepare frame for processing"""
        if self.mode == "load":
            # Load existing mask if available
            mask_path = os.path.join(self.directory, f"{self.block_target_name}.pkl")
            if os.path.exists(mask_path):
                with open(mask_path, "rb") as f:
                    bbm = pickle.load(f)
                    name = bbm.get("name", None)
                    if name != self.block_target_name:
                        return {
                            "result": "error",
                            "message": "Mask name does not match target name",
                        }
                    self.raw_mask = bbm.get("mask", None)
            else:
                return {"result": "error", "message": "Mask file not found"}

            masked_image = cv2.bitwise_and(frame, frame, mask=self.raw_mask)
            frame = masked_image

        return {"result": "ok"}

    def process_frame(self, frame):
        """
        Process frame and generate brightness-based block mask

        Args:
            frame: Input BGR frame

        Returns:
            Dictionary containing mask and analysis results
        """

        if self.mode == "generate":
            self.frame_count += 1
            if self.frame_count > self.frame_processing_count:
                return {"info": "Frame processing limit reached"}

            # Convert to grayscale for brightness analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.copy()
            # Generate block mask
            if self.raw_mask is not None:
                block_mask = self._generate_block_mask(gray)
            else:
                self.raw_mask = self._generate_block_mask(gray)
                block_mask = self.raw_mask.copy()

            return {
                "block_mask_raw": block_mask,
            }

        return {"info": "No processing required in 'load' mode"}

    def _generate_block_mask(self, gray: np.ndarray) -> np.ndarray:
        """Generate initial block-based mask"""
        height, width = gray.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Process image in blocks
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                # Define block boundaries
                y_end = min(y + self.block_size, height)
                x_end = min(x + self.block_size, width)

                # Extract block
                block = gray[y:y_end, x:x_end]

                # Calculate block brightness statistics
                block_mean = np.mean(block)
                block_std = np.std(block)

                # Determine if block should be masked
                should_mask = self._evaluate_block(block_mean, block_std, block)

                if should_mask:
                    mask[y:y_end, x:x_end] = 255

        return mask

    def _evaluate_block(
        self, mean_brightness: float, std_brightness: float, block: np.ndarray
    ) -> bool:
        """Evaluate whether a block should be masked based on brightness criteria"""

        # Check brightness thresholds
        if (
            mean_brightness < self.brightness_threshold_low
            or mean_brightness > self.brightness_threshold_high
        ):
            return True

        # Check for low variance (potentially problematic areas)
        if std_brightness < 10:  # Very uniform areas
            return True

        # Check for extreme values within block
        min_val, max_val = np.min(block), np.max(block)
        if max_val - min_val < 20:  # Very low dynamic range
            return True

        return False

    def handle_results(self, result, frame):
        """
        Handle results after processing the frame.
        This method can be overridden to implement custom result handling logic.
        """
        # For now, just print the statistics
        if self.frame_count > self.frame_processing_count:
            if "block_mask_raw" in result:
                cur_raw_mask = result["block_mask_raw"].astype(np.float32)
                corrected = (self.raw_mask.astype(np.float32) + cur_raw_mask) / 2
                self.raw_mask = corrected.astype(np.uint8)

    def shutdown(self) -> list:
        """Release plugin resources"""
        # Save mask
        if self.raw_mask is not None:
            # Create directory if it doesn't exist
            os.makedirs(self.directory, exist_ok=True)
            # put a png here for debug purpose
            cv2.imwrite(os.path.join(self.directory, "bbm.png"), self.raw_mask)

            # Create filename based on fingerprint ID and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bbm_{timestamp}.pkl"
            filepath = os.path.join(self.directory, filename)
            bbm = {
                "name": self.block_target_name,
                "mask": self.raw_mask,
            }
            with open(filepath, "wb") as f:
                pickle.dump(bbm, f)
        return []