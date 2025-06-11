# Copyright 2025 Hake Huang <hakehuang@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, Any

class DetectionPlugin(ABC):
    def __init__(self, name, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.update_count = 0
        self.old_result = {}

    @abstractmethod
    def initialize(self):
        """initialize"""

    @abstractmethod
    def process_frame(self, frame) -> Dict[str, Any]:
        """process_frame"""

    @abstractmethod
    def handle_results(self, result, frame):
        """handle_results"""

    @abstractmethod
    def shutdown(self) -> list:
        """release resources"""


class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name: str, plugin_class):
        self.plugins[name] = plugin_class

    def create_plugin(self, name: str, config: Dict) -> DetectionPlugin:
        return self.plugins[name](config)
