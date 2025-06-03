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
        """初始化检测所需资源"""

    @abstractmethod
    def process_frame(self, frame) -> Dict[str, Any]:
        """处理视频帧并返回检测结果"""

    @abstractmethod
    def handle_results(self, result, frame):
        """handle results"""

    @abstractmethod
    def shutdown(self):
        """释放插件资源"""


class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name: str, plugin_class):
        self.plugins[name] = plugin_class

    def create_plugin(self, name: str, config: Dict) -> DetectionPlugin:
        return self.plugins[name](config)
