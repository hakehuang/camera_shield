# UVC摄像头质量检测框架

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

实时检测摄像头画面异常（黑屏/花屏/伪影），支持插件式扩展检测算法

## 功能特性
- 🎥 自动对焦/白平衡控制
- 🔌 模块化插件系统
- 📊 多维检测指标输出
- ⚙️ 动态配置热加载

## 安装指南
```powershell
# 创建虚拟环境
python -m venv .ven

# 激活环境
.venv\Scripts\activate

# 安装依赖
uv pip install -r requirements.txt  # 或使用 pip install -r requirements.txt
```

## 快速开始
```python
# 运行检测程序
python main.py --config config.yaml

# 实时输出示例
[DEBUG] 帧率:30 | 黑屏检测:正常 | 方差:85.6 | 亮度均值:127
[ALERT] 检测到黑屏! 方差:12.3 < 阈值50
```

## 配置文件
```yaml:c:\github\uvc_shield\config.yaml
device_id: 0
frame_rate: 30
plugins:
  black_screen_detector:
    variance_threshold: 50
    histogram_threshold: 0.95
    edge_threshold: 50
    brightness_threshold: 20
```

## 插件开发
1. 在`plugins/`目录创建新插件
2. 继承`DetectionPlugin`基类
3. 注册到`PluginManager`
```python
from uvc_core.plugin_base import DetectionPlugin

class ArtifactDetector(DetectionPlugin):
    def process_frame(self, frame):
        # 实现你的检测逻辑
        return {'is_artifact': False}
```

## 贡献指南
欢迎提交PR完善以下内容：
- 更多图像异常检测算法
- 可视化数据看板
- 单元测试用例

许可证：MIT