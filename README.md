# UVC Camera Quality Detection Framework

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Real-time detection of camera anomalies (black screen/pattern noise/artifacts) with plugin-based algorithm extension

## Features
- 🎥 Auto focus/white balance control
- 🔌 Modular plugin system
- 📊 Multi-dimensional detection metrics output
- ⚙️ Dynamic configuration hot-reloading

## Installation Guide
```powershell
# Create virtual environment
python -m venv .ven

# Activate environment
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt  # or use pip install -r requirements.txt
```

## Quick Start
```python
# Run detection program
python main.py --config config.yaml

# Real-time output example
[DEBUG] FPS:30 | Black screen:Normal | Variance:85.6 | Brightness:127
[ALERT] Black screen detected! Variance:12.3 < threshold 50
```

## Configuration
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

## Plugin Development
1. Create new plugin in `plugins/` directory
2. Extend `DetectionPlugin` base class
3. Register with `PluginManager`
```python
from uvc_core.plugin_base import DetectionPlugin

class ArtifactDetector(DetectionPlugin):
    def process_frame(self, frame):
        # 实现你的检测逻辑
        return {'is_artifact': False}
```

## Contribution Guide
PR contributions are welcome to improve the following:
- More image anomaly detection algorithms
- Visualization dashboard
- Unit test cases

License: Apache-2.0