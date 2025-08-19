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

# add video to user group
> sudo usermod -a -G video $USER
need log out and login to effective or run
> newgrp video
```

if you are in ubuntu 2404 with xwayland do below
```bash
 export DISPLAY=:0
 cp /run/user/1000/.mutter-Xwaylandauth.AIT2A3 ~/.Xauthority
```
or
```bash
 export QT_QPA_PLATFORM=xcb
```

if you server does not have display please do below
```bash
> pip uninstall opencv-python
> pip install opencv-python-headless
> export QT_QPA_PLATFORM=offscreen
```

## Quick Start
```python
# Run detection program
python -m camera_shield.main --config camera_shield/config.yaml
```

## Configuration
```yaml:c:\github\uvc_shield\config.yaml
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
        return {'is_artifact': False}
```

## Contribution Guide
PR contributions are welcome to improve the following:
- More image anomaly detection algorithms
- Visualization dashboard
- Unit test cases

License: Apache-2.0
