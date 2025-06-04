import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import yaml
import time
from string import Template
import cv2
from uvc_core.camera_controller import UVCCamera
from uvc_core.plugin_base import PluginManager



class Application:
    def __init__(self, config_path="config.yaml"):
        def resolve_env_vars(yaml_dict):
            """Process yaml with Template strings for safer environment variable resolution."""
            if isinstance(yaml_dict, dict):
                return {k: resolve_env_vars(v) for k, v in yaml_dict.items()}
            elif isinstance(yaml_dict, list):
                return [resolve_env_vars(i) for i in yaml_dict]
            elif isinstance(yaml_dict, str):
                # Create a template and substitute environment variables
                template = Template(yaml_dict)
                return template.safe_substitute(os.environ)
            else:
                return yaml_dict
        self.active_plugins = {}  # Initialize empty plugin dictionary
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = yaml.safe_load(f)
            self.config = resolve_env_vars(config)

        self.camera = UVCCamera(self.config.get("device_id", 0))
        self.plugin_manager = PluginManager()
        self.load_plugins()

    def load_plugins(self):
        for plugin_cfg in self.config["plugins"]:
            if plugin_cfg.get("status", "disable") == "disable":
                continue
            module = __import__(plugin_cfg["module"], fromlist=[""])
            plugin_class = getattr(module, plugin_cfg["class"])
            self.active_plugins[plugin_cfg["name"]] = plugin_class(plugin_cfg["name"],
                plugin_cfg.get("config", {})
            )
            self.plugin_manager.register_plugin(plugin_cfg["name"], plugin_class)

    def run(self):
        try:
            self.camera.initialize()
            for name, plugin in self.active_plugins.items():
                plugin.initialize()
            while True:
                ret, frame = self.camera.get_frame()
                if not ret:
                    continue

                # Maintain OpenCV event loop
                if cv2.waitKey(1) == 27:  # ESC key
                    break

                results = {}
                for name, plugin in self.active_plugins.items():
                    results[name] = plugin.process_frame(frame)

                self.handle_results(results, frame)
                #analysis = self.camera.analyze_image(frame)
                #self.camera.show_analysis(analysis, frame)
                self.camera.show_frame(frame)  # 添加预览窗口显示
                time.sleep(1 / self.config.get("fps", 30))

        except KeyboardInterrupt:
            self.shutdown()

    def handle_results(self, results, frame):
        # print('\n' + '='*40)
        #for name, result in results.items():
        #    print(f"{name}: {result}")
        for name, plugin in self.active_plugins.items():
            if name in results:
                plugin.handle_results(results[name], frame)



    def shutdown(self):
        self.camera.release()
        for plugin in self.active_plugins.values():
            plugin.shutdown()


if __name__ == "__main__":
    app = Application()
    app.run()
