import yaml
import time
import cv2
from uvc_core.camera_controller import UVCCamera
from uvc_core.plugin_base import PluginManager

class Application:
    def __init__(self, config_path='config.yaml'):
        self.active_plugins = {}  # Initialize empty plugin dictionary
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            self.config = yaml.safe_load(f)
        
        self.camera = UVCCamera(self.config.get('device_id', 0))
        self.plugin_manager = PluginManager()
        self.load_plugins()

    def load_plugins(self):
        for plugin_cfg in self.config['plugins']:
            module = __import__(plugin_cfg['module'], fromlist=[''])
            plugin_class = getattr(module, plugin_cfg['class'])
            self.active_plugins[plugin_cfg['name']] = plugin_class(plugin_cfg.get('config', {}))
            self.plugin_manager.register_plugin(
                plugin_cfg['name'], 
                plugin_class
            )

    def run(self):
        try:
            while True:
                ret, frame = self.camera.get_frame()
                if not ret:
                    continue

                # Display frame with OpenCV
                self.camera.show_frame(frame)

                # Maintain OpenCV event loop
                if cv2.waitKey(1) == 27:  # ESC key
                    break

                results = {}
                for name, plugin in self.active_plugins.items():
                    results[name] = plugin.process_frame(frame)
                    self.camera.show_frame(frame)  # 添加预览窗口显示

                self.handle_results(results)
                time.sleep(1/self.config.get('fps', 30))

        except KeyboardInterrupt:
            self.shutdown()

    def handle_results(self, results):
        #print('\n' + '='*40)
        for name, result in results.items():
            print(f'{name}: {result}')

    def shutdown(self):
        self.camera.release()
        for plugin in self.active_plugins.values():
            plugin.shutdown()

if __name__ == '__main__':
    app = Application()
    app.run()