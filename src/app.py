import base64
import cv2
import flet as ft
import logging
import threading
import time
from pathlib import Path
from .camera_manager import CameraManager
from .api_client import DeepSeekClient
from .config_manager import ConfigManager
from .utils import check_image_quality

class StudyAssistantApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "智能学习助手"
        self.page.window_width = 1200
        self.page.window_height = 800
        
        # 初始化模块
        self.cfg = ConfigManager()
        self.camera_mgr = CameraManager(self.cfg)
        self.api_client = DeepSeekClient(self.cfg)
        
        # 状态变量
        self.is_previewing = False
        self.auto_capture_timer = None
        
        self._init_ui()
        self._bind_events()
        self._start_camera_preview()

    def _init_ui(self):
        """初始化用户界面"""
        # 图像预览区
        self.img_preview = ft.Image(
            src="assets/placeholder.png", 
            width=640, 
            height=480,
            fit=ft.ImageFit.CONTAIN
        )
        
        # 控制面板
        self.capture_btn = ft.ElevatedButton("手动抓拍", icon=ft.icons.CAMERA)
        self.auto_switch = ft.Switch(label="自动抓拍", value=False)
        self.interval_input = ft.TextField(
            label="捕获间隔(秒)", 
            value=str(self.cfg.get("auto_capture_interval", 10)),
            width=150
        )
        
        # API设置区
        self.api_key_input = ft.TextField(
            label="API密钥", 
            value=self.cfg.get("api_key", ""),
            password=True,
            can_reveal_password=True
        )
        self.endpoint_input = ft.TextField(
            label="API端点", 
            value=self.cfg.get("api_endpoint", "https://api.deepseek.com/v1/analyze")
        )
        
        # 系统日志区
        self.log_view = ft.ListView(expand=True)
        
        # 布局结构
        self.page.add(
            ft.Row([
                ft.Column([
                    ft.Stack([
                        self.img_preview,
                        ft.Container(
                            content=ft.Text("摄像头未连接", size=20),
                            alignment=ft.alignment.center,
                            visible=False,
                            key="camera_status"  # 使用 key 参数
                        )
                    ]),
                    ft.Row([
                        self.capture_btn,
                        ft.VerticalDivider(),
                        self.auto_switch,
                        self.interval_input
                    ], spacing=20)
                ], width=700),
                
                ft.Column([
                    ft.Card(
                        content=ft.Column([
                            ft.Text("API 配置", style=ft.TextThemeStyle.HEADLINE_SMALL),
                            self.api_key_input,
                            self.endpoint_input,
                            ft.ElevatedButton("保存配置", icon=ft.icons.SAVE)
                        ], spacing=15)
                    ),
                    ft.Divider(),
                    ft.Card(
                        content=ft.Column([
                            ft.Text("系统日志", style=ft.TextThemeStyle.HEADLINE_SMALL),
                            ft.Container(
                                content=self.log_view,
                                height=300,
                                border=ft.border.all(1, ft.colors.OUTLINE)
                            )
                        ])
                    )
                ], expand=True)
            ], spacing=20)
        )

    def _bind_events(self):
        """绑定事件处理"""
        self.page.on_close = self._on_app_close
        self.capture_btn.on_click = self.capture_image
        self.auto_switch.on_change = self.toggle_auto_capture

    def _start_camera_preview(self):
        """启动摄像头预览"""
        try:
            self.camera_mgr.start_capture()
            self.is_previewing = True
            threading.Thread(target=self._update_preview, daemon=True).start()
        except Exception as e:
            self._log(f"摄像头初始化失败: {str(e)}")
            self._show_camera_status(False)

    def _update_preview(self):
        """实时更新预览画面"""
        while self.is_previewing:
            frame = self.camera_mgr.get_frame()
            if frame is not None:
                self._update_image(frame)
                self._show_camera_status(True)
            time.sleep(0.03)

    def _update_image(self, frame):
        """更新界面图像显示"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        self.img_preview.src = f"data:image/jpeg;base64,{img_base64}"
        self.page.update()

    def capture_image(self, e):
        """手动捕获图像"""
        if frame := self.camera_mgr.get_frame():
            if check_image_quality(frame):
                threading.Thread(target=self._process_image, args=(frame,)).start()
            else:
                self._log("图像质量不合格，请重新拍摄")

    def toggle_auto_capture(self, e):
        """切换自动捕获模式"""
        if self.auto_switch.value:
            interval = int(self.interval_input.value)
            self.auto_capture_timer = threading.Timer(interval, self._auto_capture_task)
            self.auto_capture_timer.start()
        elif self.auto_capture_timer:
            self.auto_capture_timer.cancel()

    def _auto_capture_task(self):
        """自动捕获任务"""
        if self.auto_switch.value:
            self.capture_image(None)
            interval = int(self.interval_input.value)
            self.auto_capture_timer = threading.Timer(interval, self._auto_capture_task)
            self.auto_capture_timer.start()

    def _process_image(self, frame):
        """处理图像分析流程"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_data = base64.b64encode(buffer).decode('utf-8')
            
            # 调用API
            result = self.api_client.analyze_image(img_data)
            self._show_analysis_result(result)
        except Exception as e:
            self._log(f"处理失败: {str(e)}")

    def _show_analysis_result(self, result):
        """显示分析结果弹窗"""
        content = ft.Column([
            ft.Text(f"材料类型: {result.get('material_type', '未知')}", weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text(result.get('analysis', '无分析结果'))
        ], scroll=ft.ScrollMode.ALWAYS)
        
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("分析结果"),
            content=content,
            actions=[ft.TextButton("关闭", on_click=lambda e: self.page.close_dialog())]
        )
        self.page.dialog.open = True
        self.page.update()

    def _log(self, message):
        """记录日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.controls.append(ft.Text(f"[{timestamp}] {message}"))
        if len(self.log_view.controls) > 100:
            self.log_view.controls.pop(0)
        self.page.update()
        logging.info(message)

    def _on_app_close(self, e):
        """应用关闭时的清理操作"""
        self.is_previewing = False
        self.camera_mgr.stop()
        self.cfg.save_config({
            "api_key": self.api_key_input.value,
            "api_endpoint": self.endpoint_input.value,
            "auto_capture_interval": self.interval_input.value
        })
        if self.auto_capture_timer:
            self.auto_capture_timer.cancel()

    def _show_camera_status(self, connected):
        """更新摄像头状态显示"""
        status_container = self.page.get_control("camera_status")
        if status_container is not None:
            status_container.visible = not connected
        else:
            logging.warning("未找到 key 为 'camera_status' 的控件")
        self.page.update()

if __name__ == "__main__":
    ft.app(target=StudyAssistantApp)