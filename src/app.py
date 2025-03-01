import base64
import cv2
import flet as ft
import logging
import threading
import time
from pathlib import Path
from threading import Lock
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
        self.camera_mgr = None  # 延迟初始化
        self.api_client = None  # 延迟初始化
        
        # 状态变量
        self.is_previewing = False
        self.preview_lock = Lock()  # 添加锁以保护并发访问
        self.auto_capture_timer = None
        self.update_ui_timer = None
        self.frame_buffer = None
        self.frame_buffer_lock = Lock()
        self.latest_result = None  # 存储最新的分析结果
        
        # 初始化界面
        self._init_ui()
        self._bind_events()
        
        # 初始化服务
        self._init_services()
        
    def _init_services(self):
        """初始化服务组件"""
        try:
            self.api_client = DeepSeekClient(self.cfg)
            self._log("API客户端初始化完成")
            
            self.camera_mgr = CameraManager(self.cfg)
            self._start_camera_preview()
        except Exception as e:
            self._log(f"服务初始化失败: {str(e)}")
            self._show_error_dialog("初始化错误", f"无法启动必要服务: {str(e)}")

    def _init_ui(self):
        """初始化用户界面"""
        # 创建主导航栏
        self.page.appbar = ft.AppBar(
            title=ft.Text("智能学习助手"),
            actions=[
                ft.IconButton(
                    icon=ft.icons.SETTINGS,
                    tooltip="设置",
                    on_click=self._show_settings
                )
            ]
        )
        
        # 创建视图容器
        self.main_view = self._create_main_view()
        self.settings_view = self._create_settings_view()
        
        # 默认显示主视图
        self.page.add(self.main_view)
        
    def _create_main_view(self):
        """创建主视图"""
        # 图像预览区
        self.img_preview = ft.Image(
            src="assets/placeholder.png", 
            width=640, 
            height=480,
            fit=ft.ImageFit.CONTAIN
        )
        
        # 状态指示器
        self.status_indicator = ft.Container(
            content=ft.Text("摄像头未连接", size=20, color=ft.colors.WHITE),
            bgcolor=ft.colors.RED_400,
            padding=10,
            border_radius=5,
            alignment=ft.alignment.center,
            visible=True
        )
        
        # 控制面板
        self.capture_btn = ft.ElevatedButton(
            "手动抓拍", 
            icon=ft.icons.CAMERA,
            disabled=True  # 初始状态禁用
        )
        
        self.auto_switch = ft.Switch(
            label="自动抓拍", 
            value=False,
            disabled=True  # 初始状态禁用
        )
        
        self.interval_input = ft.TextField(
            label="捕获间隔(秒)", 
            value=str(self.cfg.get("auto_capture_interval", 10)),
            width=150,
            keyboard_type=ft.KeyboardType.NUMBER
        )
        
        # 结果显示区
        self.result_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("分析结果", style=ft.TextThemeStyle.HEADLINE_SMALL),
                    ft.Divider(),
                    ft.Container(
                        content=ft.Text("尚无分析结果，请捕获图像进行分析"),
                        padding=20,
                        alignment=ft.alignment.center,
                        height=400
                    )
                ]),
                padding=15
            ),
            expand=True
        )
        
        # 主视图布局
        return ft.Column([
            ft.Row([
                ft.Column([
                    ft.Stack([
                        self.img_preview,
                        self.status_indicator
                    ]),
                    ft.Row([
                        self.capture_btn,
                        ft.VerticalDivider(),
                        self.auto_switch,
                        self.interval_input
                    ], spacing=20)
                ], width=700),
                
                # 结果显示区位于右侧
                ft.Column([
                    self.result_card
                ], expand=True)
            ], spacing=20, expand=True)
        ], expand=True)
    
    def _create_settings_view(self):
        """创建设置视图"""
        # API设置区
        self.api_key_input = ft.TextField(
            label="API密钥", 
            value=self.cfg.get("api_key", ""),
            password=True,
            can_reveal_password=True,
            expand=True
        )
        
        self.endpoint_input = ft.TextField(
            label="API端点", 
            value=self.cfg.get("api_endpoint", "https://api.deepseek.com/v1/analyze"),
            expand=True
        )
        
        # 保存配置按钮
        self.save_config_btn = ft.ElevatedButton(
            "保存配置", 
            icon=ft.icons.SAVE,
            on_click=self._save_config
        )
        
        # 系统日志区
        self.log_view = ft.ListView(
            expand=True,
            auto_scroll=True
        )
        
        # 返回按钮
        back_button = ft.ElevatedButton(
            "返回主界面",
            icon=ft.icons.ARROW_BACK,
            on_click=self._show_main_view
        )
        
        # 设置视图布局
        return ft.Column([
            ft.Container(
                content=ft.Text("应用设置", size=24, weight=ft.FontWeight.BOLD),
                margin=ft.margin.only(bottom=20)
            ),
            
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("API 配置", style=ft.TextThemeStyle.HEADLINE_SMALL),
                        ft.Row([self.api_key_input], expand=True),
                        ft.Row([self.endpoint_input], expand=True),
                        ft.Row([
                            ft.Container(expand=True),
                            self.save_config_btn
                        ])
                    ], spacing=15),
                    padding=15
                )
            ),
            ft.Divider(),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("系统日志", style=ft.TextThemeStyle.HEADLINE_SMALL),
                        ft.Container(
                            content=self.log_view,
                            height=300,
                            border=ft.border.all(1, ft.colors.OUTLINE),
                            expand=True
                        )
                    ]),
                    padding=15
                ),
                expand=True
            ),
            ft.Container(
                content=back_button,
                alignment=ft.alignment.center,
                margin=ft.margin.only(top=20)
            )
        ], expand=True, spacing=20, padding=20)

    def _show_main_view(self, e=None):
        """显示主视图"""
        self.page.controls.clear()
        self.page.add(self.main_view)
        self.page.update()
        
    def _show_settings(self, e=None):
        """显示设置视图"""
        self.page.controls.clear()
        self.page.add(self.settings_view)
        self.page.update()

    def _bind_events(self):
        """绑定事件处理"""
        self.page.on_close = self._on_app_close
        self.capture_btn.on_click = self._on_capture_btn_click
        self.auto_switch.on_change = self._on_auto_switch_change
        self.interval_input.on_blur = self._validate_interval
        
    def _validate_interval(self, e):
        """验证间隔输入"""
        try:
            value = int(self.interval_input.value)
            if value < 1:
                self.interval_input.value = "1"
                self._log("捕获间隔不能小于1秒")
            elif value > 3600:
                self.interval_input.value = "3600"
                self._log("捕获间隔不能大于3600秒")
        except ValueError:
            self.interval_input.value = str(self.cfg.get("auto_capture_interval", 10))
            self._log("请输入有效的数字")
        self.page.update()

    def _start_camera_preview(self):
        """启动摄像头预览"""
        try:
            self.camera_mgr.start_capture()
            with self.preview_lock:
                self.is_previewing = True
            
            # 启动预览线程
            preview_thread = threading.Thread(target=self._update_preview, daemon=True)
            preview_thread.start()
            
            # 启动UI更新定时器 (每100ms更新一次UI，而不是每30ms)
            self.update_ui_timer = threading.Timer(0.1, self._update_ui)
            self.update_ui_timer.daemon = True
            self.update_ui_timer.start()
            
            # 启用控制按钮
            self.capture_btn.disabled = False
            self.auto_switch.disabled = False
            self.page.update()
            
            self._log("摄像头预览已启动")
        except Exception as e:
            self._log(f"摄像头初始化失败: {str(e)}")
            self._update_camera_status(False, f"摄像头错误: {str(e)}")

    def _update_preview(self):
        """实时获取预览画面（不直接更新UI）"""
        while True:
            with self.preview_lock:
                if not self.is_previewing:
                    break
                
            try:
                frame = self.camera_mgr.get_frame()
                if frame is not None:
                    with self.frame_buffer_lock:
                        self.frame_buffer = frame
                    self._update_camera_status(True)
            except Exception as e:
                self._log(f"获取画面错误: {str(e)}")
                self._update_camera_status(False, str(e))
                # 出错后短暂暂停，避免大量错误消息
                time.sleep(1)
                
            # 预览循环的暂停时间
            time.sleep(0.03)
    
    def _update_ui(self):
        """定期更新UI"""
        if not self.is_previewing:
            return
            
        try:
            with self.frame_buffer_lock:
                if self.frame_buffer is not None:
                    self._update_image(self.frame_buffer)
        except Exception as e:
            logging.error(f"UI更新错误: {str(e)}")
        
        # 重新调度UI更新
        self.update_ui_timer = threading.Timer(0.1, self._update_ui)
        self.update_ui_timer.daemon = True
        self.update_ui_timer.start()

    def _update_image(self, frame):
        """更新界面图像显示"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            self.img_preview.src = f"data:image/jpeg;base64,{img_base64}"
            self.page.update()
        except Exception as e:
            logging.error(f"图像转换错误: {str(e)}")

    def _on_capture_btn_click(self, e):
        """手动捕获按钮点击事件"""
        self.capture_image()
        
    def capture_image(self):
        """捕获图像并处理"""
        try:
            frame = self.camera_mgr.get_frame()
            if frame is None:
                self._log("无法获取图像，请检查摄像头")
                return
                
            quality_check, sharpness, brightness = check_image_quality(frame)
            if not quality_check:
                self._log(f"图像质量不合格 (清晰度: {sharpness:.1f}, 亮度: {brightness:.1f})")
                self._show_error_dialog(
                    "图像质量检查", 
                    f"图像质量不合格:\n清晰度: {sharpness:.1f}\n亮度: {brightness:.1f}\n\n请调整拍摄环境后重试。"
                )
                return
                
            self._log("正在分析图像...")
            threading.Thread(
                target=self._process_image, 
                args=(frame.copy(),),  # 复制frame避免线程安全问题
                daemon=True
            ).start()
        except Exception as e:
            self._log(f"捕获图像失败: {str(e)}")

    def _on_auto_switch_change(self, e):
        """切换自动捕获模式"""
        if self.auto_switch.value:
            try:
                interval = int(self.interval_input.value)
                if interval < 1:
                    interval = 1
                    self.interval_input.value = "1"
                    self.page.update()
                    
                self._log(f"自动捕获已启用, 间隔: {interval}秒")
                self._schedule_auto_capture(interval)
            except ValueError:
                self.auto_switch.value = False
                self._log("请输入有效的时间间隔")
                self.page.update()
        else:
            self._cancel_auto_capture()
            self._log("自动捕获已禁用")

    def _schedule_auto_capture(self, interval):
        """调度自动捕获任务"""
        self._cancel_auto_capture()  # 确保取消之前的定时器
        
        self.auto_capture_timer = threading.Timer(interval, self._auto_capture_task)
        self.auto_capture_timer.daemon = True
        self.auto_capture_timer.start()

    def _cancel_auto_capture(self):
        """取消自动捕获"""
        if self.auto_capture_timer:
            self.auto_capture_timer.cancel()
            self.auto_capture_timer = None

    def _auto_capture_task(self):
        """自动捕获任务"""
        if not self.auto_switch.value:
            return
            
        self.capture_image()
        
        try:
            interval = int(self.interval_input.value)
            if interval < 1:
                interval = 1
            self._schedule_auto_capture(interval)
        except ValueError:
            self._log("自动捕获异常: 无效的间隔值")
            self.auto_switch.value = False
            self.page.update()

    def _process_image(self, frame):
        """处理图像分析流程"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_data = base64.b64encode(buffer).decode('utf-8')
            
            self._log("正在调用API分析图像...")
            # 检查API配置
            if not self.cfg.get("api_key"):
                self._log("错误: API密钥未配置")
                self._show_error_dialog("API配置错误", "请在配置面板中设置有效的API密钥")
                return
                
            # 调用API
            result = self.api_client.analyze_image(img_data)
            self._log("分析完成")
            
            # 保存结果并更新显示
            self.latest_result = result
            self._update_result_display(result)
        except Exception as e:
            self._log(f"处理失败: {str(e)}")
            self._show_error_dialog("处理错误", f"图像分析失败: {str(e)}")

    def _update_result_display(self, result):
        """更新结果显示区域"""
        material_type = result.get('material_type', '未知')
        analysis_text = result.get('analysis', '无分析结果')
        
        # 创建新的结果显示内容
        result_content = ft.Column([
            ft.Text("分析结果", style=ft.TextThemeStyle.HEADLINE_SMALL),
            ft.Divider(),
            ft.Container(
                content=ft.Column([
                    ft.Text(f"材料类型: {material_type}", weight=ft.FontWeight.BOLD, size=16),
                    ft.Divider(height=10, thickness=1),
                    ft.Text("分析内容:", weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=ft.Text(analysis_text),
                        padding=ft.padding.only(left=10, top=5)
                    )
                ]),
                padding=15
            ),
            ft.Row([
                ft.Container(expand=True),
                ft.ElevatedButton(
                    "导出结果",
                    icon=ft.icons.DOWNLOAD,
                    on_click=self._save_analysis
                )
            ], alignment=ft.MainAxisAlignment.END)
        ])
        
        # 更新结果卡片内容
        self.result_card.content = ft.Container(
            content=result_content,
            padding=15
        )
        
        # 确保更新UI
        if self.page.controls and self.page.controls[0] == self.main_view:
            self.page.update()

    def _save_analysis(self, e=None):
        """保存分析结果"""
        if not self.latest_result:
            self._log("没有可保存的分析结果")
            return
            
        try:
            # 创建保存目录
            save_dir = Path.home() / "学习助手分析结果"
            save_dir.mkdir(exist_ok=True)
            
            # 创建文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            material_type = self.latest_result.get('material_type', '未知类型')
            filename = f"{timestamp}_{material_type}.txt"
            
            # 保存内容
            with open(save_dir / filename, 'w', encoding='utf-8') as f:
                f.write(f"材料类型: {material_type}\n\n")
                f.write(f"分析内容:\n{self.latest_result.get('analysis', '无分析结果')}")
            
            self._log(f"分析结果已保存至: {save_dir / filename}")
            self._show_info_dialog("保存成功", f"分析结果已保存至:\n{save_dir / filename}")
        except Exception as e:
            self._log(f"保存分析结果失败: {str(e)}")

    def _log(self, message):
        """记录日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.controls.append(ft.Text(f"[{timestamp}] {message}"))
        
        # 限制日志条目数量，避免内存问题
        if len(self.log_view.controls) > 100:
            self.log_view.controls.pop(0)
            
        # 只有在设置页面可见时才更新UI
        if self.page.controls and self.page.controls[0] == self.settings_view:
            self.page.update()
            
        logging.info(message)

    def _on_app_close(self, e):
        """应用关闭时的清理操作"""
        # 停止自动捕获
        self._cancel_auto_capture()
        
        # 停止UI更新定时器
        if self.update_ui_timer:
            self.update_ui_timer.cancel()
        
        # 停止预览
        with self.preview_lock:
            self.is_previewing = False
        
        # 停止相机
        if self.camera_mgr:
            try:
                self.camera_mgr.stop()
            except Exception as e:
                logging.error(f"关闭相机错误: {str(e)}")
        
        # 保存配置
        self._save_config()
        
        # 确认关闭
        return True
        
    def _save_config(self, e=None):
        """保存配置"""
        try:
            new_config = {
                "api_key": self.api_key_input.value,
                "api_endpoint": self.endpoint_input.value,
                "auto_capture_interval": self.interval_input.value
            }
            
            self.cfg.save_config(new_config)
            
            # 更新API客户端
            if self.api_client:
                self.api_client = DeepSeekClient(self.cfg)
                
            self._log("配置已保存")
            
            if e:  # 只有在用户点击保存按钮时才显示确认对话框
                self._show_info_dialog("保存成功", "应用配置已保存")
                
        except Exception as e:
            self._log(f"保存配置失败: {str(e)}")
            if e:
                self._show_error_dialog("保存失败", f"无法保存配置: {str(e)}")

    def _update_camera_status(self, connected, error_msg=None):
        """更新摄像头状态显示"""
        if connected:
            self.status_indicator.visible = False
        else:
            self.status_indicator.content = ft.Text(
                error_msg or "摄像头未连接", 
                size=20,
                color=ft.colors.WHITE
            )
            self.status_indicator.visible = True
            
        # 状态变化时更新UI
        if self.status_indicator.visible != (not connected):
            self.page.update()
    
    def _show_error_dialog(self, title, message):
        """显示错误对话框"""
        dialog = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Text(message),
            actions=[
                ft.TextButton("确定", on_click=lambda e: self.page.close_dialog())
            ]
        )
        self.page.dialog = dialog
        self.page.dialog.open = True
        self.page.update()
        
    def _show_info_dialog(self, title, message):
        """显示信息对话框"""
        dialog = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Text(message),
            actions=[
                ft.TextButton("确定", on_click=lambda e: self.page.close_dialog())
            ]
        )
        self.page.dialog = dialog
        self.page.dialog.open = True
        self.page.update()

def main():
    ft.app(target=StudyAssistantApp)

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path.home() / ".study_assistant" / "app.log"),
            logging.StreamHandler()
        ]
    )
    
    # 启动应用
    main()