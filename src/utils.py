import cv2
import numpy as np

def check_image_quality(image, min_sharpness=100, min_brightness=50, max_brightness=200):
    """
    图像质量检测
    :return: (是否合格, 清晰度, 亮度)
    """
    # 计算清晰度 (拉普拉斯方差)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 计算亮度 (HSV V通道平均值)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[...,2])
    
    # 评估结果
    is_qualified = (
        sharpness >= min_sharpness and 
        min_brightness <= brightness <= max_brightness
    )
    
    return is_qualified, sharpness, brightness