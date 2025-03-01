import requests
import logging

class DeepSeekClient:
    def __init__(self, config):
        self.config = config
        self.base_url = config.get("api_endpoint", "https://api.deepseek.com/v1")
        self.timeout = 15

    def analyze_image(self, image_data):
        """分析学习材料"""
        endpoint = f"{self.base_url}/analyze"
        headers = {
            "Authorization": f"Bearer {self.config.get('api_key', '')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "image": image_data,
            "parameters": {
                "analyze_type": "learning_material",
                "detail_level": "high"
            }
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API请求失败: {str(e)}")
            raise RuntimeError("分析服务暂时不可用") from e