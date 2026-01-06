# utils/aiTranslate.py
import requests
import json


class BaiduTranslator:
    """百度 AI 翻译 API 封装"""

    def __init__(self, appid: str, api_key: str):
        self.appid = str(appid).strip()
        self.api_key = str(api_key).strip()
        self.url = 'https://fanyi-api.baidu.com/ait/api/aiTextTranslate'

    def translate(self, query: str, from_lang: str = 'en', to_lang: str = 'zh') -> str:
        if not query.strip():
            return ""

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        payload = {
            'appid': self.appid,
            'from': from_lang,
            'to': to_lang,
            'q': query
        }

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=30
            )
            result = response.json()

            if 'error_code' in result:
                return f"翻译错误: {result.get('error_msg', result.get('error_code'))}"

            # 直接从根级别获取 trans_result
            trans_result = result.get('trans_result', [])
            if trans_result:
                return '\n'.join([item.get('dst', '') for item in trans_result])

            return f"无翻译结果: {result}"

        except requests.exceptions.Timeout:
            return "翻译超时，请重试"
        except Exception as e:
            return f"翻译失败: {str(e)}"


if __name__ == '__main__':
    translator = BaiduTranslator(
        appid='20260105002533609',
        api_key='8qBw_d5do3deol13gd3crgg7g'
    )
    result = translator.translate("Hello World! This is a test.", from_lang='en', to_lang='zh')
    print(result)