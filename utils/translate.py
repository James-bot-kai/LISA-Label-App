# utils/translate.py
import requests
import random
from hashlib import md5


class BaiduTranslator:
    """百度翻译 API 封装"""

    def __init__(self, appid: str, appkey: str):
        self.appid = str(appid).strip()
        self.appkey = str(appkey).strip()
        self.endpoint = 'http://api.fanyi.baidu.com'
        self.path = '/api/trans/vip/translate'
        self.url = self.endpoint + self.path

    def _make_md5(self, s: str, encoding: str = 'utf-8') -> str:
        return md5(s.encode(encoding)).hexdigest()

    def translate(self, query: str, from_lang: str = 'en', to_lang: str = 'zh') -> str:
        if not query.strip():
            return ""

        # 生成 salt（与官方文档一致）
        salt = random.randint(32768, 65536)

        # 签名字符串: appid + query + salt + appkey（salt 必须转 str）
        sign_str = self.appid + query + str(salt) + self.appkey
        sign = self._make_md5(sign_str)

        # 构建请求（与官方文档完全一致）
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'appid': self.appid,
            'q': query,
            'from': from_lang,
            'to': to_lang,
            'salt': salt,  # 这里传 int 或 str 都可以
            'sign': sign
        }

        try:
            # 使用 params 传参（官方文档用法）
            response = requests.post(self.url, params=payload, headers=headers, timeout=10)
            result = response.json()

            if 'error_code' in result:
                return f"翻译错误: {result.get('error_msg', '未知错误')}"

            trans_result = result.get('trans_result', [])
            translated_lines = [item.get('dst', '') for item in trans_result]
            return '\n'.join(translated_lines)

        except requests.exceptions.Timeout:
            return "翻译超时，请重试"
        except requests.exceptions.RequestException as e:
            return f"网络错误: {str(e)}"
        except Exception as e:
            return f"翻译失败: {str(e)}"


if __name__ == '__main__':
    translator = BaiduTranslator(
        appid='20260105002533609',
        appkey='fIFodJNEMlRAetRHM8Ec'
    )
    test_text = "Hello World! This is 1st paragraph.\nThis is 2nd paragraph."
    translated = translator.translate(test_text, from_lang='en', to_lang='zh')
    print(f"原文: {test_text}\n译文: {translated}")