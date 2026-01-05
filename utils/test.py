# test_translate.py
import requests
import random
import json
from hashlib import md5

appid = '20260105002533609'
appkey = 'fIFodJNEMlRAetRHM8Ec'

query = 'apple'  # 使用简单单词测试
salt = 65478     # 固定 salt 便于对比

# 打印调试信息
sign_str = appid + query + str(salt) + appkey
sign = md5(sign_str.encode('utf-8')).hexdigest()

print("=== 调试信息 ===")
print(f"appid: {repr(appid)}")
print(f"appkey: {repr(appkey)}")
print(f"query: {repr(query)}")
print(f"salt: {salt}")
print(f"签名原文: {repr(sign_str)}")
print(f"签名结果: {sign}")
print()

# 方法1: GET 请求
url = f"http://api.fanyi.baidu.com/api/trans/vip/translate?q={query}&from=en&to=zh&appid={appid}&salt={salt}&sign={sign}"
print(f"GET URL: {url}")
r1 = requests.get(url)
print(f"GET 结果: {r1.json()}")
print()

# 方法2: POST 请求
payload = {'appid': appid, 'q': query, 'from': 'en', 'to': 'zh', 'salt': salt, 'sign': sign}
r2 = requests.post("http://api.fanyi.baidu.com/api/trans/vip/translate", data=payload)
print(f"POST 结果: {r2.json()}")