import urllib.request
import json

url = "https://api.github.com/repos/google-deepmind/gemma/issues/675"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(data['title'])
        print(data['body'])
except Exception as e:
    print(e)
