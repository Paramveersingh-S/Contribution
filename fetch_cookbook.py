import urllib.request
import json

req = urllib.request.Request("https://api.github.com/repos/google-gemma/cookbook/issues?state=open")
try:
    with urllib.request.urlopen(req) as response:
        issues = json.loads(response.read())
        print("Cookbook Issues:")
        for i in issues:
            if 'pull_request' not in i:
                print(f"#{i['number']}: {i['title']}")
except Exception as e:
    print(f"Error fetching issues: {e}")
