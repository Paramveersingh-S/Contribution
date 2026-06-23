import urllib.request
import json

def fetch_issues(repo):
    url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=10"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"--- Issues for {repo} ---")
            for i in data:
                if "pull_request" not in i:
                    print(f"Issue #{i['number']}: {i['title']}")
    except Exception as e:
        print(f"Failed to fetch {repo}: {e}")

fetch_issues("google-deepmind/gemma")
fetch_issues("google-gemma/cookbook")
