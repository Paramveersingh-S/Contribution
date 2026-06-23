import urllib.request
import json

issue_numbers = [399, 372, 275, 274]

print("Fetching issue details...\n")
for num in issue_numbers:
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/google-gemma/cookbook/issues/{num}")
        with urllib.request.urlopen(req) as response:
            issue = json.loads(response.read())
            print(f"==================================================")
            print(f"ISSUE #{num}: {issue['title']}")
            print(f"==================================================")
            print(issue.get('body', 'No body provided.'))
            print("\n")
    except Exception as e:
        print(f"Error fetching issue #{num}: {e}")
