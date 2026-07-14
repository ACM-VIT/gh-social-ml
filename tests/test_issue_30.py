import urllib.request
import json

def test_fetch_issue_30():
    url = "https://api.github.com/repos/ACM-VIT/gh-social-ml/issues/30"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("\n=== ISSUE 30 BODY START ===")
        print(data.get("body", "No body found"))
        print("=== ISSUE 30 BODY END ===\n")
    assert False
