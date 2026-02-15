#!/usr/bin/env python3
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
url_file = repo / "media" / "demo-video.url"
readme = repo / "README.md"

url = url_file.read_text(encoding="utf-8").strip()
if not url:
    raise SystemExit("media/demo-video.url is empty")

start = "<!-- DEMO_VIDEO_START -->"
end = "<!-- DEMO_VIDEO_END -->"
block = (
    f"{start}\n"
    f'<video src="{url}" controls muted playsinline width="100%"></video>\n'
    f"{end}"
)

content = readme.read_text(encoding="utf-8")
if start in content and end in content:
    s = content.index(start)
    e = content.index(end) + len(end)
    updated = content[:s] + block + content[e:]
else:
    raise SystemExit("README markers not found")

readme.write_text(updated, encoding="utf-8")
print("README demo video updated")
