#!/usr/bin/env python3
from pathlib import Path
import re

repo = Path(__file__).resolve().parents[1]
url_file = repo / "media" / "demo-video.url"
readme = repo / "README.md"

url = url_file.read_text(encoding="utf-8").strip()
if not url:
    raise SystemExit("media/demo-video.url is empty")

thumb = "./media/demo-preview.svg"
m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
if m:
    file_id = m.group(1)
    thumb = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1200"

start = "<!-- DEMO_VIDEO_START -->"
end = "<!-- DEMO_VIDEO_END -->"
block = (
    f"{start}\n"
    f'[![TME-viewer demo]({thumb})]({url})\n\n'
    f'<sub><a href="{url}">Watch demo video</a></sub>\n'
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
print("README demo block updated")
