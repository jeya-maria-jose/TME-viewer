# TME-viewer

Local web viewer for tissue image viewing of H&E and IHC/mIF.

## Demo
<!-- DEMO_VIDEO_START -->
[![TME-viewer demo](https://drive.google.com/thumbnail?id=1FZRKPvkzHbVtDJnvB13a-mBqelZFJE2j&sz=w1200)](https://drive.google.com/file/d/1FZRKPvkzHbVtDJnvB13a-mBqelZFJE2j/view?usp=sharing)

<sub><a href="https://drive.google.com/file/d/1FZRKPvkzHbVtDJnvB13a-mBqelZFJE2j/view?usp=sharing">Watch demo video</a></sub>
<!-- DEMO_VIDEO_END -->


## What it supports
- Side-by-side visualization:
  - Left: H&E image
  - Right: active Image tab
- Dynamic Image windows:
  - Add any number of image paths (`+ Add Image Tab`)
  - All loaded images are shown simultaneously (Image 1, Image 2, ...)
  - Each Image can be mIF, IHC, or activation map
- Image inputs can be:
  - mIF / activation-map OME-TIFF channel stacks
  - IHC RGB images (`.tif/.tiff/.ndpi`, etc.)
- Zoom + pan:
  - Mouse wheel zoom
  - Drag pan
  - Synchronized between H&E and signal panel


## Quick start
```bash
cd TME-viewer
pip install -r requirements.txt
python app.py
```
Open: `http://127.0.0.1:8765`

## OpenSlide notes
This app can run without OpenSlide, but OpenSlide improves compatibility with large WSI formats (especially some `.ndpi`).

### macOS
```bash
brew install openslide
pip install openslide-python
```

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y libopenslide0 openslide-tools
pip install openslide-python
```

## Run on VM / server
```bash
python app.py --host 0.0.0.0 --port 8765
```
Then open: `http://<vm-ip>:8765`

Make sure:
- Port `8765` is open in firewall/security group.
- File paths in the UI exist on that VM filesystem.
