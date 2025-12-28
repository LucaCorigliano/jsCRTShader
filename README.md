# crtShader (WebGPU)

HDR CRT shader pipeline for JavaScript emulators using WebGPU.

## Run locally
1. `python3 -m http.server 8080`
2. Open `http://localhost:8080/index.html`

Notes:
- If you clone the repo, run `git submodule update --init --recursive`.
- WebGPU requires a recent Chromium-based browser and OS HDR enabled.
- `file://` will fail due to browser security restrictions.
- `assets/bios/` and `assets/isos/` are local-only and ignored by git.
