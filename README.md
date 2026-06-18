# Cloak

Project page for *Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA*.

## Run locally

The site is static (no build step). Serve the `docs/` directory and open it in a browser:

```bash
python3 -m http.server 8000 --directory docs
```

Then visit http://localhost:8000.


## Run on a remote machine

```bash
ssh -L 8000:localhost:8000 vilnius
cd src/cloak
python3 -m http.server 8000 --directory docs
```
