# docs

Project website. Serve locally with:

```bash
ssh -L 8000:localhost:8000 vilnius # If sshing into remote
uv run python -m http.server 8000 --directory docs
```
`

## Converting figures

Figures are stored as SVG. To convert a PDF to SVG (requires poppler's `pdftocairo`):

```bash
pdftocairo -svg assets/overview.pdf assets/overview.svg
```
