# Project website

Serve this directory as the project website root. The setup guide is at `setup/index.html` and shares the project navigation and assets. The repository README links here for deployment instructions.

For local review: `python -m http.server 8769 --directory docs`.

GitHub Pages publishes `main:/docs` at https://tml.stanford.edu/cloak/.

To rebuild the guides from repository source: `python tools/setup_site/build_guides.py`. To regenerate STL thumbnails: `python tools/setup_site/render_mesh_previews.py`. The annotated Desk illustrations are committed assets.
