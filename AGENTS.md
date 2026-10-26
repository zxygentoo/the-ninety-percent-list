# AGENTS.md

## Process

One list item at a time:

1. **Read** the item.
2. **Design** the experiment or build.
3. **Build, run, iterate** in a notebook.
4. **Publish and update the docs:** run the export; in `README.md`, add the notebook to the table and a Build note under its item.

## Layout

- `notebooks/NN.Title.With.Dots.jl`: the notebooks, numbered in reading order. Each is a self-contained [Pluto](https://plutojl.org) file.
- `notebooks/data/`: a download cache shared by the notebooks, ignored by git.
- `scripts/export.jl`: runs every notebook headlessly and writes `site/`, one static HTML page per notebook plus `index.html`. It fails if any cell errors. Its environment is `scripts/Project.toml`.
- `.github/workflows/pages.yml`: builds `site/` on every push to `main` and deploys it to GitHub Pages.

## Export

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=scripts scripts/export.jl                    # writes site/
```

The export runs every notebook in a fresh Pluto session and never writes the notebook files. It exits with an error that names any notebook with errored cells.
