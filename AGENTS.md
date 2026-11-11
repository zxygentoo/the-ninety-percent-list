# AGENTS.md

## Rules

- **NEVER commit unless explicitly told to.** Always ask for review first.

## Goal

This is a learning project. Work with the user to:

- Maximize understanding per unit of effort, but don't minimize struggle: the struggle is the learning.
- Focus on the WHY: the core ideas, and the concepts and engineering practices that matter most.
- If it's a claim or concept, build a targeted, ideally minimal, experiment that verifies it and helps the user build good intuition and mental models.
- If it's an implementation practice, guide the user through it without giving away the solution, using as few libraries and frameworks as possible.

## Process

One list item at a time:

1. **Read** the item, and research further if the material warrants it.
2. **Design** the experiment or build.
3. **Build, run, iterate** in a notebook, or first in a temporary script and then in the notebook.
4. **Publish and update the docs:** run the export; in `README.md`, add the notebook to the table with its status and update the progress.

## Style

### Code

- Strongly prefer a functional style; Julia supports it well.
- Favor clarity and readability over performance. Don't be afraid to use more cells if it helps.
- Strive to make it beautiful and pleasant to read.

### Prose

- These are essays more than papers, so write for readability.
- When the math gets abstract, use a metaphor or an illustration to help the reader understand.
- Don't lay down walls of text. Break things up, and let the prose breathe and flow.
- Elegance comes from clarity and simplicity.

## Layout

- `notebooks/NN.Title.With.Dots.jl`: the notebooks, numbered in reading order. Each is a self-contained [Pluto](https://plutojl.org) file.
- `notebooks/data/`: a download cache shared by the notebooks, ignored by git.
- `_scratch/NN/`: temporary scripts and drafts used while building notebook `NN`, one folder per notebook, ignored by git. Name the files without the number, which the folder already carries: `_scratch/04/noisy.jl`, not `_scratch/04.noisy.jl`.
- `scripts/export.jl`: runs every notebook headlessly and writes `site/`, one static HTML page per notebook plus `index.html`. It fails if any cell errors. Its environment is `scripts/Project.toml`.
- `scripts/run.jl`: runs one notebook headlessly, as the export does, prints any errored cells, and writes every cell's output to a directory: text to `outputs.txt`, images as PNGs. Use it to check a notebook's numbers and figures without opening Pluto. Same environment.
- `.github/workflows/pages.yml`: builds `site/` on every push to `main` and deploys it to GitHub Pages.

## Export

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=scripts scripts/export.jl                    # writes site/
julia --project=scripts scripts/run.jl NN.Title.jl <dir>     # one notebook, its outputs in <dir>/outputs
```

The export runs every notebook in a fresh Pluto session and never writes the notebook files. It exits with an error that names any notebook with errored cells.

## Julia version

Julia is pinned to **1.13.3** in two places, and they must stay in sync:

- locally, by a juliaup override on this directory (`juliaup override set 1.13.3`), so start Julia and Pluto from here;
- in CI, by `version` in `.github/workflows/pages.yml`, which builds the published pages.

Julia doesn't promise the same random numbers across versions, so a different version can change every dataset and every number a notebook prints. To upgrade, bump both together, rerun the export, and re-check any prose written against specific results.
