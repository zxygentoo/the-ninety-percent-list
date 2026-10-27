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
4. **Publish and update the docs:** run the export; in `README.md`, add the notebook and its Build note to the table, link it under its item, and update the progress.

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
- `_scratch/`: temporary scripts used while building a notebook, ignored by git.
- `scripts/export.jl`: runs every notebook headlessly and writes `site/`, one static HTML page per notebook plus `index.html`. It fails if any cell errors. Its environment is `scripts/Project.toml`.
- `.github/workflows/pages.yml`: builds `site/` on every push to `main` and deploys it to GitHub Pages.

## Export

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=scripts scripts/export.jl                    # writes site/
```

The export runs every notebook in a fresh Pluto session and never writes the notebook files. It exits with an error that names any notebook with errored cells.
