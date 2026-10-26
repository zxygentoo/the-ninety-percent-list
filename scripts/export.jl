# Run every notebook in notebooks/ and export it as a static HTML page into site/, with an index page.
# Fails if any cell of any notebook errors, so a broken notebook is never published.
#
#     julia --project=scripts scripts/export.jl

import PlutoSliderServer

const ROOT = dirname(@__DIR__)
const NOTEBOOKS = joinpath(ROOT, "notebooks")
const SITE = joinpath(ROOT, "site")
const REPOSITORY = "https://github.com/zxygentoo/the-ninety-percent-list"

"Number of errored cells in each exported notebook, as `path => count`."
errored_cells(sessions) = [s.path => count(r -> r["errored"], values(s.run.original_state["cell_results"])) for s in sessions]

"Title and subtitle of a notebook: the `#` heading of its first markdown cell, and the italic line under it."
function heading(path)
    m = match(r"md\"\"\"\s*\n# ([^\n]+)\n\s*\n\*([^\n]+)\*", read(joinpath(NOTEBOOKS, path), String))
    m === nothing ? (title = basename(path), subtitle = "") : (title = m[1], subtitle = m[2])
end

escape(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

function entry(path)
    (; title, subtitle) = heading(path)
    page = replace(path, r"\.jl$" => ".html")
    """
          <li>
            <a href="$(escape(page))">$(escape(title))</a>
            <p>$(escape(subtitle))</p>
          </li>"""
end

function index_page(paths)
    """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>The Ninety Percent List</title>
      <style>
        :root { --surface: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --rule: #e4e3de; --link: #1c5cab; }
        @media (prefers-color-scheme: dark) {
          :root { --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --rule: #383835; --link: #86b6ef; }
        }
        body { margin: 0; background: var(--surface); color: var(--text);
               font: 17px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
        main { max-width: 40rem; margin: 0 auto; padding: 4rem 1rem; }
        h1 { font-size: 2rem; margin: 0 0 0.5rem; }
        .lede, footer { color: var(--muted); }
        .lede a { color: var(--link); }
        ol { list-style: none; padding: 0; margin: 2.5rem 0; }
        li { padding: 1.25rem 0; border-top: 1px solid var(--rule); }
        li a { color: var(--link); font-size: 1.2rem; font-weight: 600; text-decoration: none; }
        li a:hover { text-decoration: underline; }
        li p { margin: 0.25rem 0 0; color: var(--muted); }
        footer a { color: inherit; }
      </style>
    </head>
    <body>
      <main>
        <h1>The Ninety Percent List</h1>
        <p class="lede">A rebuild of the reading list Ilya Sutskever gave John Carmack, worked through by reimplementing it: one notebook per item, to read like an essay and run like an experiment. <a href="$REPOSITORY#the-list">The full list</a>.</p>
        <ol>
    $(join(entry.(paths), "\n"))
        </ol>
        <footer>Source and instructions for running the notebooks: <a href="$REPOSITORY">$REPOSITORY</a></footer>
      </main>
    </body>
    </html>
    """
end

function main()
    paths = sort(filter(endswith(".jl"), readdir(NOTEBOOKS)))    # relative to NOTEBOOKS, as the exporter expects
    failures = Pair{String,Int}[]
    PlutoSliderServer.export_directory(NOTEBOOKS;
        notebook_paths = paths,
        Export_output_dir = SITE,
        Export_create_index = false,
        Export_baked_notebookfile = false,    # publish the .jl next to the page, so its "Edit or run" guide can link to it
        on_ready = result -> append!(failures, filter(p -> last(p) > 0, errored_cells(result.notebook_sessions))),
    )
    write(joinpath(SITE, "index.html"), index_page(paths))
    isempty(failures) || error("cells errored: " * join(["$path ($n)" for (path, n) in failures], ", "))
    println("exported $(length(paths)) notebook(s) to $SITE")
end

main()
