# Run one notebook headlessly, as the export does, and print every errored cell with its message.
# Also writes each cell's output, in display order, to <output directory>/outputs: text as outputs.txt, images as PNGs.
#
#     julia --project=scripts scripts/run.jl <notebook file name> <output directory> [notebook directory]

import PlutoSliderServer
using Base64

const NOTEBOOKS = length(ARGS) >= 3 ? ARGS[3] : joinpath(dirname(@__DIR__), "notebooks")    # optional third argument: another directory

"Readable text from an HTML fragment: tags dropped, entities for <, >, & restored."
plain(html) = replace(html, r"<[^>]*>" => "", "&lt;" => "<", "&gt;" => ">", "&amp;" => "&", "&quot;" => "\"", r"\n\s*\n+" => "\n")

function dump_outputs(state, dir)
    mkpath(dir)
    open(joinpath(dir, "outputs.txt"), "w") do io
        for (k, id) in enumerate(state["cell_order"])
            out = state["cell_results"][id]["output"]
            mime, body = out["mime"], out["body"]
            embedded = body isa AbstractString ? match(r"data:image/png;base64,\s*([A-Za-z0-9+/=]+)", body) : nothing
            if mime == "image/png" || embedded !== nothing
                file = joinpath(dir, "cell$(lpad(k, 3, '0')).png")
                write(file, embedded !== nothing ? base64decode(embedded[1]) : body isa AbstractVector{UInt8} ? body : base64decode(body))
                println(io, "[$k] image: ", file)
            elseif body isa AbstractString && isempty(strip(plain(body)))
                write(joinpath(dir, "cell$(lpad(k, 3, '0')).raw"), body)
                println(io, "[$k] ", mime, " (no text; raw saved, ", length(body), " chars)")
            elseif body isa AbstractString && !isempty(strip(body))
                println(io, "[$k] ", mime, "\n", strip(plain(body)), "\n")
            end
        end
    end
end

function report(session, dir)
    state = session.run.original_state
    dump_outputs(state, joinpath(dir, "outputs"))
    errored = [(id, r) for (id, r) in state["cell_results"] if r["errored"]]
    println(session.path, ": ", length(errored), " errored cell(s) of ", length(state["cell_results"]))
    for (id, r) in errored
        code = first(split(state["cell_inputs"][id]["code"], '\n'))
        body = r["output"]["body"]
        message = body isa AbstractDict ? get(body, "msg", string(body)) : string(body)
        println("  ", code, "\n    ", first(split(message, '\n')))
    end
end

PlutoSliderServer.export_directory(NOTEBOOKS;
    notebook_paths = [ARGS[1]],
    Export_output_dir = ARGS[2],
    Export_create_index = false,
    Export_baked_notebookfile = false,
    on_ready = result -> foreach(s -> report(s, ARGS[2]), result.notebook_sessions),
)
