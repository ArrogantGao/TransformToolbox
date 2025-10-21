using CSV, DataFrames, CairoMakie, LaTeXStrings

df = CSV.read("../data/nufft_vs_nudft3d1.csv", DataFrame)

function parse_name(name)
    m = match(r"BM_(\w+)/(\d+)/(\d+)", name)
    if m === nothing
        return ("Unknown", missing, missing)
    else
        return (m.captures[1], parse(Int, m.captures[2]), parse(Int, m.captures[3]))
    end
end

df_parsed = DataFrame(
    test = String[],
    M = Int[],
    N = Int[],
    cpu_time = Float64[],
)

for row in eachrow(df)
    (t, M, N) = parse_name(row.name)
    if !ismissing(M)
        push!(df_parsed, (t, M, N, row.cpu_time))
    end
end

Ns = sort(unique(df_parsed.N))
Ms = sort(unique(df_parsed.M))

fig = Figure(size=(1000, 800), fontsize=20)

titles = ["N = 8", "N = 16", "N = 32", "N = 64"]
axs = [
    Axis(fig[1, 1], xlabel="number of particles", ylabel=L"time\ (μs)", title=titles[1],
         xscale=log2, yscale=log10),
    Axis(fig[1, 2], xlabel="number of particles", ylabel=L"time\ (μs)", title=titles[2],
         xscale=log2, yscale=log10),
    Axis(fig[2, 1], xlabel="number of particles", ylabel=L"time\ (μs)", title=titles[3],
         xscale=log2, yscale=log10),
    Axis(fig[2, 2], xlabel="number of particles", ylabel=L"time\ (μs)", title=titles[4],
         xscale=log2, yscale=log10),
]

colors = Dict(
    "Finufft_Setpts" => :red,
    "Finufft_Execute" => :orange,
    "NUDFT3D1" => :blue,
)

labels = Dict(
    "Finufft_Setpts" => "Finufft Setpts",
    "Finufft_Execute" => "Finufft Execute",
    "NUDFT3D1" => "NUDFT3D1",
)

for (i, N) in enumerate(Ns)
    dfN = df_parsed[df_parsed.N .== N, :]
    for (tname, color) in colors
        if tname == "Finufft_Setpts"
            df_t = dfN[dfN.test .== "Finufft_Setpts", :]
        elseif tname == "Finufft_Execute"
            df_t = dfN[dfN.test .== "Finufft_Execute", :]
        elseif tname == "NUDFT3D1"
            df_t = dfN[dfN.test .== "NUDFT3D1", :]
        else
            continue
        end
        if nrow(df_t) > 0
            scatter!(axs[i], df_t.M, df_t.cpu_time, label=labels[tname], color=color, markersize=10)
        end
    end
end

axislegend(axs[1], position=:lt)

save("nufft_breakdown.png", fig)
