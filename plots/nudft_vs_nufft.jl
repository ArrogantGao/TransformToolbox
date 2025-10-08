using CSV, DataFrames, CairoMakie, LaTeXStrings

df = CSV.read("../data/nudft_vs_nufft_time.csv", DataFrame)

Ms = unique(df.M)
Ns = unique(df.N)

fig = Figure(size=(800, 600), fontsize = 20)
ax1 = Axis(fig[1, 1], xlabel="number of particles",ylabel=L"time ($ns$)", title="N = 8", xscale=log2, yscale=log10)

ax2 = Axis(fig[1, 2], xlabel="number of particles",ylabel=L"time ($ns$)", title="N = 16", xscale=log2, yscale=log10)

ax3 = Axis(fig[2, 1], xlabel="number of particles",ylabel=L"time ($ns$)", title="N = 32", xscale=log2, yscale=log10)

ax4 = Axis(fig[2, 2], xlabel="number of particles",ylabel=L"time ($ns$)", title="N = 64", xscale=log2, yscale=log10)

axs = [ax1, ax2, ax3, ax4]

for (i, N) in enumerate(Ns)
    scatter!(axs[i],df[df.N .== N, :M], df[df.N .== N, :finufft], label="Finufft", color="red", markersize=10)
    scatter!(axs[i],df[df.N .== N, :M], df[df.N .== N, :nudft_s3], label="nudft", color="blue", markersize=10)
    axislegend(axs[i], position=:rb)
end

save("nudft_vs_nufft.png", fig)