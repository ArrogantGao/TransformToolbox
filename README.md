# TransformToolbox

A simple tool box for math transforms.

## Non-uniform discrete Fourier transform

For $M$ sources, the type-1 NUDFT is computed as:
$$
f(k_x, k_y, k_z) = \sum_{i=1}^{M} c_i \exp(i(k_x x_i + k_y y_i + k_z z_i))\\
- N_j/2 \leq k_j \leq (N_j - 1)/2 \quad \text{for } j \in \{x, y, z\}
$$

The `nudft3d1_direct` function computes each Fourier mode value directly, requiring $O(M N_1 N_2 N_3)$ time for exponential calculations.

The `nudft3d1` function pre-computes phase factors for each source by iterating in each dimension and storing them as arrays. It then computes the value on each Fourier mode by multiplying these phase factors. This approach requires $O(M)$ time for exponential calculations and $O(M N_1 N_2 N_3)$ time for multiplications and additions, with linear memory access $M$ times.

The implementation in [finufft/dirft3d.hpp](https://github.com/flatironinstitute/finufft/blob/0d7360c798023c6c508c84eba06da0d6365fa8a9/test/utils/dirft3d.hpp#L26), referred to as `nudft3d1_iterate`, iteratively updates phase factors during a triple loop instead of storing them in an array. Benchmarking shows it is approximately 8 times slower than `nudft3d1`, likely because the inner loop of `nudft3d1` benefits from automatic vectorization by the compiler.

In `nudft3d1_s2`, an attempt was made to optimize cache usage by iterating through the $k_z$ index first, then the particle indices. This means that for each $k_z$, a plane of $f[:, :, k_z]$ is processed by looping over $M$, $k_y$, and $k_x$. Since this plane is contiguous in memory and of size $N_1 \times N_2$, it was hypothesized to be more cache-friendly. However, its performance was worse than `nudft3d1`, suggesting that cache utilization was not the primary bottleneck.

The `nudft3d1_s1` function is similar to `nudft3d1_s2`, but it fixes both $k_z$ and $k_y$ first, then loops over $M$ and $k_x$. In this case, the memory footprint is $N_1$. Both `nudft3d1_s1` and `nudft3d1_s2` require more exponential calculations than `nudft3d1`.