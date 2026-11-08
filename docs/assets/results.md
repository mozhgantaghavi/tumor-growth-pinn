**Noise σ = 0.0** (first 100 test cases, identical for all methods)

| Method | MAPE D | MAPE ρ | median APE D | median APE ρ | R² (log D) | R² (log ρ) | time / case |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN (3-member ensemble) | 2.1% | 1.2% | 1.7% | 1.0% | 0.998 | 0.999 | 1.6 ms |
| Spectral least squares | 0.5% | 0.1% | 0.2% | 0.0% | 1.000 | 1.000 | 0.1 s |

**Noise σ = 0.02** (first 24 test cases, identical for all methods)

| Method | MAPE D | MAPE ρ | median APE D | median APE ρ | R² (log D) | R² (log ρ) | time / case |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN (3-member ensemble) | 3.2% | 0.9% | 3.3% | 0.6% | 0.997 | 1.000 | 1.6 ms |
| Spectral least squares | 8.6% | 3.1% | 3.9% | 2.1% | 0.962 | 0.990 | 0.1 s |
| PINN | 13.2% | 1.1% | 8.8% | 0.9% | 0.919 | 0.999 | 213.3 s |
| CNN-init PINN | 2.8% | 0.6% | 1.5% | 0.5% | 0.997 | 1.000 | 70.1 s |

**Noise σ = 0.05** (first 24 test cases, identical for all methods)

| Method | MAPE D | MAPE ρ | median APE D | median APE ρ | R² (log D) | R² (log ρ) | time / case |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN (3-member ensemble) | 3.5% | 1.7% | 2.6% | 1.2% | 0.996 | 0.998 | 1.5 ms |
| Spectral least squares | 19.4% | 10.2% | 11.9% | 8.0% | 0.864 | 0.910 | 0.1 s |
| PINN | 11.9% | 1.3% | 9.0% | 0.7% | 0.937 | 0.999 | 215.6 s |
| CNN-init PINN | 3.0% | 1.3% | 2.7% | 0.9% | 0.997 | 0.999 | 70.0 s |

**Noise σ = 0.1** (first 100 test cases, identical for all methods)

| Method | MAPE D | MAPE ρ | median APE D | median APE ρ | R² (log D) | R² (log ρ) | time / case |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN (3-member ensemble) | 5.8% | 2.4% | 4.4% | 1.7% | 0.987 | 0.996 | 1.5 ms |
| Spectral least squares | 24.4% | 17.1% | 15.9% | 15.6% | 0.806 | 0.811 | 0.1 s |
