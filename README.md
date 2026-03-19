# ScaledPolynomials

[![Build Status](https://github.com/Manarom/ScaledPolynomials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Manarom/ScaledPolynomials.jl/actions/workflows/CI.yml?query=branch%3Amain)

# ScaledPolynomials.jl

This is a small utility package for working with polynomials (Chebyshev, Legendre, Bernstein, etc.) on arbitrary intervals $[x_{min}, x_{max}]$.

The main goal is to eliminate manual coordinate mapping between your working range and the standard interval (e.g., $[-1, 1]$).

### Features:
* **Automatic Mapping**: The `ScaledPolynomial` wrapper handles $x \to \xi$ transformation during evaluation.
* **Minimal Dependencies**: Built only on `LinearAlgebra` and `StaticArrays`. No heavy external packages.
* **Performance**: Leverages `StaticArrays` for coefficients and the Clenshaw scheme for stable evaluation
(check tests for performance comparison).
* **Bases**: Supports Chebyshev (`ChebPoly`), Legendre (`LegPoly`), Bernstein, Standard, and Trigonometric polynomials.
* **Utilities**: Quick derivative computation and basic `polyfit` via Vandermonde matrices.

### Installation

Since this package is not registered in the General registry, you can install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Manarom/ScaledPolynomials.jl.git")
```

### Quick Start

```julia
using ScaledPolynomials, StaticArrays

# Create a Bernstein basis polynomial on a custom interval 0.0 - 10.0

p = ScaledPolynomial(BernsteinSymPoly, (1.0, 0.5, 0.1), 0.0, 10.0)

# Evaluate at a specific point
val = p(5.0) 

# Compute the derivative (returns a new ScaledPolynomial)
dp = derivative(p) 
```