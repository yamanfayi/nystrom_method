# nystrom_method
EFIE_NYSTROM


# Nyström Method for EFIE

This repository presents a Python implementation of the Nyström method 
for solving the Electric Field Integral Equation (EFIE) on PEC surfaces.

## Overview
The Nyström method is used to discretize surface integral equations 
using numerical quadrature techniques.

## Features
- Green's function implementation
- Numerical integration (Gaussian quadrature)
- Separation of singular and non-singular terms (planned)

## Example
```python
r = 0.1
k = 2 * np.pi / 0.03
G = np.exp(-1j * k * r) / r
