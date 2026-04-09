import numpy as np

def green_function(r, k):
    return np.exp(-1j * k * r) / r

def example():
    r = 0.1
    wavelength = 0.03
    k = 2 * np.pi / wavelength
    
    G = green_function(r, k)
    print("Green function value:", G)

if __name__ == "__main__":
    example()