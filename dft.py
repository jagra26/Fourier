import math
import cmath
import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt

def factor (x, N, k, n):
  return x*cmath.exp(-(1j*2*math.pi*k*n)/N)

def dft (signal, inverse=False):
  if inverse:
    way = -1
  else:
    way = 1
  size = len(signal)
  out = np.zeros(size, dtype = 'complex_')
  with alive_bar(size*size) as bar:
    for i in range(size):
        for j in range(size):
            out[i] += factor(signal[j], size, i, j*way)
            bar()
  if inverse:
    return np.round(out)/size
  else:
    return np.round(out)


def plot(original, transformed, fname):
    plt.figure()
    plt.subplot(311)
    plt.stem(range(len(original)), original, linefmt='-')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.title('Input sequence')
    f = range(len(transformed))
    m = np.absolute(transformed)
    plt.subplot(312)
    plt.stem(f, m, linefmt='-')
    plt.xlabel('Frequency')
    plt.ylabel('|X(k)|')
    plt.title('Magnitude Response')
    plt.subplot(313)
    p = np.angle(transformed)
    plt.stem(f, p, linefmt='-')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase Response')
    plt.savefig(fname)
    plt.show()

sig = np.array([6, 8, 5, 4, 5, 6])
sig_dft = dft(sig)
plot(sig, sig_dft, "dft_py.png")