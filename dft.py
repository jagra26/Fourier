import math
import cmath
import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt

def factor (x, N, k, n):
  return x*cmath.exp(-(1j*2*math.pi*k*n)/N)

def wb(N, k):
  return cmath.exp((-2*math.pi*1j*k)/N)

def wc(N, k):
  return cmath.exp((-2*math.pi*1j*2*k)/N)

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

def check(base, number):
  result = math.log(number, base)
  integer = int(result)
  if result - integer == 0:
    return 0
  else:
    print("integer ",integer)
    pow = 1+integer
    return (2**pow) - number

def fft2(signal):
  N = len(signal)
  if N == 1:
    return signal
  Aout = []
  Bout= []
  P = []
  Q = []
  for j in range(N):
    if j%2 == 0: 
      P.append(signal[j])
    else:
      Q.append(signal[j])
  A = fft2(P)
  B =  fft2(Q)
  for j in range(min(len(A), len(B))):
    Aout.append(A[j] + wb(N, j)*B[j])
    Bout.append(A[j] - wb(N, j)*B[j])
  out = Aout+Bout
  return np.round(out)

def fft3(signal):
  N = len(signal)
  if N == 1:
    return signal
  P = []
  Q = []
  R = []
  A_out = []
  B_out = []
  C_out = []
  for k in range(N):
    if k % 3 == 0:
      P.append(signal[k])
    elif k % 3 == 1:
      Q.append(signal[k])
    else:
      R.append(signal[k])
  A_line = fft3(P)
  B_Line = fft3(Q)
  C_line = fft3(R)
  
  for k in range(min(len(B_Line), len(C_line))):
    B_Line[k] *= wb(N, k)
    C_line[k] *= wc(N, k)
    A_out.append(A_line[k] + B_Line[k] + C_line[k])
    B_out.append(A_line[k]+complex((-1/2),(-(3**(1/2))/2))*B_Line[k]+complex((-1/2),((3**(1/2))/2))*C_line[k])
    C_out.append(A_line[k]+complex((-1/2),((3**(1/2))/2))*B_Line[k]+complex((-1/2),(-(3**(1/2))/2))*C_line[k])
  out = A_out + B_out + C_out
  return np.round(out)

def plot(original, transformed, inverse, fname):
    plt.figure()
    plt.subplot(411)
    plt.stem(range(len(original)), original, linefmt='-')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.title('Input sequence')
    f = range(len(transformed))
    m = np.absolute(transformed)
    plt.subplot(412)
    plt.stem(f, m, linefmt='-')
    plt.xlabel('Frequency')
    plt.ylabel('|X(k)|')
    plt.title('Magnitude Response')
    plt.subplot(413)
    p = np.angle(transformed)
    plt.stem(f, p, linefmt='-')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase Response')
    plt.subplot(414)
    plt.stem(range(len(inverse)), inverse, linefmt='-')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.title('Inverse transform sequence')
    plt.savefig(fname)

sig = np.array([6, 8, 5, 4, 5, 6])
sig_dft = dft(sig)
inv_dft = dft(sig_dft, inverse=True)
pad = np.zeros(2)
sig_dft2 = dft((np.concatenate((sig, pad), axis=0)))
sig_fft2 = fft2(np.concatenate((sig, pad), axis=0))
pad = np.zeros(3)
sig_dft3 = dft((np.concatenate((sig, pad), axis=0)))
sig_fft3 = fft3(np.concatenate((sig, pad), axis=0))
inv_fft = dft(sig_fft2, inverse=True)
print("DFT: ", sig_dft)
print("DFT with padding 2: ", sig_dft2)
print("radix 2: ", sig_fft2)
print("DFT with padding 3: ", sig_dft3)
print("radix 3: ", sig_fft3)
print("IDFT: ", inv_fft)
plot(sig, sig_dft, inv_dft, "dft_py.png")
plot(sig, sig_fft2, inv_dft, "fft2_py.png")
plot(sig, sig_fft3, inv_dft, "fft3_py.png")