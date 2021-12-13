import sys
import signal
import os
import jack
import threading
import queue
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
import numpy as np
import time
from numpy import sin, sum, cos, arctan2, pi, real, exp, abs, argmax, sqrt, maximum, concatenate, mean, floor
from numpy.fft import fft, ifft
from numpy.random import randn
from numpy.linalg import eig
from numba import jit, complex128
from scipy.signal import find_peaks
import matplotlib as mpl
import gc
import numpy as np
import time

import collections

gc.disable()

global iters, system_active

print("[INFO]: Initializing DOATracker")
###############
#Se utiliza MUSIC para precompilar la función
###############
#Direcciones de arribo (conocidas, suponemos que no a futuro)
doas = np.array([-30, 40])
#Distancia entre microfonos en metros
d = 0.21
#Presencia de ruido
noise_w = 0.0
#Signal Size in samples
K = 1024
#Vector de frecuencias
fs = 1024


w = np.fft.fftfreq(n=K, d=1/fs)
#Base frequency for signals
freq = np.array([2, 4])
c = 343 # Speed of sound
fs = K #Sampling frequency same as signal size (1 second)
t = np.arange(1,K+1)/K #Time vector (1 second)

N = 3 #Number of microphones
r = 2 #Number of signals in signal sub-space
#original signals
s1 = sin(2*pi*freq[0]*t)
s2 = sin(2*pi*freq[1]*t)
s1_f = fft(s1)
s2_f = fft(s2)

#Simulating microphones
x = s1 + s2
theta_m3 = -arctan2(-d/2,-d*sqrt(3)/2)
y = real(ifft(fft(s1)*exp(-1j*2*pi*w*(-d/c)*sin(doas[0]*pi/180)))) + real(ifft(fft(s2)*exp(-1j*2*pi*w*(-d/c)*sin(doas[1]*pi/180))))
z = real(ifft(fft(s1)*exp(-1j*2*pi*w*(-d/c)*cos(theta_m3 - doas[0]*pi/180)))) + real(ifft(fft(s2)*exp(-1j*2*pi*w*(-d/c)*cos(theta_m3 - doas[1]*pi/180))))

#Adding noise
x = x + randn(K)*noise_w/10
y = y + randn(K)*noise_w/10
z = z + randn(K)*noise_w/10

X = np.vstack([fft(x), fft(y), fft(z)])

this_ws = np.array([2, 4])
#this_ws = np.array([85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225])
r = 2

#MUSIC
num_angles_search = 1800
#Angles to look for orthogonality
angles = np.linspace(-90, 90, num_angles_search)
music_spectrum = np.zeros(num_angles_search)

#Version Numba
@jit(nopython=True)
def MUSIC(X, d, r, w, search_w, num_angles_search, angles, music_spectrum):
    theta_m3 = -arctan2(-d/2, -d*sqrt(3)/2)
    r = 1
    for search_f_idx in this_ws:
        this_X = X[:,search_f_idx].copy().reshape((-1,1))
        #Cov matrix
        R =  this_X @ this_X.T.conj()
        #Eigendecomposition
        D,Q = eig(R)
        #Sorting eigenvalues/vectors
        abs_eig = np.abs(D)
        idx = abs_eig.argsort()[::-1]
        exp_var = abs_eig[idx]/sum(abs_eig)
        found_r = 2
        if exp_var[0] > 0.9:
            found_r = 1
        Q = Q[:,idx]
        #D = D[idx]
        #Getting signal and noise eigenvectors (based on r)
        #Qs = Q[:,:r]
        Qn = Q[:,found_r:].copy()
        #compute steering vectors corresponding to values in angles
        a1 = np.zeros((N,num_angles_search), dtype=np.complex128)
        a1[0,:] = 1 #First mic, no delay
        a1[1,:] = exp(-1j*2*pi*w[search_f_idx]*(-d/c)*sin(angles*pi/180)) #Second mic, delayed one distance
        a1[2,:] = exp(-1j*2*pi*w[search_f_idx]*(-d/c)*cos(theta_m3 - angles*pi/180)) #Third mic, delayed double distance

        #Noise matrix
        Qn_2 = Qn@Qn.T.conj()
        #compute MUSIC spectrum
        r = max(r, found_r)
        for k in range(0, num_angles_search):
            interest = a1[:,k].copy().reshape((1,-1)).transpose()
            music_spectrum[k] = (music_spectrum[k] + abs(1/(interest.conj().T@Qn_2@interest).item()))
    return r


MUSIC(X, d, r, w, this_ws, num_angles_search, angles, music_spectrum)

#######################
##FIN DE PRECOMPILACION
#######################

print("[INFO]: MUSIC precompilation done")

mpl.use("Qt5Agg")

client = jack.Client('DOATracker')

blocksize = client.blocksize
samplerate = client.samplerate
buffersize = 10

qx = queue.Queue(maxsize=blocksize*buffersize)
qy = queue.Queue(maxsize=blocksize*buffersize)
qz = queue.Queue(maxsize=blocksize*buffersize)

timeout = blocksize * buffersize / samplerate


###CAMBIAR ESTO PARA CADA EXPERIMENTO#####
#Direcciones de arribo (conocidas, suponemos que no a futuro)
doas = np.array([-30, 90])
d = 0.18
r = 2
##############
#Vector de frecuencias
num_windows_iter = 2
K = blocksize*num_windows_iter
fs = 48000
w = np.fft.fftfreq(n=K, d=1/fs)
#Arreglando indices
w = w[:K//2+1]
w[-1] *= -1

c = 343 # Speed of sound
#fs = K #Sampling frequency same as signal size (1 second)
t = np.arange(1,K+1)/K #Time vector (1 second)

N = 3 #Number of microphones
#MUSIC
num_angles_search = 3601
#Angles to look for orthogonality
angles = np.linspace(-160, 160, num_angles_search)
music_spectrum = np.zeros(num_angles_search)

iters = 0

noise_magnitude = 0

##################END INIT##################

#################Start of JACK Functions##########3
def print_error(*args):
    print(*args, file=sys.stderr)

def xrun(delay):
    print_error("An xrun occured, increase JACK's period size?")

def stop_callback(msg=''):
    if msg:
        print_error(msg)
    event.set()
    raise jack.CallbackExit


if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

event = threading.Event()

in1 = client.inports.register('input_1')
in2 = client.inports.register('input_2')
in3 = client.inports.register('input_3')

system_active = False

@client.set_port_connect_callback
def port_connect(a, b, connect):
    if a in client.inports or b in client.inports:
        print(['disconnected', 'connected'][connect], a, 'and', b)
        global iters, system_active, noise_magnitude
        if connect == 0:
            system_active = False
            noise_magnitude = 0
            iters = 0
        else:
            system_active = True

@client.set_process_callback
def process(frames):
    global system_active
    if system_active:
        try:
            qx.put_nowait(in1.get_array())
            qy.put_nowait(in2.get_array())
            qz.put_nowait(in3.get_array())
        except (queue.Full):
            print("Full Queue")
            stop_callback("A")

@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    event.set()

#PLOT INIT
global fig, ax

iters = 0

mic_rads = [0, 0.25, 0.25]
mic_thetas = [0, 270*2*pi/360, 210*2*pi/360]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#Propiedades del plot
ax.set_rmax(1)
ax.set_rticks([0.25, 0.5, 0.75, 1])  # Less radial ticks
ax.set_yticklabels([""]*4)
ax.set_title("Direcciones de arribo", va='bottom')

#Si se conocen las doas, mostramos doas reales
ax.plot(doas*2*pi/360, [0.95]*r, marker='o', color='g', ls='', mfc='none', markersize=10)
#Micros
ax.plot(mic_thetas, mic_rads, marker='x', color='b', ls='')
#Modificando posicionamiento del plot
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
#Doas obtenidas
line, = ax.plot([], [], marker='o', color='r', ls='')

def fft_spectrum(s, num_el, eps=1e-7):
    r = (abs(s)/num_el)[:len(s)//2+1]
    r[1:-1] = 2*r[1:-1]
    return r + eps

def init():
    line.set_data([], [])
    return line,

def func_animate(i):
    global system_active, iters, noise_magnitude

    if not system_active:
        line.set_data([], [])
        return line, 
    try:
        music_spectrum[:] = 0
        x = concatenate((qx.get(timeout=timeout), qx.get(timeout=timeout)))
        y = concatenate((qy.get(timeout=timeout), qy.get(timeout=timeout)))
        z = concatenate((qz.get(timeout=timeout), qz.get(timeout=timeout)))
        #Filtro pre-énfasis
        x = np.append(x[0], x[1:] - 0.97 * x[:-1])
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        z = np.append(z[0], z[1:] - 0.97 * z[:-1])
        xfft = fft(x)
        amp_spectrum = fft_spectrum(xfft, len(x))
        if(iters < 5):
            #Build noise floor so we can detect voice activity
            noise_magnitude = max(sum(abs(x)), noise_magnitude)
            line.set_data([], [])
        elif sum(abs(x)) > noise_magnitude*1.2:
            yfft = fft(y)
            zfft = fft(z)
            X = np.vstack([xfft, yfft, zfft])
                
            num_max_freqs = 10

            freq_peaks = find_peaks(amp_spectrum)[0]
            peaks_ordered = freq_peaks[np.argsort(amp_spectrum[freq_peaks])[::-1]]
            search_fs = peaks_ordered[:num_max_freqs]

            if len(search_fs) > 0:
                MUSIC(X, d, r, w, search_fs, num_angles_search, angles, music_spectrum)
                music_peaks = find_peaks(music_spectrum)[0]
                peak_pos = np.argsort(music_spectrum[music_peaks])[::-1]
                out = -angles[music_peaks[peak_pos][:r]]*2*pi/360
                if len(out) > 0:
                    rads = [0.95]*len(out)
                    line.set_data(out, rads)
            else:
                line.set_data([], [])
        else:
            line.set_data([], [])
        iters += num_windows_iter
    except (queue.Empty):
        print("Empty Queue")
    return line,
ani = FuncAnimation(fig,
                    func_animate,
                    init_func=init,
                    interval=47,
                    blit=True)
#END PLOT INIT

with client:
    print("[INFO]: Started JACK client.")
    #start = time.time()
    plt.show()
    while True:
        pass
