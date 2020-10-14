#-------------------------------------------------------------------------------
# Name:        High Dynamic Range image reconstruction
# Purpose:     cs484
#
# Author:      sadueck@edu.uwaterloo
#
# Created:     18/12/2018
#-------------------------------------------------------------------------------


import numpy as np
import scipy.misc
import scipy.sparse
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

# gsolve: Reproduced from sample Matlab code by Debevec and Malik
# Accepts:  Z sample pixels (along all exposure values)
#           B exposure values
#           l smoothing (regularization parameter)
# Returns:  g response curve (log exposure at pixel z)
#           lE log film irradiance of pizels in Z
def gsolve(Z, B, l):
    n = 256
    w = np.ones(n)/n
    m = Z.shape[0]
    p = Z.shape[1]

    A = scipy.sparse.lil_matrix((m*p+n+1, n+m))  #sparse([], [], [], m*p+n+1, n+m, m*p*2+n*3)
    b = np.zeros(A.shape[0])

    k = 0
    for i in range(0,m):
        for j in range(0,p):
            wij = w[Z[i,j]]
            A[k, Z[i,j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[j]
            k += 1

    A[k, 128] = 1
    k += 1

    for i in range (0, n-2):
        A[k,i  ] =    l*w[i+1]
        A[k,i+1] = -2*l*w[i+1]
        A[k,i+2] =    l*w[i+1]

    x, istop = scipy.sparse.linalg.lsqr(A,b)[:2] #np.linalg.lstsq(A,b)

    print(istop)
    g = x[0:n]
    lE = x[n:len(x)]
    return g, lE

# Read in Sample Data
m1 = scipy.misc.imread("agia_sofia1.jpg")
m2 = scipy.misc.imread("agia_sofia2.jpg")
m3 = scipy.misc.imread("agia_sofia3.jpg")
m4 = scipy.misc.imread("agia_sofia4.jpg")
m5 = scipy.misc.imread("agia_sofia5.jpg")
m6 = scipy.misc.imread("agia_sofia6.jpg")
m7 = scipy.misc.imread("agia_sofia7.jpg")
m8 = scipy.misc.imread("agia_sofia8.jpg")
m9 = scipy.misc.imread("agia_sofia9.jpg")
m10 = scipy.misc.imread("agia_sofia10.jpg")
m11 = scipy.misc.imread("agia_sofia11.jpg")
B = np.array([1/8, 1/15, 1/30, 1/60, 1/125, 1/250, 1/400, 1/640, 1/1000, 1/1600, 1/4000])
l = 10
P = 11

# Z is [pixels, images] (flattened images stacked)
ZR = np.stack((m1[:,:,0].flatten(), m2[:,:,0].flatten(), m3[:,:,0].flatten(), m4[:,:,0].flatten(), m5[:,:,0].flatten(), m6[:,:,0].flatten(), m7[:,:,0].flatten(), m8[:,:,0].flatten(), m9[:,:,0].flatten(), m10[:,:,0].flatten(), m11[:,:,0].flatten()), axis=1)
ZG = np.stack((m1[:,:,1].flatten(), m2[:,:,1].flatten(), m3[:,:,1].flatten(), m4[:,:,1].flatten(), m5[:,:,1].flatten(), m6[:,:,1].flatten(), m7[:,:,1].flatten(), m8[:,:,1].flatten(), m9[:,:,1].flatten(), m10[:,:,1].flatten(), m11[:,:,1].flatten()), axis=1)
ZB = np.stack((m1[:,:,2].flatten(), m2[:,:,2].flatten(), m3[:,:,2].flatten(), m4[:,:,2].flatten(), m5[:,:,2].flatten(), m6[:,:,2].flatten(), m7[:,:,2].flatten(), m8[:,:,2].flatten(), m9[:,:,2].flatten(), m10[:,:,2].flatten(), m11[:,:,2].flatten()), axis=1)

# Find camera response curves
ZR_sample = ZR[np.random.randint(ZR.shape[0], size=1000)]
gR, lER = gsolve(ZR_sample, np.log(B), l)

ZG_sample = ZG[np.random.randint(ZG.shape[0], size=1000)]
gG, lEG = gsolve(ZG_sample, np.log(B), l)

ZB_sample = ZB[np.random.randint(ZB.shape[0], size=1000)]
gB, lEB = gsolve(ZB_sample, np.log(B), l)

# Sum recovered log irradiance + log exposure time
XR = np.zeros(ZR_sample.shape)
for i in range(0, XR.shape[0]):
    for j in range(0, XR.shape[1]):
        XR[i,j] = lER[i] + math.log(B[j])

XG = np.zeros(ZG_sample.shape)
for i in range(0, XG.shape[0]):
    for j in range(0, XG.shape[1]):
        XG[i,j] = lEG[i] + math.log(B[j])

XB = np.zeros(ZB_sample.shape)
for i in range(0, XB.shape[0]):
    for j in range(0, XB.shape[1]):
        XB[i,j] = lEB[i] + math.log(B[j])

# Plot response curve, with samples used in gsolve
plt.plot(gR, range(0, 256), 'r')
plt.scatter(XR, ZR_sample, s=0.1)
plt.title('Response Curve for Red Channel with Fitted Data')
plt.ylabel('Pixel Value (Z)')
plt.xlabel('log Exposure')
plt.show()

plt.plot(gG, range(0, 256), 'g')
plt.scatter(XG, ZG_sample, s=0.1)
plt.title('Response Curve for Green Channel with Fitted Data')
plt.ylabel('Pixel Value (Z)')
plt.xlabel('log Exposure')
plt.show()

plt.plot(gB, range(0, 256), 'b')
plt.scatter(XB, ZB_sample, s=0.1)
plt.title('Response Curve for Blue Channel with Fitted Data')
plt.ylabel('Pixel Value (Z)')
plt.xlabel('log Exposure')
plt.show()


# Recover HRD radiance map
# ? improve performance by setting every pixel of he same value x \in [0-255]
# simultaneously. Reduces to 255 iterations from I (>400000)
lnER = np.zeros(ZR.shape[0])
for i in range(0, lnER.shape[0]):
    r = 0
    for j in range(0, P):
        r += gR[ZR[i,j]] - math.log(B[j])
    lnER[i] = r/P
ER = np.exp(lnER)

lnEG = np.zeros(ZG.shape[0])
for i in range(0, lnEG.shape[0]):
    r = 0
    for j in range(0, P):
        r += gG[ZG[i,j]] - math.log(B[j])
    lnEG[i] = r/P
EG = np.exp(lnEG)

lnEB = np.zeros(ZB.shape[0])
for i in range(0, lnEB.shape[0]):
    r = 0
    for j in range(0, P):
        r += gB[ZB[i,j]] - math.log(B[j])
    lnEB[i] = r/P
EB = np.exp(lnEB)


# Display full radiance maps
plt.imshow(lnER.reshape(m1.shape[0:2])).set_cmap('nipy_spectral')
plt.colorbar()
plt.title('Recovered Radiance Map (Red Channel) \n (logarithmic scale)')
plt.show()

plt.imshow(lnEG.reshape(m1.shape[0:2])).set_cmap('nipy_spectral')
plt.colorbar()
plt.title('Recovered Radiance Map (Green Channel) \n (logarithmic scale)')
plt.show()

plt.imshow(lnEB.reshape(m1.shape[0:2])).set_cmap('nipy_spectral')
plt.colorbar()
plt.title('Recovered Radiance Map (Blue Channel) \n (logarithmic scale)')
plt.show()


# Tone mapping
#normalize brightness
ER_min = np.amin(ER)
ER_max = np.amax(ER)
ER_norm = (ER - ER_min) / (ER_max - ER_min)

EG_min = np.amin(EG)
EG_max = np.amax(EG)
EG_norm = (EG - EG_min)/(EG_max - EG_min)

EB_min = np.amin(EB)
EB_max = np.amax(EB)
EB_norm = (EB - EB_min)/(EB_max - EB_min)

En = np.stack((ER_norm.reshape(m1.shape[0:2]), EG_norm.reshape(m1.shape[0:2]), EB_norm.reshape(m1.shape[0:2])), axis=2)
plt.imshow(En)
plt.title('Normalized Randiance Map')
plt.show()

# gamma curve
gamma = 0.3
ER_gamma = np.power(ER_norm, gamma)
EG_gamma = np.power(EG_norm, gamma)
EB_gamma = np.power(EB_norm, gamma)

# restack channels
E = np.stack((ER_gamma.reshape(m1.shape[0:2]), EG_gamma.reshape(m1.shape[0:2]), EB_gamma.reshape(m1.shape[0:2])), axis=2)
plt.imshow(E)
plt.title('Gamma Corrected Normalized Randiance Map')
plt.show()
