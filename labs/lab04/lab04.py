import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.gaussian as gau

plt.figure()
XPlot = np.linspace(-8, 12, 1000).reshape(1, 1000)
m = np.ones((1,1)) * 1.0
C = np.ones((1,1)) * 2.0
plt.plot(XPlot.ravel(), np.exp(gau.log_pdf_MVG(XPlot, m, C)))
plt.show()

pdfSol = np.load('labs/lab04/llGAU.npy')
pdfGau = gau.log_pdf_MVG(XPlot, m, C)
print(np.abs(pdfSol - pdfGau).max())


XND = np.load('labs/lab04/XND.npy')
mu = np.load('labs/lab04/muND.npy')
C = np.load('labs/lab04/CND.npy')
pdfSol = np.load('labs/lab04/llND.npy')
pdfGau = gau.log_pdf_MVG(XND, mu, C)
print(np.abs(pdfSol - pdfGau).max())

mu_ML, C_ML = gau.MVG_estimate(XND)
print(gau.log_likelihood(XND, mu_ML, C_ML, gau.log_pdf_MVG).sum())


X1D = np.load('labs/lab04/X1D.npy')
mu_ML, C_ML = gau.MVG_estimate(X1D)

plt.figure()
plt.hist(X1D.ravel(), bins=50, density=True)
XPlot = np.linspace(-8, 12, 1000).reshape(1, 1000)
plt.plot(XPlot.ravel(), np.exp(gau.log_pdf_MVG(XPlot, mu_ML, C_ML)))
plt.show()

print(gau.log_likelihood(X1D, mu_ML, C_ML, gau.log_pdf_MVG).sum())