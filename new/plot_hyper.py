import numpy as np
import os
import sys
import matplotlib.pyplot as plt


### minDCFs plot for lin log reg
#
#minDCFs_lin_log_reg_raw = np.array([0.762, 0.762, 0.762, 0.762, 0.762, 0.762, 0.761, 0.755, 0.748, 0.761])
#minDCFs_lin_log_reg_stand = np.array([0.762, 0.762, 0.762, 0.762, 0.761, 0.758, 0.752, 0.764, 0.896, 0.955])
#minDCFs_lin_log_reg_stand_PCA5 = np.array([0.759, 0.759, 0.759, 0.759, 0.758, 0.755, 0.753, 0.764, 0.895, 0.955])
#minDCFs_lin_log_reg_stand_PCA4 = np.array([0.748, 0.748, 0.748, 0.748, 0.748, 0.748, 0.749, 0.762, 0.894, 0.955])
#minDCFs_lin_log_reg_stand_PCA3 = np.array([0.758, 0.758, 0.758, 0.758, 0.758, 0.758, 0.759, 0.774, 0.896, 0.957])
#
#params_x = np.array([0.] + [10 ** lam_exp for lam_exp in range(-6, 3)])
#
#fig, ax = plt.subplots()
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_raw, label='Raw')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand, label='Standardized')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand_PCA5, label='PCA5')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand_PCA4, label='PCA4')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand_PCA4, label='PCA3')
#ax.set_xlabel('lambda')
#ax.set_ylabel('minMetric')
#ax.set_ylim(0.745, 0.765)
#ax.set_title('minMetric for linear logistic regression')
#ax.legend()
#ax.grid()
#plt.show()
#
#exit()

### minDCFs plot for quad log reg
#
#minDCFs_quad_log_reg_raw = np.array([0.262, 0.262, 0.262, 0.262, 0.262, 0.260, 0.257, 0.249, 0.248, 0.258])
#minDCFs_quad_log_reg_stand = np.array([0.262, 0.262, 0.260, 0.255, 0.249, 0.263, 0.310, 0.433, 0.607, 0.833])
#minDCFs_quad_log_reg_stand_PCA5 = np.array([0.262, 0.262, 0.260, 0.259, 0.248, 0.261, 0.313, 0.436, 0.609, 0.833])
#minDCFs_quad_log_reg_stand_PCA4 = np.array([0.308, 0.308, 0.308, 0.306, 0.317, 0.323, 0.363, 0.447, 0.618, 0.832])
#minDCFs_quad_log_reg_stand_PCA3 = np.array([0.331, 0.331, 0.331, 0.332, 0.332, 0.334, 0.376, 0.494, 0.635, 0.841])
#
#params_x = np.array([0.] + [10 ** lam_exp for lam_exp in range(-6, 3)])
#
#fig, ax = plt.subplots()
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_raw, label='Raw')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand, label='Standardized')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand_PCA5, label='PCA5')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand_PCA4, label='PCA4')
#ax.plot(np.log10(params_x), minDCFs_quad_log_reg_stand_PCA4, label='PCA3')
#ax.set_xlabel('lambda')
#ax.set_ylabel('minMetric')
#ax.set_ylim(0.24, 0.28)
#ax.set_title('minMetric for quadratic logistic regression')
#ax.legend()
#ax.grid()
#plt.show()
#
#exit()

### minDCFs plot for lin svm

minDCFs_lin_svm_raw = np.array([[0.740, 0.743, 0.849], [0.740, 0.745, 0.855], [0.740, 0.743, 0.853], [0.742, 0.935, 0.962], [0.973, 0.805, 0.973]])
minDCFs_lin_svm_stand = np.array([[0.748, 0.763, 0.773], [0.748, 0.763, 0.772], [0.748, 0.762, 0.774], [0.748, 0.974, 0.951], [0.959, 0.966, 0.982]])
minDCFs_lin_svm_stand_PCA5 = np.array([[0.748, 0.762, 0.771], [0.748, 0.762, 0.773], [0.748, 0.762, 0.779], [0.748, 0.898, 0.955], [0.969, 0.999, 0.976]])
minDCFs_lin_svm_stand_PCA4 = np.array([[0.744, 0.746, 0.762], [0.744, 0.746, 0.762], [0.744, 0.746, 0.763], [0.746, 0.873, 0.983], [0.979, 0.990, 0.966]])
minDCFs_lin_svm_stand_PCA3 = np.array([[0.752, 0.746, 0.767], [0.752, 0.746, 0.768], [0.752, 0.747, 0.764], [0.753, 0.821, 0.948], [0.968, 0.949, 0.930]])

params_x = np.array([-6, -4, -2, 0, 2])
x_name = 'K'
params_y = np.array([-4, -2, 0])
y_name = 'C'

X, Y = np.meshgrid(params_x, params_y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_raw.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on raw data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on standardized data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand_PCA5.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on PCA5-reduced data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand_PCA3.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on PCA4-reduced data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand_PCA3.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on PCA3-reduced data')
plt.show()

exit()

### minDCFs plot for poly2 svm

minDCFs_lin_svm_raw = np.array([])
minDCFs_lin_svm_stand = np.array([])
minDCFs_lin_svm_stand_PCA5 = np.array([])
minDCFs_lin_svm_stand_PCA4 = np.array([])
minDCFs_lin_svm_stand_PCA3 = np.array([])

params_x = np.array([-6, -4, -2, 0, 2])
x_name = 'K'
params_y = np.array([-4, -2, 0])
y_name = 'C'

X, Y = np.meshgrid(params_x, params_y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_raw.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on raw data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on standardized data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand_PCA5.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on PCA5-reduced data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand_PCA3.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on PCA4-reduced data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, minDCFs_lin_svm_stand_PCA3.T)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel('minMetric')
ax.set_title(f'minMetric for linear svm on PCA3-reduced data')
plt.show()

exit()
