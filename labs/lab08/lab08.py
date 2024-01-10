import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.gaussian as gau
import ML_lib.utils as utils
import ML_lib.preprocessing as pre
from ML_lib.plot_lib import ROCcurve, BayesErrorPlot

#D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
#DTR, LTR, DTE, LTE = utils.shuffle_and_divide(D, L, 2.0/3.0, 0)
#print(DTR.shape)
#print(LTR.shape)
#print(DTE.shape)
#print(LTE.shape)
#
#log_predictions = gau.Gaussian_Classifier_new(DTR, LTR, DTE, {})
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy base: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error base: ' + str(log_error*100) + '%')
#
#print(utils.getConfusionMatrix(log_predictions, LTE))
#
#log_predictions = gau.Gaussian_Classifier_new(DTR, LTR, DTE, {'version': 'tied'})
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy base: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error base: ' + str(log_error*100) + '%')
#
#print(utils.getConfusionMatrix(log_predictions, LTE))






#log_S = np.load('labs/lab08/data/commedia_ll.npy')
#labels = np.load('labs/lab08/data/commedia_labels.npy')
#
#priors = None
#
#log_S_joint_uniform = gau.log_joint(log_S, priors= priors)
#log_S_marginal_uniform = gau.log_marginal(log_S_joint_uniform)
#log_posteriors_uniform = gau.log_posterior(log_S_joint_uniform, log_S_marginal_uniform)
#log_predictions_uniform = np.argmax(log_posteriors_uniform, axis= 0)
#
#print(utils.getConfusionMatrix(log_predictions_uniform, labels))
#
#costs = np.array([[0, 1, 1],
#                  [1, 0, 1],
#                  [1, 1, 0]]) # i= prediction, j= actual,     K x K
#
#predictions_bayes = utils.getBayesDecision(log_posteriors_uniform, costs)
#
#print(utils.getConfusionMatrix(predictions_bayes, labels))






#labels_infpar = np.load('labs/lab08/data/commedia_labels_infpar.npy')
#llr = np.load('labs/lab08/data/commedia_llr_infpar.npy')
#
#p1 = 0.5
#Cfp = 1
#Cfn = 1
#print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')
#conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
#print(conf)
#FPR, FNR, _, _ = utils.getFR(conf)
#print('%.3f' % utils.empiricalBayesRisk(p1, Cfp, Cfn, FPR, FNR))
#print('%.3f' % utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR))
#minDCF = utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)
#print('%.3f' % minDCF)
#
#
#p1 = 0.8
#Cfp = 1
#Cfn = 1
#print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')
#conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
#print(conf)
#FPR, FNR, _, _ = utils.getFR(conf)
#print('%.3f' % utils.empiricalBayesRisk(p1, Cfp, Cfn, FPR, FNR))
#print('%.3f' % utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR))
#minDCF = utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)
#print('%.3f' % minDCF)
#
#
#p1 = 0.5
#Cfp = 1
#Cfn = 10
#print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')
#conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
#print(conf)
#FPR, FNR, _, _ = utils.getFR(conf)
#print('%.3f' % utils.empiricalBayesRisk(p1, Cfp, Cfn, FPR, FNR))
#print('%.3f' % utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR))
#minDCF = utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)
#print('%.3f' % minDCF)
#
#
#p1 = 0.8
#Cfp = 10
#Cfn = 1
#print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')
#conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
#print(conf)
#FPR, FNR, _, _ = utils.getFR(conf)
#print('%.3f' % utils.empiricalBayesRisk(p1, Cfp, Cfn, FPR, FNR))
#print('%.3f' % utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR))
#minDCF = utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)
#print('%.3f' % minDCF)
#
#
#ROCcurve(llr, labels_infpar, p1, Cfp, Cfn)
#
#
#effPriorLogOdds = np.linspace(-3, 3, 21)
#BayesErrorPlot(llr, labels_infpar, effPriorLogOdds)







labels_infpar = np.load('labs/lab08/data/commedia_labels_infpar.npy')
llr = np.load('labs/lab08/data/commedia_llr_infpar.npy')
labels_infpar_eps1 = np.load('labs/lab08/data/commedia_labels_infpar_eps1.npy')
llr_eps1 = np.load('labs/lab08/data/commedia_llr_infpar_eps1.npy')

p1 = 0.5
Cfp = 1
Cfn = 1
print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')

conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps0.001\t normDCF: %.3f  -  minNormDCF: %.3f' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)))

conf = utils.getConfusionMatrix((llr_eps1 > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar_eps1)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps1.\t\t normDCF: %.3f  -  minNormDCF: %.3f\n' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr_eps1, labels_infpar_eps1, p1, Cfp, Cfn)))


p1 = 0.8
Cfp = 1
Cfn = 1
print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')

conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps0.001\t normDCF: %.3f  -  minNormDCF: %.3f' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)))

conf = utils.getConfusionMatrix((llr_eps1 > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar_eps1)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps1.\t\t normDCF: %.3f  -  minNormDCF: %.3f\n' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr_eps1, labels_infpar_eps1, p1, Cfp, Cfn)))


p1 = 0.5
Cfp = 1
Cfn = 10
print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')

conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps0.001\t normDCF: %.3f  -  minNormDCF: %.3f' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)))

conf = utils.getConfusionMatrix((llr_eps1 > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar_eps1)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps1.\t\t normDCF: %.3f  -  minNormDCF: %.3f\n' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr_eps1, labels_infpar_eps1, p1, Cfp, Cfn)))


p1 = 0.8
Cfp = 10
Cfn = 1
print(f'p1: {p1}, Cfp: {Cfp}, Cfn: {Cfn}')

conf = utils.getConfusionMatrix((llr > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps0.001\t normDCF: %.3f  -  minNormDCF: %.3f' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr, labels_infpar, p1, Cfp, Cfn)))

conf = utils.getConfusionMatrix((llr_eps1 > utils.getBayesThreshold(p1, Cfp, Cfn)).astype(int), labels_infpar_eps1)
FPR, FNR, _, _ = utils.getFR(conf)
print('eps1.\t\t normDCF: %.3f  -  minNormDCF: %.3f\n' % (utils.normalizedDCF(p1, Cfp, Cfn, FPR, FNR), utils.getMinNormDCF(llr_eps1, labels_infpar_eps1, p1, Cfp, Cfn)))


effPriorLogOdds = np.linspace(-3, 3, 21)
plt.figure()
BayesErrorPlot(llr, labels_infpar, effPriorLogOdds, legend_text= 'eps0.001', stack= True)
BayesErrorPlot(llr_eps1, labels_infpar_eps1, effPriorLogOdds, legend_text= 'eps1.', stack= True)
plt.ylim([0, 1.1])
plt.xlim([-3, 3])
plt.grid()
plt.legend()
plt.show()