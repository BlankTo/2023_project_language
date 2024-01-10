import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.gaussian as gau
import ML_lib.utils as utils
import ML_lib.preprocessing as pre

D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
DTR, LTR, DTE, LTE = utils.shuffle_and_divide(D, L, 2.0/3.0, 0)
print(DTR.shape)
print(LTR.shape)
print(DTE.shape)
print(LTE.shape)


######  Base Gaussian
#print('----- BASE -----')
#
#mu, C = gau.multiclass_MVG_estimate(DTR, LTR)
#
######
#
#S = gau.multiclass_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#S_joint = gau.joint(S)
#S_joint_prof = np.load('labs/lab05/SJoint_MVG.npy')
#print(abs(S_joint - S_joint_prof).max())
#
#S_marginal = gau.marginal(S_joint)
#
#posteriors = gau.posterior(S_joint, S_marginal)
#posteriors_prof = np.load('labs/lab05/Posterior_MVG.npy')
#print(abs(posteriors - posteriors_prof).max())
#
#predictions = np.argmax(posteriors, axis= 0)
#accuracy = (predictions==LTE).sum()/LTE.size
#print('accuracy: ' + str(accuracy*100) + '%')
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#
#
######
#
#
#log_S = gau.multiclass_log_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#log_S_joint = gau.log_joint(log_S)
#log_S_joint_prof = np.load('labs/lab05/logSJoint_MVG.npy')
#print(abs(log_S_joint - log_S_joint_prof).max())
#
#log_S_marginal = gau.log_marginal(log_S_joint)
#log_S_marginal_prof = np.load('labs/lab05/logMarginal_MVG.npy')
#print(abs(log_S_marginal - log_S_marginal_prof).max())
#
#log_posteriors = gau.log_posterior(log_S_joint, log_S_marginal)
#log_posteriors_prof = np.load('labs/lab05/logPosterior_MVG.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#
#log_predictions = np.argmax(log_posteriors, axis= 0)
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error: ' + str(log_error*100) + '%')
#
######
#
#
######  Naive Bayes
#print('----- Naive Bayes -----')
#
#mu, C = gau.multiclass_MVG_estimate(DTR, LTR, 'naive')
#
######
#
#S = gau.multiclass_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#S_joint = gau.joint(S)
#S_joint_prof = np.load('labs/lab05/SJoint_NaiveBayes.npy')
#print(abs(S_joint - S_joint_prof).max())
#
#S_marginal = gau.marginal(S_joint)
#
#posteriors = gau.posterior(S_joint, S_marginal)
#posteriors_prof = np.load('labs/lab05/Posterior_NaiveBayes.npy')
#print(abs(posteriors - posteriors_prof).max())
#
#predictions = np.argmax(posteriors, axis= 0)
#accuracy = (predictions==LTE).sum()/LTE.size
#print('accuracy: ' + str(accuracy*100) + '%')
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#
#
######
#
#
#log_S = gau.multiclass_log_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#log_S_joint = gau.log_joint(log_S)
#log_S_joint_prof = np.load('labs/lab05/logSJoint_NaiveBayes.npy')
#print(abs(log_S_joint - log_S_joint_prof).max())
#
#log_S_marginal = gau.log_marginal(log_S_joint)
#log_S_marginal_prof = np.load('labs/lab05/logMarginal_NaiveBayes.npy')
#print(abs(log_S_marginal - log_S_marginal_prof).max())
#
#log_posteriors = gau.log_posterior(log_S_joint, log_S_marginal)
#log_posteriors_prof = np.load('labs/lab05/logPosterior_NaiveBayes.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#
#log_predictions = np.argmax(log_posteriors, axis= 0)
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error: ' + str(log_error*100) + '%')
#
######
#
#
######  Tied Covariance
#print('----- Tied Covariance -----')
#
#mu, C = gau.multiclass_MVG_estimate(DTR, LTR, 'tied')
#
######
#
#S = gau.multiclass_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#S_joint = gau.joint(S)
#S_joint_prof = np.load('labs/lab05/SJoint_TiedMVG.npy')
#print(abs(S_joint - S_joint_prof).max())
#
#S_marginal = gau.marginal(S_joint)
#
#posteriors = gau.posterior(S_joint, S_marginal)
#posteriors_prof = np.load('labs/lab05/Posterior_TiedMVG.npy')
#print(abs(posteriors - posteriors_prof).max())
#
#predictions = np.argmax(posteriors, axis= 0)
#accuracy = (predictions==LTE).sum()/LTE.size
#print('accuracy: ' + str(accuracy*100) + '%')
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#
#
######
#
#
#log_S = gau.multiclass_log_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#log_S_joint = gau.log_joint(log_S)
#log_S_joint_prof = np.load('labs/lab05/logSJoint_TiedMVG.npy')
#print(abs(log_S_joint - log_S_joint_prof).max())
#
#log_S_marginal = gau.log_marginal(log_S_joint)
#log_S_marginal_prof = np.load('labs/lab05/logMarginal_TiedMVG.npy')
#print(abs(log_S_marginal - log_S_marginal_prof).max())
#
#log_posteriors = gau.log_posterior(log_S_joint, log_S_marginal)
#log_posteriors_prof = np.load('labs/lab05/logPosterior_TiedMVG.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#
#log_predictions = np.argmax(log_posteriors, axis= 0)
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error: ' + str(log_error*100) + '%')
#
######
#
#
######  Tied Naive Bayes
#print('----- Tied Naive Bayes -----')
#
#mu, C = gau.multiclass_MVG_estimate(DTR, LTR, 'tied-naive')
#
######
#
#S = gau.multiclass_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#S_joint = gau.joint(S)
#S_joint_prof = np.load('labs/lab05/SJoint_TiedNaiveBayes.npy')
#print(abs(S_joint - S_joint_prof).max())
#
#S_marginal = gau.marginal(S_joint)
#
#posteriors = gau.posterior(S_joint, S_marginal)
#posteriors_prof = np.load('labs/lab05/Posterior_TiedNaiveBayes.npy')
#print(abs(posteriors - posteriors_prof).max())
#
#predictions = np.argmax(posteriors, axis= 0)
#accuracy = (predictions==LTE).sum()/LTE.size
#print('accuracy: ' + str(accuracy*100) + '%')
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#
#
######
#
#
#log_S = gau.multiclass_log_likelihood(DTE, mu, C, gau.log_pdf_MVG)
#
#log_S_joint = gau.log_joint(log_S)
#log_S_joint_prof = np.load('labs/lab05/logSJoint_TiedNaiveBayes.npy')
#print(abs(log_S_joint - log_S_joint_prof).max())
#
#log_S_marginal = gau.log_marginal(log_S_joint)
#log_S_marginal_prof = np.load('labs/lab05/logMarginal_TiedNaiveBayes.npy')
#print(abs(log_S_marginal - log_S_marginal_prof).max())
#
#log_posteriors = gau.log_posterior(log_S_joint, log_S_marginal)
#log_posteriors_prof = np.load('labs/lab05/logPosterior_TiedNaiveBayes.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#
#log_predictions = np.argmax(log_posteriors, axis= 0)
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error: ' + str(log_error*100) + '%')
#
######


#####
#
#log_posteriors, log_predictions = gau.Gaussian_Classifier_new(DTR, LTR, DTE, {'retPosteriors': True})
#log_posteriors_prof = np.load('labs/lab05/logPosterior_MVG.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy base: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error base: ' + str(log_error*100) + '%')
#
#log_posteriors, log_predictions = gau.Gaussian_Classifier_new(DTR, LTR, DTE, {'version': 'naive', 'retPosteriors': True})
#log_posteriors_prof = np.load('labs/lab05/logPosterior_NaiveBayes.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy naive: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error naive: ' + str(log_error*100) + '%')
#
#log_posteriors, log_predictions = gau.Gaussian_Classifier_new(DTR, LTR, DTE, {'version': 'tied', 'retPosteriors': True})
#log_posteriors_prof = np.load('labs/lab05/logPosterior_TiedMVG.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy tied: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error tied: ' + str(log_error*100) + '%')
#
#log_posteriors, log_predictions = gau.Gaussian_Classifier_new(DTR, LTR, DTE, {'version': 'tied-naive', 'retPosteriors': True})
#log_posteriors_prof = np.load('labs/lab05/logPosterior_TiedNaiveBayes.npy')
#print(abs(log_posteriors - log_posteriors_prof).max())
#log_accuracy = (log_predictions==LTE).sum()/LTE.size
#print('log_accuracy tied-naive: ' + str(log_accuracy*100) + '%')
#log_error = 1 - log_accuracy
#print('log_error tied-naive: ' + str(log_error*100) + '%')
#
#####


#####  Cross Validation

#for prepr in [False, pre.normalizer, pre.standardizer]:
#    for dim_r in [False, pre.PCA, pre.LDA]:
#        for m in [1, 2, 3]:
#            print('base with ' + str(prepr) + '   ' + str(dim_r) + '   m: ' + str(m))
#            LTE_cross, predictions = utils.cross_validation(DTR, LTR, gau.Gaussian_Classifier_new, n_fold= -1, prep= prepr, dim_red= dim_r, m_dim_red= m, args_in= {'version': 'base'})
#            accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#            error = 1 - accuracy
#            print('error: ' + str(error*100) + '%')
#            print('---------------------------------')

#for prepr in [False, pre.normalizer, pre.standardizer]:
#    for dim_r in [False, pre.PCA, pre.LDA]:
#        for m in [1, 2, 3]:
#            print('naive with ' + str(prepr) + '   ' + str(dim_r) + '   m: ' + str(m))
#            LTE_cross, predictions = utils.cross_validation(DTR, LTR, gau.Gaussian_Classifier_new, n_fold= -1, prep= prepr, dim_red= dim_r, m_dim_red= m, args_in= {'version': 'naive'})
#            accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#            error = 1 - accuracy
#            print('error: ' + str(error*100) + '%')
#            print('---------------------------------')

#for prepr in [False, pre.normalizer, pre.standardizer]:
#    for dim_r in [False, pre.PCA, pre.LDA]:
#        for m in [1, 2, 3]:
#            print('tied with ' + str(prepr) + '   ' + str(dim_r) + '   m: ' + str(m))
#            LTE_cross, predictions = utils.cross_validation(DTR, LTR, gau.Gaussian_Classifier_new, n_fold= -1, prep= prepr, dim_red= dim_r, m_dim_red= m, args_in= {'version': 'tied'})
#            accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#            error = 1 - accuracy
#            print('error: ' + str(error*100) + '%')
#            print('---------------------------------')

#for prepr in [False, pre.normalizer, pre.standardizer]:
#    for dim_r in [False, pre.PCA, pre.LDA]:
#        for m in [1, 2, 3]:
#            print('tied-naive with ' + str(prepr) + '   ' + str(dim_r) + '   m: ' + str(m))
#            LTE_cross, predictions = utils.cross_validation(DTR, LTR, gau.Gaussian_Classifier_new, n_fold= -1, prep= prepr, dim_red= dim_r, m_dim_red= m, args_in= {'version': 'tied-naive'})
#            accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#            error = 1 - accuracy
#            print('error: ' + str(error*100) + '%')
#            print('---------------------------------')

#####

#LTE_cross, predictions = utils.cross_validation(D, L, gau.Gaussian_Classifier_new, n_folds= -1, args_in= {'version': 'base'})
#accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#print('---------------------------------')
#
#LTE_cross, predictions = utils.cross_validation(D, L, gau.Gaussian_Classifier_new, n_folds= -1, args_in= {'version': 'naive'})
#accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#print('---------------------------------')
#
#LTE_cross, predictions = utils.cross_validation(D, L, gau.Gaussian_Classifier_new, n_folds= -1, args_in= {'version': 'tied'})
#accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#print('---------------------------------')
#
#LTE_cross, predictions = utils.cross_validation(D, L, gau.Gaussian_Classifier_new, n_folds= -1, args_in= {'version': 'tied-naive'})
#accuracy = (predictions==LTE_cross).sum()/LTE_cross.size
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#print('---------------------------------')

#####

utils.cross_validation(D, L, 2, gau.GaussianClassifier, [ # check new and new_2
    [],
    ['naive'],
    ['tied'],
    ['naive-tied']
    ], progress= True, print_act= False, save= False, prepro= [

    [(pre.NoTransform, [])],
    [(pre.Standardizer, [])],
    [(pre.Gaussianizer, [])],
    ])
#error = 1 - accuracy
#print('error: ' + str(error*100) + '%')
#print('---------------------------------')