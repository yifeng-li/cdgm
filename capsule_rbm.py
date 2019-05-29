# Allow X-Z interaction
#from __future__ import division
import numpy
#import scipy.special
import math
import os
import time
import classification as cl
import sys
import copy
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt

class capsule_rbm:
    def __init__(self, features=None, M=None, K=None, J=16, visible_type="Bernoulli", visible_type_fixed_param=100, hidden_type="Bernoulli", hidden_type_fixed_param=0, tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=False, a=None, fix_a_log_ind=None, xz_interaction=True, tol_poisson_max=numpy.log(255), rec_error_max=0.5, rng=numpy.random.RandomState(100)):
        self.features=features
        self.M=M # number of visible variables
        self.K=K # number of hidden capsules
        self.J=J # number of hidden units in a chidden apsule
        if numpy.isscalar(J): # self.J is a list of length K
            self.J=[J]*self.K
        self.rng=rng
        self.visible_type=visible_type
        self.tie_W_for_pretraining_DBM_bottom=tie_W_for_pretraining_DBM_bottom
        self.tie_W_for_pretraining_DBM_top=tie_W_for_pretraining_DBM_top
        self.visible_type_fixed_param=visible_type_fixed_param
        self.mfe_for_loglikelihood_train=None
        self.mfe_for_loglikelihood_test=None
        self.xz_interaction=xz_interaction
        self.tol_poisson_max=tol_poisson_max
        self.rec_error_max=rec_error_max
        
        # initiate bias for visible variables and weights
        self.W=[None]*(self.K+1) # the weight matrices for all capsules
        if self.visible_type=="Bernoulli":
            self.a=numpy.zeros(shape=(self.M,1))
            for k in range(self.K):
                self.W[k]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.J[k]))
                #self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
            # for the interaction between X and Z
            self.W[self.K]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K))
        elif self.visible_type=="Gaussian":
            self.a=[None]*2
            #self.a[0]=numpy.abs(self.rng.normal(loc=0, scale=1, size=(self.M,1))) # M X 1  a1=mu*lambda  # coefficient for x
            self.a[0]=self.rng.normal(loc=0, scale=1, size=(self.M,1))
            self.a[1]=-0.5*numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=-lambda/2<0 coefficient for x^2
            for k in range(self.K):
                #self.W[k]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.J[k])) # M by K, initialize weight matrix
                self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
            self.W[K]=numpy.zeros(shape=(self.M,self.K),dtype=float)
        elif self.visible_type=="Gaussian2":
            self.a=[None]*2
            #self.a[0]=numpy.abs(self.rng.normal(loc=0, scale=1, size=(self.M,1))) # M X 1  a1=mu*lambda  # coefficient for x
            self.a[0]=self.rng.normal(loc=0, scale=1, size=(self.M,1))
            self.a[1]=-5*numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=-lambda/2<0 coefficient for x^2
            for k in range(self.K):
                self.W[k]=[None,None]
                self.W[k][0]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
                self.W[k][1]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
        elif self.visible_type=="Gaussian_FixPrecision1": 
            #self.a=numpy.ones(shape=(self.M,1)) # a=mu, statistics is lambda*x
            self.a=self.rng.random_sample(size=(self.M,1)) # a=mu, statistics is lambda*x
            for k in range(self.K):
                #self.W[k]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.J[k]))
                self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
            # self.visible_type_fixed_param is precision lambda
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Gaussian_FixPrecision2":
            #self.a=numpy.ones(shape=(self.M,1)) # a=mu*lambda, statistics is x
            self.a=self.rng.random_sample(size=(self.M,1))
            for k in range(self.K):
                self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
            self.W[K]=numpy.zeros(shape=(self.M,self.K),dtype=float)
            # self.visible_type_fixed_param is precision lambda
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Gaussian_Hinton":
            self.a=[None]*2
            #self.a[0]=self.rng.normal(loc=0, scale=0.001, size=(self.M,1)) # M X 1  a1=mu
            self.a[0]=self.rng.random_sample(size=(self.M,1))
            self.a[1]=1*numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=lambda
            for k in range(self.K):
                #self.W[k]=self.rng.normal(loc=0, scale=0.1, size=(self.M,self.J[k])) # M by K, initialize weight matrix
                self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
        elif self.visible_type=="Poisson":
            #self.a=numpy.ones(shape=(self.M,1)) # a=log lambda
            self.a=self.rng.normal(loc=0, scale=0.1, size=(self.M,1))
            for k in range(self.K):
                self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
        elif self.visible_type=="NegativeBinomial":
            self.a=numpy.log(0.5)*numpy.ones(shape=(self.M,1)) # a=log(1-p), a <=0
            for k in range(self.K):
                self.W[k]=-numpy.abs(self.rng.normal(loc=0, scale=0.00001, size=(self.M,self.J[k])))
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Multinoulli":
            self.I=self.visible_type_fixed_param # a list of vector, the length of each multinoulli variable
            self.a=[None]*self.M
            for m in range(self.M):
                self.a[m]=math.log(1/self.I[m])*numpy.ones(shape=(self.I[m],1))
            for k in range(self.K):
                Wk=[None]*self.M
                for m in range(self.M):
                    Wk[m]=self.rng.normal(loc=0, scale=0.001, size=(self.I[m],self.J[k]))
                self.W[k]=Wk
        elif self.visible_type=="Multinomial":
            self.a=numpy.zeros(shape=(self.M,1))
            for k in range(self.K):
                self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
        else:
            print("Error! Please select a correct data type for visible variables from {Bernoulli,Gaussian,Multinoulli,Poisson}.")
            sys.exit(0)

        self.if_fix_vis_bias=if_fix_vis_bias
        self.fix_a_log_ind=fix_a_log_ind
        if self.if_fix_vis_bias:
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
        else:
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([False]*self.M)
            if a is not None:
                self.a=a
                print("I will fix a using the new a in this CRBM.")
            else:
                print("I will fix the existing a in this CRBM.")

        # initialize bias for hidden variables, h
        self.hidden_type=hidden_type
        self.b=[None]*self.K
        self.hidden_type_fixed_param=hidden_type_fixed_param
        if self.hidden_type=="Bernoulli" or self.hidden_type=="Binomial" or self.hidden_type=="Multinomial":
            for k in range(self.K):
                self.b[k]=numpy.zeros(shape=(self.J[k],1)) # length K, b=log(p/(1-p))
        elif self.hidden_type=="Gaussian":
            for k in range(self.K):
                bk=[None]*2
                #bk[0]=numpy.abs(self.rng.normal(loc=0, scale=1, size=(self.J[k],1)))
                bk[0]=self.rng.normal(loc=0, scale=1, size=(self.J[k],1))
                bk[1]=-0.5*numpy.ones(shape=(self.J[k],1),dtype=float)
                self.b[k]=bk
        elif self.hidden_type=="Gaussian_FixPrecision1":
            for k in range(self.K):
                #self.b[k]=numpy.one(shape=(self.J[k],1))
                self.b[k]=self.rng.random_sample(size=(self.J[k],1))
        elif self.hidden_type=="Gaussian_FixPrecision2":
            for k in range(self.K):
                #self.b[k]=numpy.ones(shape=(self.J[k],1))
                self.b[k]=self.rng.random_sample(size=(self.J[k],1))
        elif self.hidden_type=="Gaussian_Hinton":
            for k in range(self.K):
                bk=[None]*2
                bk[0]=self.rng.normal(loc=0, scale=0.001, size=(self.J[k],1)) # mean
                bk[1]=1*numpy.ones(shape=(self.J[k],1),dtype=float) # precision
                self.b[k]=bk
        else:
            print("Error! The required distribution for hidden nodes has not implemented yet.")
    
        #initialize bias c
        self.c=numpy.zeros(shape=(self.K,1),dtype=float)
        
        # backup parameters to reset bad parameters
        self.backup_param(0)
        
        
    def backup_param(self, i, dif=240):
        if i==0:
            self.a_backup1=copy.deepcopy(self.a)
            self.b_backup1=copy.deepcopy(self.b)
            self.c_backup1=copy.deepcopy(self.c)
            self.W_backup1=copy.deepcopy(self.W)
            self.a_backup2=copy.deepcopy(self.a)
            self.b_backup2=copy.deepcopy(self.b)
            self.c_backup2=copy.deepcopy(self.c)
            self.W_backup2=copy.deepcopy(self.W)
            self.backup_iter=i
        elif i-self.backup_iter==dif:
            self.a_backup1=copy.deepcopy(self.a_backup2)
            self.b_backup1=copy.deepcopy(self.b_backup2)
            self.c_backup1=copy.deepcopy(self.c_backup2)
            self.W_backup1=copy.deepcopy(self.W_backup2)
            self.a_backup2=copy.deepcopy(self.a)
            self.b_backup2=copy.deepcopy(self.b)
            self.c_backup2=copy.deepcopy(self.c)
            self.W_backup2=copy.deepcopy(self.W)
            self.backup_iter=i
        
        
    def reset_param_use_backup(self, i, dif=120):
        if i-self.backup_iter >= dif:
            self.a=copy.deepcopy(self.a_backup2)
            self.b=copy.deepcopy(self.b_backup2)
            self.c=copy.deepcopy(self.c_backup2)
            self.W=copy.deepcopy(self.W_backup2)
            self.backup_iter=i
        else:
            self.a=copy.deepcopy(self.a_backup1)
            self.b=copy.deepcopy(self.b_backup1)
            self.c=copy.deepcopy(self.c_backup1)
            self.W=copy.deepcopy(self.W_backup1)
            self.backup_iter=i
            
            

    def sample_h_given_xz(self, X, Z, b_hat=None, hidden_value_or_meanfield="value"):
        
        # compute b_hat if not given
        if b_hat is None:
            b_hat=[None]*self.K
            
            if self.hidden_type=="Bernoulli" or self.hidden_type=="Binomial" or self.hidden_type=="Multinomial" or self.hidden_type=="Gaussian_FixPrecision1" or self.hidden_type=="Gaussian_FixPrecision2":
                if self.visible_type=="Multinoulli":                    
                    b_hat=copy.deepcopy(self.b)
                    for k in range(self.K):
                        for m in range(self.M):
                            if self.tie_W_for_pretraining_DBM_bottom:
                                b_hat[k]=b_hat[k] + 2*Z[k,:]*numpy.dot(self.W[k][m].transpose(),X[m])
                            else:
                                b_hat[k]=b_hat[k] + Z[k,:]*numpy.dot(self.W[k][m].transpose(),X[m])
                elif self.visible_type=="Gaussian2":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k]=self.b[k] + 2*Z[k,:]*(numpy.dot(self.W[k][0].transpose(),X) + numpy.dot(self.W[k][1].transpose(),X**2))
                        else:
                            b_hat[k]=self.b[k] + Z[k,:]*(numpy.dot(self.W[k][0].transpose(),X) + numpy.dot(self.W[k][1].transpose(),X**2))
                elif self.visible_type=="Gaussian_Hinton":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k]=self.b[k] + 2*Z[k,:]*numpy.dot(self.W[k].transpose(),numpy.sqrt(self.a[1])*X) # in Hinton's model, X must be scaled
                        else: 
                            b_hat[k]=self.b[k] + Z[k,:]*numpy.dot(self.W[k].transpose(),numpy.sqrt(self.a[1])*X)
                elif self.visible_type=="Gaussian_FixPrecision1":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k]=self.b[k] + 2*Z[k,:]*numpy.dot(self.W[k].transpose(),self.visible_type_fixed_param*X)
                        else:
                            b_hat[k]=self.b[k] + Z[k,:]*numpy.dot(self.W[k].transpose(),self.visible_type_fixed_param*X)
                else: #not a Multinoulli, not Gaussian_Hinton 
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k]=self.b[k] + 2*Z[k,:]*numpy.dot(self.W[k].transpose(),X)
                        else:
                            b_hat[k]=self.b[k] + Z[k,:]*numpy.dot(self.W[k].transpose(),X)
                            
            elif self.hidden_type=="Gaussian":
                if self.visible_type=="Multinoulli":                    
                    b_hat=self.b
                    for k in range(self.K):
                        for m in range(self.M):
                            if self.tie_W_for_pretraining_DBM_bottom:
                                b_hat[k][0]=b_hat[k][0] + 2*Z[k,:]*numpy.dot(self.W[k][m].transpose(),X[m])
                            else:
                                b_hat[k][0]=b_hat[k][0] + Z[k,:]*numpy.dot(self.W[k][m].transpose(),X[m])
                else: #not a Multinoulli, not Gaussian_Hinton 
                    for k in range(self.K):
                        b_hat[k]=[None]*2
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k][0]=self.b[k][0] + 2*Z[k,:]*numpy.dot(self.W[k].transpose(),X)
                        else:
                            b_hat[k][0]=self.b[k][0] + Z[k,:]*numpy.dot(self.W[k].transpose(),X)
                        #b_hat[k][1]=numpy.tile(self.b[k][1],b_hat[k][0].shape[1])
                        b_hat[k][1]=self.b[k][1]
                    
            elif self.hidden_type=="Gaussian_Hinton":
                if self.visible_type=="Multinoulli":                    
                    b_hat=self.b
                    for k in range(self.K):
                        for m in range(self.M):
                            if self.tie_W_for_pretraining_DBM_bottom:
                                b_hat[k][0]=b_hat[k][0] + 2*Z[k,:]*(1/numpy.sqrt(self.b[k][1])*numpy.dot(self.W[k][m].transpose(),X[m]))
                            else:
                                b_hat[k][0]=b_hat[k][0] + Z[k,:]*(1/numpy.sqrt(self.b[k][1])*numpy.dot(self.W[k][m].transpose(),X[m]))
                        #b_hat[k][1]=numpy.tile(self.b[k][1],b_hat[k][0].shape[1]) # self.b[k][1] is a vector, thus we need to repeat it multiple times to align with the size of b_hat[k][0]
                        b_hat[k][1]=self.b[k][1]
                elif self.visible_type=="Gaussian_Hinton":
                    for k in range(self.K):
                        b_hat[k]=[None]*2
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k][0]=self.b[k][0] + 2*Z[k,:]*(1/numpy.sqrt(self.b[k][1])*numpy.dot(self.W[k].transpose(),numpy.sqrt(self.a[1])*X)) # in Hinton's model, X must be scaled
                        else:
                            b_hat[k][0]=self.b[k][0] + Z[k,:]*(1/numpy.sqrt(self.b[k][1])*numpy.dot(self.W[k].transpose(),numpy.sqrt(self.a[1])*X))
                        #b_hat[k][1]=numpy.tile(self.b[k][1],b_hat[k][0].shape[1])
                        b_hat[k][1]=self.b[k][1]
                else: #not a Multinoulli, not Gaussian_Hinton 
                    for k in range(self.K):
                        b_hat[k]=[None]*2
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[k][0]=self.b[k][0] + 2*Z[k,:]*(1/numpy.sqrt(self.b[k][1])*numpy.dot(self.W[k].transpose(),X))
                        else:
                            b_hat[k][0]=self.b[k][0] + Z[k,:]*(1/numpy.sqrt(self.b[k][1])*numpy.dot(self.W[k].transpose(),X))
                        #b_hat[k][1]=numpy.tile(self.b[k][1],b_hat[k][0].shape[1])
                        b_hat[k][1]=self.b[k][1]
        
        # sample h
        P=[None]*self.K
        HM=[None]*self.K
        H=[None]*self.K
        if self.hidden_type=="Bernoulli":
            for k in range(self.K):
                b_hat[k][b_hat[k]<-200]=-200 # to avoid overflow
                P[k]=cl.sigmoid(b_hat[k])
                HM[k]=P[k] # mean of hidden variables
                if hidden_value_or_meanfield=="value":
                    H[k]=cl.Bernoulli_sampling(P[k],rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Binomial":
            for k in range(self.K):
                P[k]=cl.sigmoid(b_hat[k]) # probability
                HM[k]=self.hidden_type_fixed_param*P[k] # mean
                if hidden_value_or_meanfield=="value":
                    # hidden_type_fixed_param is the number of trials
                    H[k]=cl.Binomial_sampling(self.hidden_type_fixed_param, P[k], rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Multinomial":
            for k in range(self.K):
                P[k]=numpy.exp(b_hat[k]) # probability
                P[k]=cl.normalize_probability(P[k])
                HM[k]=self.hidden_type_fixed_param*P[k] # mean
                if hidden_value_or_meanfield=="value":
                    # hidden_type_fixed_param is the number of trials
                    H[k]=cl.Multinomial_sampling(self.hidden_type_fixed_param, P=P[k], rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Gaussian":
            for k in range(self.K):
                HM[k]=b_hat[k][0]/(-2*b_hat[k][1] + 1e-10) # to prevent overflow
                if hidden_value_or_meanfield=="value":
                    H[k]=cl.Gaussian_sampling(HM[k],-2*b_hat[k][1] + 1e-10,rng=self.rng)
                    #H[k]=cl.Gaussian_samplingP(HM[k],-2*b_hat[k][1], n_jobs=20, rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Gaussian2":
            for k in range(self.K):
                HM[k]=b_hat[k][0]/(-2*b_hat[k][1] + 1e-10) # to prevent overflow
                if hidden_value_or_meanfield=="value":
                    H[k]=cl.Gaussian_sampling(HM[k],-2*b_hat[k][1] + 1e-10,rng=self.rng)
                    #H[k]=cl.Gaussian_samplingP(HM[k],-2*b_hat[k][1], n_jobs=20, rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Gaussian_Hinton":
            for k in range(self.K):
                HM[k]=b_hat[k][0]
                if hidden_value_or_meanfield=="value":
                    H[k]=cl.Gaussian_sampling(b_hat[k][0],b_hat[k][1],rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Gaussian_FixPrecision1":
            for k in range(self.K):
                HM[k]=b_hat[k]
                if hidden_value_or_meanfield=="value":
                    H[k]=cl.Gaussian_sampling(b_hat[k],self.hidden_type_fixed_param,rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        elif self.hidden_type=="Gaussian_FixPrecision2":
            for k in range(self.K):
                HM[k]=b_hat[k]/self.hidden_type_fixed_param
                if hidden_value_or_meanfield=="value":
                    H[k]=cl.Gaussian_sampling(HM[k],self.hidden_type_fixed_param,rng=self.rng)
                elif hidden_value_or_meanfield=="meanfield":
                    H[k]=HM[k]
        return H,HM
    
            
    def sample_z_given_xh(self, X, H, c_hat=None, hidden_value_or_meanfield="value"):
        if c_hat is None:
            if self.visible_type=="Multinoulli":
                num_samples=X[0].shape[0]
            else:
                num_samples=X.shape[1]
            c_hat=numpy.tile(self.c, num_samples)
            for n in range(num_samples):
                for k in range(self.K):
                    if self.visible_type=="Gaussian_Hinton":
                        tx=numpy.sqrt(self.a[1])
                    elif self.visible_type=="Gaussian_FixPrecision1":
                        tx=self.visible_type_fixed_param
                    else:
                        tx=1
                        
                    if self.hidden_type=="Gaussian_Hinton":
                        th=numpy.sqrt(self.b[k][1])
                    elif self.hidden_type=="Gaussian_FixPrecision1":
                        th=self.hidden_type_fixed_param
                    else:
                        th=1
                    if self.visible_type=="Multinoulli":
                        for m in range(self.M):
                            if self.tie_W_for_pretraining_DBM_bottom:
                                c_hat[k,n]=c_hat[k,n] + 2*numpy.dot((tx*X[m][:,[n]]).transpose(),self.W[k][m]).dot(th*H[k][:,[n]])
                            else:
                                c_hat[k,n]=c_hat[k,n] + numpy.dot((tx*X[m][:,[n]]).transpose(),self.W[k][m]).dot(th*H[k][:,[n]])
                    elif self.visible_type=="Gaussian2":
                        if self.tie_W_for_pretraining_DBM_bottom:
                            c_hat[k,n]=c_hat[k,n] + 2*( numpy.dot((tx*X[:,[n]]).transpose(),self.W[k][0]).dot(th*H[k][:,[n]]) + numpy.dot((tx*X[:,[n]]**2).transpose(),self.W[k][1]).dot(th*H[k][:,[n]]) )
                        else:
                            c_hat[k,n]=c_hat[k,n] + numpy.dot((tx*X[:,[n]]).transpose(),self.W[k][0]).dot(th*H[k][:,[n]]) + numpy.dot((tx*X[:,[n]]**2).transpose(),self.W[k][1]).dot(th*H[k][:,[n]])
                    else:
                        if self.tie_W_for_pretraining_DBM_bottom:
                            c_hat[k,n]=c_hat[k,n] + 2*numpy.dot((tx*X[:,[n]]).transpose(),self.W[k]).dot(th*H[k][:,[n]])
                        else:
                            c_hat[k,n]=c_hat[k,n] + numpy.dot((tx*X[:,[n]]).transpose(),self.W[k]).dot(th*H[k][:,[n]])
            
            if self.xz_interaction:
                if self.tie_W_for_pretraining_DBM_bottom:
                    c_hat= c_hat + 2*numpy.dot(self.W[self.K].transpose(),X)
                else:
                    c_hat= c_hat + numpy.dot(self.W[self.K].transpose(),X)
                   
        c_hat[c_hat<-200]=-200 # prevent overflow
        ZP=cl.sigmoid(c_hat)
        ZM=ZP # mean of hidden variable z
        if hidden_value_or_meanfield=="value":
            Z=cl.Bernoulli_sampling(ZP,rng=self.rng)
        else:
            Z=ZM
        return  Z,ZM
    
    def sample_z_given_xh2(self, X, H, c_hat=None, hidden_value_or_meanfield="value"):
        if c_hat is None:
            if self.visible_type=="Multinoulli":
                num_samples=X[0].shape[0]
            else:
                num_samples=X.shape[1]
            c_hat=numpy.tile(self.c, num_samples)
        
            for k in range(self.K):
                if self.visible_type=="Gaussian_Hinton":
                    tx=numpy.sqrt(self.a[1])
                elif self.visible_type=="Gaussian_FixPrecision1":
                    tx=self.visible_type_fixed_param
                else:
                    tx=1
                    
                if self.hidden_type=="Gaussian_Hinton":
                    th=numpy.sqrt(self.b[k][1])
                elif self.hidden_type=="Gaussian_FixPrecision1":
                    th=self.hidden_type_fixed_param
                else:
                    th=1
                if self.visible_type=="Multinoulli":
                    for m in range(self.M):
                        if self.tie_W_for_pretraining_DBM_bottom:
                            c_hat[k,:]=c_hat[k,:] + 2*numpy.diag( numpy.dot((tx*X[m]).transpose(),self.W[k][m]).dot(th*H[k]) )
                        else:
                            c_hat[k,:]=c_hat[k,:] + numpy.diag( numpy.dot((tx*X[m]).transpose(),self.W[k][m]).dot(th*H[k]) )
                elif self.visible_type=="Gaussian2":
                    if self.tie_W_for_pretraining_DBM_bottom:
                        c_hat[k,:]=c_hat[k,:] + 2*numpy.diag( numpy.dot((tx*X).transpose(),self.W[k][0]).dot(th*H[k]) + numpy.dot((tx*X**2).transpose(),self.W[k][1]).dot(th*H[k]) )
                    else:
                        c_hat[k,:]=c_hat[k,:] + numpy.diag( numpy.dot((tx*X).transpose(),self.W[k][0]).dot(th*H[k]) + numpy.dot((tx*X**2).transpose(),self.W[k][1]).dot(th*H[k]) )
                else:
                    if self.tie_W_for_pretraining_DBM_bottom:
                        c_hat[k,:]=c_hat[k,:] + 2*numpy.diag( numpy.dot((tx*X).transpose(),self.W[k]).dot(th*H[k]) )
                    else:
                        c_hat[k,:]=c_hat[k,:] + numpy.diag( numpy.dot((tx*X).transpose(),self.W[k]).dot(th*H[k]) )
            
            if self.xz_interaction:
                if self.tie_W_for_pretraining_DBM_bottom:
                    c_hat= c_hat + 2*numpy.dot(self.W[self.K].transpose(),X)
                else:
                    c_hat= c_hat + numpy.dot(self.W[self.K].transpose(),X)
                
        c_hat[c_hat<-200]=-200 # prevent overflow
        ZP=cl.sigmoid(c_hat)
        ZM=ZP # mean of hidden variable z
        if hidden_value_or_meanfield=="value":
            Z=cl.Bernoulli_sampling(ZP,rng=self.rng)
        else:
            Z=ZM
        return  Z,ZM    
    
    
    def sample_x_given_hz(self, H, Z, a_hat=None, value_or_meanfield="value"):
        if a_hat is None:
            num_samples=Z.shape[1]
            if self.visible_type=="Bernoulli" or self.visible_type=="Poisson" or self.visible_type=="Binomial" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision2":
                a_hat=numpy.tile(self.a, num_samples)
                if self.hidden_type=="Gaussian_Hinton":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*Z[k,:]*numpy.dot(self.W[k],numpy.sqrt(self.b[k][1])*H[k])
                        else:
                            a_hat = a_hat + Z[k,:]*numpy.dot(self.W[k],numpy.sqrt(self.b[k][1])*H[k])
                    # consider x z interactions
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat = a_hat + numpy.dot(self.W[self.K],Z)
                elif self.hidden_type=="Gaussian_FixPrecision1":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*Z[k,:]*numpy.dot(self.self.W[k],self.hidden_type_fixed_param*H[k])
                        else:
                            a_hat = a_hat + Z[k,:]*numpy.dot(self.W[k],self.hidden_type_fixed_param*H[k])
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat = a_hat + numpy.dot(self.W[self.K],Z)
                else:
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*Z[k,:]*numpy.dot(self.W[k],H[k])
                        else:
                            a_hat = a_hat + Z[k,:]*numpy.dot(self.W[k],H[k])
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat = a_hat + numpy.dot(self.W[self.K],Z)
            elif self.visible_type=="Gaussian_FixPrecision1":
                a_hat=numpy.tile(self.a, num_samples)
                if self.hidden_type=="Gaussian_FixPrecision1":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*Z[k,:]*numpy.dot(self.W[k],self.hidden_type_fixed_param*H[k])
                        else:
                            a_hat = a_hat + Z[k,:]*numpy.dot(self.W[k],self.hidden_type_fixed_param*H[k])
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat = a_hat + numpy.dot(self.W[self.K],Z)
                else:
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*Z[k,:]*numpy.dot(self.W[k],H[k])
                        else:
                            a_hat = a_hat + Z[k,:]*numpy.dot(self.W[k],H[k])
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat = a_hat + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat = a_hat + numpy.dot(self.W[self.K],Z)
            elif self.visible_type=="Gaussian":
                a_hat1=numpy.tile(self.a[0],num_samples)
                a_hat2=self.a[1]
                for k in range(self.K):
                    if self.tie_W_for_pretraining_DBM_top:
                        a_hat1 = a_hat1 + 2*Z[k,:]*numpy.dot(self.W[k],H[k])
                    else:
                        a_hat1 = a_hat1 + Z[k,:]*numpy.dot(self.W[k],H[k])
                if self.xz_interaction:
                    if self.tie_W_for_pretraining_DBM_top:
                        a_hat1 = a_hat1 + 2*numpy.dot(self.W[self.K],Z)
                    else:
                        a_hat1 = a_hat1 + numpy.dot(self.W[self.K],Z)
                a_hat=[a_hat1,a_hat2]
            elif self.visible_type=="Gaussian2":
                a_hat1=numpy.tile(self.a[0],num_samples)
                a_hat2=numpy.tile(self.a[1],num_samples)
                for k in range(self.K):
                    if self.tie_W_for_pretraining_DBM_top:
                        a_hat1 = a_hat1 + 2*Z[k,:]*numpy.dot(self.W[k][0],H[k])
                        a_hat2 = a_hat2 + 2*Z[k,:]*numpy.dot(self.W[k][1],H[k])
                    else:
                        a_hat1 = a_hat1 + Z[k,:]*numpy.dot(self.W[k][0],H[k])
                        a_hat2 = a_hat2 + Z[k,:]*numpy.dot(self.W[k][1],H[k])
                if self.xz_interaction:
                    if self.tie_W_for_pretraining_DBM_top:
                        a_hat1 = a_hat1 + 2*numpy.dot(self.W[self.K][0],Z)
                        a_hat2 = a_hat2 + 2*numpy.dot(self.W[self.K][1],Z)
                    else:
                        a_hat1 = a_hat1 + numpy.dot(self.W[self.K][0],Z)
                        a_hat2 = a_hat2 + numpy.dot(self.W[self.K][1],Z)
                a_hat=[a_hat1,a_hat2]
            elif self.visible_type=="Gaussian_Hinton":
                a_hat1=numpy.tile(self.a[0],num_samples)
                a_hat2=self.a[1]
                if self.hidden_type=="Gaussian_Hinton":
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat1 = a_hat1 + 2*Z[k,:]*(1/numpy.sqrt(self.a[1])*numpy.dot(self.W[k],numpy.sqrt(self.b[k][1])*H[k]))
                        else:
                            a_hat1 = a_hat1 + Z[k,:]*(1/numpy.sqrt(self.a[1])*numpy.dot(self.W[k],numpy.sqrt(self.b[k][1])*H[k]))
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat1 = a_hat1 + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat1 = a_hat1 + numpy.dot(self.W[self.K],Z)
                else:
                    for k in range(self.K):
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat1 = a_hat1 + 2*Z[k,:]*(1/numpy.sqrt(self.a[1])*numpy.dot(self.W[k],H[k]))
                        else:
                            a_hat1 = a_hat1 + Z[k,:]*(1/numpy.sqrt(self.a[1])*numpy.dot(self.W[k],H[k]))
                    if self.xz_interaction:
                        if self.tie_W_for_pretraining_DBM_top:
                            a_hat1 = a_hat1 + 2*numpy.dot(self.W[self.K],Z)
                        else:
                            a_hat1 = a_hat1 + numpy.dot(self.W[self.K],Z)
                a_hat=[a_hat1,a_hat2]
                
            elif self.visible_type=="Multinoulli":
                a_hat=[None]*self.M
                for m in range(self.M):
                    a_hatm=numpy.tile(self.a[m],num_samples)
                    if self.hidden_type=="Gaussian_Hinton":
                        for k in range(self.K):
                            if self.tie_W_for_pretraining_DBM_top:
                                a_hatm = a_hatm + 2*Z[k,:]*numpy.dot(self.W[k][m],numpy.sqrt(self.b[k][1])*H[k])
                            else:
                                a_hatm = a_hatm + Z[k,:]*numpy.dot(self.W[k][m],numpy.sqrt(self.b[k][1])*H[k])
                        if self.xz_interaction:
                            if self.tie_W_for_pretraining_DBM_top:
                                a_hatm = a_hatm + 2*numpy.dot(self.W[self.K][m],Z)
                            else:
                                a_hatm = a_hatm + numpy.dot(self.W[self.K][m],Z)
                    elif self.hidden_type=="Gaussian_FixPrecision1":
                        for k in range(self.K):
                            if self.tie_W_for_pretraining_DBM_top:
                                a_hatm = a_hatm + 2*Z[k,:]*numpy.dot(self.W[k][m],self.hidden_type_fixed_param*H[k])
                            else:
                                a_hatm = a_hatm + Z[k,:]*numpy.dot(self.W[k][m],self.hidden_type_fixed_param*H[k])
                        if self.xz_interaction:
                            if self.tie_W_for_pretraining_DBM_top:
                                a_hatm = a_hatm + 2*numpy.dot(self.W[self.K][m],Z)
                            else:
                                a_hatm = a_hatm + numpy.dot(self.W[self.K][m],Z)
                    else:
                        for k in range(self.K):
                            if self.tie_W_for_pretraining_DBM_top:
                                a_hatm = a_hatm + 2*Z[k,:]*numpy.dot(self.W[k][m],H[k])
                            else:
                                a_hatm = a_hatm + Z[k,:]*numpy.dot(self.W[k][m],H[k])
                        if self.xz_interaction:
                            if self.tie_W_for_pretraining_DBM_top:
                                a_hatm = a_hatm + 2*numpy.dot(self.W[self.K][m],Z)
                            else:
                                a_hatm = a_hatm + numpy.dot(self.W[self.K][m],Z)
                    a_hat[m]=a_hatm
    
        # sample x
        if self.visible_type=="Bernoulli":
            a_hat[a_hat<-200]=-200
            P=cl.sigmoid(a_hat)
            XM=P
            if value_or_meanfield=="value":
                X=cl.Bernoulli_sampling(P,rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Binomial":
            P=cl.sigmoid(a_hat)
            XM=self.visible_type_fixed_param*P
            if value_or_meanfield=="value":
                X=cl.Binomial_sampling(self.visible_type_fixed_param, P, rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Multinomial":
            P=numpy.exp(a_hat)
            P=cl.normalize_probability(P)
            XM=self.visible_type_fixed_param*(P)
            if value_or_meanfield=="value":
                X=cl.Multinomial_sampling(N=self.visible_type_fixed_param,P=P,rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Poisson":
            a_hat[a_hat>self.tol_poisson_max]=self.tol_poisson_max
            #a_hat[a_hat>8]=8 # for impages
            XM=numpy.exp(a_hat) # a_hat here must be small, if very big, overflow problem will raise.
            P=None
            if value_or_meanfield=="value":
                X=cl.Poisson_sampling(XM,rng=self.rng)
            else:
                X=XM
#        elif self.visible_type=="NegativeBinomial": 
#            tol_negbin_max=-1e-8
#            tol_negbin_min=-100
#            a_hat[a_hat>=0]=tol_negbin_max
#            a_hat[a_hat<tol_negbin_min]=tol_negbin_min
#            P_failure=numpy.exp(a_hat)
#            P=P_failure # also return the probability of failure
#            P_success=1-P_failure
#            XM=self.visible_type_fixed_param*(P_failure/P_success)
#            X=cl.NegativeBinomial_sampling(K=self.visible_type_fixed_param,P=P_success,rng=self.rng)
        elif self.visible_type=="Gaussian":
            XM=a_hat[0]/(-2*a_hat[1]+ 1e-10)# prevent overflow
            P=None
            if value_or_meanfield=="value":
                X=cl.Gaussian_sampling(XM,-2*a_hat[1]+ 1e-10,rng=self.rng)
                #X=cl.Gaussian_samplingP(XM, -2*a_hat[1], n_jobs=20, rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Gaussian2":
            XM=-a_hat[0]/(2*a_hat[1])
            P=None
            if value_or_meanfield=="value":
                X=cl.Gaussian_sampling2(XM,-2*a_hat[1],rng=self.rng)
                #X=cl.Gaussian_samplingP(XM, -2*a_hat[1], n_jobs=20, rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Gaussian_Hinton":
            XM=a_hat[0]
            P=None
            if value_or_meanfield=="value":
                X=cl.Gaussian_sampling(a_hat[0],a_hat[1],rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Gaussian_FixPrecision1":
            XM=a_hat
            P=None
            if value_or_meanfield=="value":
                X=cl.Gaussian_sampling(XM,self.visible_type_fixed_param,rng=self.rng)
            else:
                X=XM
        elif self.visible_type=="Gaussian_FixPrecision2":
            XM=a_hat/self.visible_type_fixed_param
            P=None
            if value_or_meanfield=="value":
                X=cl.Gaussian_sampling(XM,self.visible_type_fixed_param,rng=self.rng)
            else:
                X=XM
        return X,XM
    
    
    def mean_field_approximate_inference(self, Xbatch, NMF=20, Zto1=False):
        print("in mean_field_approximate_inference ...")
        if self.visible_type=="Multinoulli":
            NS=Xbatch[0].shape[1]
        else:
            NS=Xbatch.shape[1]
            
        self.NMF=NMF
            
        # initialize h and z
#        b=self.repeat_b(self.b, NS)
#        _,HbatchM=self.sample_h_given_xz(X=None, Z=None, b_hat=b, hidden_value_or_meanfield="meanfield")
#        _,ZbatchM=self.sample_z_given_xh2(X=Xbatch, H=HbatchM, c_hat=None, hidden_value_or_meanfield="meanfield")
        if Zto1:
            ZbatchM=numpy.ones((self.K,NS))
        else:
            _,ZbatchM=self.sample_z_given_xh2(X=None, H=None, c_hat=numpy.tile(self.c,NS)+numpy.random.normal(0,0.1,(self.K,NS)), hidden_value_or_meanfield="meanfield")
        _,HbatchM=self.sample_h_given_xz(X=Xbatch, Z=ZbatchM, b_hat=None, hidden_value_or_meanfield="meanfield")
        
        if Zto1:
            _,HbatchM=self.sample_h_given_xz(X=Xbatch, Z=ZbatchM, b_hat=None, hidden_value_or_meanfield="meanfield")
        else:
            for n in range(NMF):

    #            _,HbatchM=self.sample_h_given_xz(X=Xbatch, Z=ZbatchM, b_hat=None, hidden_value_or_meanfield="meanfield")
    #            _,ZbatchM=self.sample_z_given_xh2(X=Xbatch, H=HbatchM, c_hat=None, hidden_value_or_meanfield="meanfield")
                _,ZbatchM=self.sample_z_given_xh2(X=Xbatch, H=HbatchM, c_hat=None, hidden_value_or_meanfield="meanfield")
                _,HbatchM=self.sample_h_given_xz(X=Xbatch, Z=ZbatchM, b_hat=None, hidden_value_or_meanfield="meanfield")
 
        _,XbatchM=self.sample_x_given_hz(H=HbatchM, Z=ZbatchM, a_hat=None, value_or_meanfield="meanfield")
        
        return XbatchM,HbatchM,ZbatchM

    
    def pcd_sampling(self, pcdk=20, NS=20, X0=None, persistent=True, rand_init=False, init=False, Zto1=False): 
        "persistent cd sampling"
        
        print("in pcd_sampling ...")
        if not persistent:
            if X0 is not None: # X0 should be the current mini-batch
                init=True
                rand_init=False
            else:
                print("Error! You want to use CD-k sampling, but you did not give me a batch of training samples.")
                sys.exit(0)

        if init:
            self.NS=NS
            # initialize Markov chains
            if rand_init:
                a=self.repeat_a(self.a,self.NS)
                Xs,XM=self.sample_x_given_hz(H=None, Z=None, a_hat=a, value_or_meanfield="value")
                X0=Xs
            else: # not randomly initialize X, then sample a subset of samples from the training set
                if X0 is None:
                    X0=self.sample_minibatch(self.NS)
                XM=X0
                    
            self.chainX=X0
            self.chainXM=XM
            
#            b=self.repeat_b(self.b, self.NS)
#            H0,HM=self.sample_h_given_xz(X=None, Z=None, b_hat=b, hidden_value_or_meanfield="value")
#            self.chainH=H0
#            self.chainHM=HM
#            
#            Z0,ZM=self.sample_z_given_xh2(X=X0, H=H0, c_hat=None, hidden_value_or_meanfield="value")
#            self.chainZ=Z0
#            self.chainZM=ZM
            
            if Zto1:
                Z0=numpy.ones((self.K,self.NS))
                ZM=numpy.ones((self.K,self.NS))
            else:
                Z0,ZM=self.sample_z_given_xh2(X=None, H=None, c_hat=numpy.tile(self.c,self.NS), hidden_value_or_meanfield="value")
            #self.chainZ=Z0
            self.chainZ=ZM
#            self.chainZ=ZM
            self.chainZM=ZM
            
            #H0,HM=self.sample_h_given_xz(X=X0, Z=Z0, b_hat=None, hidden_value_or_meanfield="value")
            H0,HM=self.sample_h_given_xz(X=X0, Z=ZM, b_hat=None, hidden_value_or_meanfield="value")
            self.chainH=H0
            self.chainHM=HM
                        
            self.chain_length=0
        
        for s in range(pcdk):
#            self.chainX,self.chainXM=self.sample_x_given_hz(self.chainH, self.chainZ)
#            self.chainH,self.chainHM=self.sample_h_given_xz(self.chainX, self.chainZ)
#            self.chainZ,self.chainZM=self.sample_z_given_xh2(self.chainX, self.chainH)
            self.chainX,self.chainXM=self.sample_x_given_hz(self.chainH, self.chainZ)
            if Zto1:
                self.chainZ=numpy.ones((self.K,self.NS))
                self.chainZM=numpy.ones((self.K,self.NS))
            else:
                self.chainZ,self.chainZM=self.sample_z_given_xh2(self.chainX, self.chainH)
                #self.chainZ=self.chainZM
            self.chainH,self.chainHM=self.sample_h_given_xz(self.chainX, self.chainZ)
            self.chain_length=self.chain_length+1

        return self.chainX,self.chainH,self.chainZ,self.chainXM,self.chainHM,self.chainZM,self.chain_length


    def pcd_sampling_fix_z(self, pcdk=20, NS=20, X0=None, Z0=None, persistent=True, rand_init=False, init=False): 
        "persistent cd sampling"
        
        print("in pcd_sampling ...")
        if not persistent:
            if X0 is not None: # X0 should be the current mini-batch
                init=True
                rand_init=False
            else:
                print("Error! You want to use CD-k sampling, but you did not give me a batch of training samples.")
                sys.exit(0)

        if init:
            self.NS=NS
            # initialize Markov chains
            if rand_init:
                a=self.repeat_a(self.a,self.NS)
                Xs,XM=self.sample_x_given_hz(H=None, Z=None, a_hat=a, value_or_meanfield="value")
                X0=Xs
            else: # not randomly initialize X, then sample a subset of samples from the training set
                if X0 is None:
                    X0=self.sample_minibatch(self.NS)
                XM=X0
                    
            self.chainX=X0
            self.chainXM=XM
            
            self.chainZ=Z0
            self.chainZM=Z0
            
            #H0,HM=self.sample_h_given_xz(X=X0, Z=Z0, b_hat=None, hidden_value_or_meanfield="value")
            H0,HM=self.sample_h_given_xz(X=X0, Z=Z0, b_hat=None, hidden_value_or_meanfield="value")
            self.chainH=H0
            self.chainHM=HM
                        
            self.chain_length=0
        
        for s in range(pcdk):

            self.chainX,self.chainXM=self.sample_x_given_hz(self.chainH, self.chainZ)
            self.chainH,self.chainHM=self.sample_h_given_xz(self.chainX, self.chainZ)
            self.chain_length=self.chain_length+1

        return self.chainX,self.chainH,self.chainXM,self.chainHM,self.chain_length
        

    def repeat_a(self, a, N):
        # repeat a N times column-wise
        if self.visible_type=="Bernoulli" or self.visible_type=="Poisson" or self.visible_type=="Binomial" or self.visible_type=="NegativeBinomial" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision1" or self.visible_type=="Gaussian_FixPrecision2":
            a_rep=numpy.repeat(a,N,axis=1)
        elif self.visible_type=="Gaussian" or self.visible_type=="Gaussian_Hinton":
            a_rep=[None,None]
            a_rep[0]=numpy.repeat(a[0],N,axis=1)
            a_rep[1]=a[1]
        elif self.visible_type=="Gaussian2":
            a_rep=[None,None]
            a_rep[0]=numpy.repeat(a[0],N,axis=1)
            a_rep[1]=numpy.repeat(a[1],N,axis=1)
        elif self.visible_type=="Gamma":
            a_rep=[None,None]
            a_rep[0]=a[0]
            a_rep[1]=numpy.repeat(a[1],N,axis=1)
        elif self.visible_type=="Multinoulli":
            a_rep=[None]*self.M
            for m in range(self.M):
                a_rep[m]=numpy.repeat(a[m],N,axis=1)
        return a_rep


    def repeat_b(self, b, N):
        # repeat b N times column-wise
        b_rep=[None]*self.K
        if self.hidden_type=="Bernoulli" or self.hidden_type=="Poisson" or self.hidden_type=="NegativeBinomial" or self.hidden_type=="Binomial" or self.hidden_type=="Multinomial" or self.hidden_type=="Gaussian_FixPrecision1" or self.hidden_type=="Gaussian_FixPrecision2":
            for k in range(self.K):
                b_rep[k]=numpy.tile(b[k],N)
        elif self.hidden_type=="Gaussian" or self.hidden_type=="Gaussian_Hinton":
            for k in range(self.K):
                b_rep[k]=[None,None]
                b_rep[k][0]=numpy.tile(b[k][0], N)
                b_rep[k][1]=b[k][1]
        return b_rep
        

    def compute_reconstruction_error(self, X0, X0RM=None):
        """
        Compute the difference between the real sample X0 and the recoverd sample X0RM by mean-field.
        """
        if X0RM is None:
            X0RM,_,_=self.mean_field_approximate_inference(X0, NMF=self.NMF)
        if self.visible_type=="Multinoulli":
            self.rec_error=0
            for m in range(self.M):
                self.rec_error=self.rec_error+numpy.mean(numpy.abs(X0RM[m]-X0[m]))
        else:
            self.rec_error=numpy.mean(numpy.abs(X0RM-X0))
        return self.rec_error
        
    def compute_gradient(self, Xbatch, Hbatch, Zbatch, XS, HS, ZS):
        """
        Compute gradient.
        Assume the hidden type is Bernoulli, Binomial, Multinomial, Gaussian_FixPrecision2.
        """
        
        print("in compute_gradient...")
        # grad for a
        if self.visible_type=="Bernoulli" or self.visible_type=="Poisson" or self.visible_type=="Binomial" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision2":
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a=data_dep - data_indep
            grad_a.shape=(self.M,1)
            
        elif self.visible_type=="Gaussian_FixPrecision1":
            data_dep=-numpy.mean(self.visible_type_fixed_param*Xbatch,axis=1)
            data_indep=-numpy.mean(self.visible_type_fixed_param*XS,axis=1)
            grad_a=data_dep - data_indep
            grad_a.shape=(self.M,1)
            
        elif self.visible_type=="Gaussian" or self.visible_type=="Gaussian2":
            # gradient of a1
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a1=data_dep - data_indep
            grad_a1.shape=(self.M,1)
            
            # gradient of a2
            data_dep=-numpy.mean(Xbatch**2,axis=1)
            data_indep=-numpy.mean(XS**2,axis=1)
            grad_a2=data_dep - data_indep
            grad_a2.shape=(self.M,1)
            grad_a=[grad_a1, grad_a2]
        
        elif self.visible_type=="Gaussian_Hinton":
            # grad of mean
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_dep.shape=(self.M,1)
            data_indep=-numpy.mean(XS,axis=1)
            data_indep.shape=(self.M,1)
            grad_a1=self.a[1]*(data_dep - data_indep)

            # grad of precision
            grad_a2=0
            for k in range(self.K):
                data_dep_a2=numpy.mean((Xbatch-self.a[0])**2,axis=1) - numpy.mean(1/numpy.sqrt(self.a[1])*Xbatch*self.W[k].dot(Hbatch[k]),axis=1)
                data_indep_a2=numpy.mean((XS-self.a[0])**2,axis=1) - numpy.mean(1/numpy.sqrt(self.a[1])*XS*self.W[k].dot(HS[k]),axis=1)
                grad_a2=grad_a2 + 0.5*(data_dep_a2 - data_indep_a2)
            grad_a2.shape=(self.M,1)
            
            grad_a=[grad_a1, grad_a2]
        
        elif self.visible_type=="Multinoulli":
            grad_a=[None]*self.M
            for m in range(self.M):
                # gradient of a
                data_dep=-numpy.mean(Xbatch[m],axis=1)
                data_indep=-numpy.mean(XS[m],axis=1)
                grad_a[m]=data_dep - data_indep
                grad_a[m].shape=(self.I[m],1)
            
        # grad for b
        grad_b=[None]*self.K
        if self.hidden_type=="Bernoulli" or self.hidden_type=="Poisson" or self.hidden_type=="Binomial" or self.hidden_type=="Multinomial" or self.hidden_type=="Gaussian_FixPrecision2":
            for k in range(self.K):
                    data_dep=-numpy.mean(Hbatch[k],axis=1)
                    data_indep=-numpy.mean(HS[k],axis=1)
                    grad_bk=data_dep - data_indep
                    grad_bk.shape=(self.J[k],1)
                    grad_b[k]=grad_bk
            
        elif self.hidden_type=="Gaussian_FixPrecision1":
            for k in range(self.K):
                data_dep=-numpy.mean(self.hidden_type_fixed_param*Hbatch[k],axis=1)
                data_indep=-numpy.mean(self.hidden_type_fixed_param*HS[k],axis=1)
                grad_bk=data_dep - data_indep
                grad_bk.shape=(self.J[k],1)
                grad_b[k]=grad_bk
            
        elif self.hidden_type=="Gaussian":
            for k in range(self.K):
                # gradient of b1
                data_dep=-numpy.mean(Hbatch[k],axis=1)
                data_indep=-numpy.mean(HS[k],axis=1)
                grad_b1k=data_dep - data_indep
                grad_b1k.shape=(self.J[k],1)
                
                # gradient of b2
                data_dep=-numpy.mean(Hbatch[k]**2,axis=1)
                data_indep=-numpy.mean(HS[k]**2,axis=1)
                grad_b2k=data_dep - data_indep
                grad_b2k.shape=(self.J[k],1)
                
                grad_b[k]=[grad_b1k, grad_b2k]
        
        elif self.hidden_type=="Gaussian_Hinton":
            for k in range(self.K):
                # grad of mean
                data_dep=-numpy.mean(Hbatch[k],axis=1)
                data_dep.shape=(self.J[k],1)
                data_indep=-numpy.mean(HS[k],axis=1)
                data_indep.shape=(self.J[k],1)
                grad_b1k=self.b[k][1]*(data_dep - data_indep)
    
                # grad of precision
                data_dep_b2k=numpy.mean((Hbatch[k]-self.b[k][0])**2,axis=1) - numpy.mean(1/numpy.sqrt(self.b[k][1])*Hbatch[k]*self.W[k].transpose().dot(Xbatch),axis=1)
                data_indep_b2k=numpy.mean((HS[k]-self.b[k][0])**2,axis=1) - numpy.mean(1/numpy.sqrt(self.b[k][1])*HS[k]*self.W[k].transpose().dot(XS),axis=1)
                grad_b2k=0.5*(data_dep_b2k - data_indep_b2k)
                grad_b2k.shape=(self.J[k],1)
                
                grad_b[k]=[grad_b1k, grad_b2k]
    
        # grad for c
        data_dep=-numpy.mean(Zbatch,axis=1)
        data_indep=-numpy.mean(ZS,axis=1)
        grad_c=data_dep - data_indep
        grad_c.shape=(self.K,1)
    
        # grad for W
        grad_W=[None]*(self.K+1)
        for k in range(self.K):
            if self.visible_type=="Gaussian_Hinton":
                tx=numpy.sqrt(self.a[1])
            elif self.visible_type=="Gaussian_FixPrecision1":
                tx=self.visible_type_fixed_param
            else:
                tx=1
                        
            if self.hidden_type=="Gaussian_Hinton":
                th=numpy.sqrt(self.b[k][1])
            elif self.hidden_type=="Gaussian_FixPrecision1":
                th=self.hidden_type_fixed_param
            else:
                th=1
                        
            if self.visible_type=="Multinoulli":
                grad_Wk=[None]*self.M
                for m in range(self.M):
                    data_dep=-numpy.dot(tx*Xbatch[m],(th*Zbatch[k,:]*Hbatch[k]).transpose())/self.batch_size
                    data_indep=-numpy.dot(tx*XS[m],(th*ZS[k,:]*HS[k]).transpose())/self.NS
                    grad_Wk[m]=data_dep - data_indep
                grad_W[k]=grad_Wk
            elif self.visible_type=="Gaussian2":
                data_dep=-numpy.dot(tx*Xbatch,(th*Zbatch[k,:]*Hbatch[k]).transpose())/self.batch_size
                data_indep=-numpy.dot(tx*XS,(th*ZS[k,:]*HS[k]).transpose())/self.NS
                grad_W1k=data_dep - data_indep
                data_dep=-numpy.dot(tx*Xbatch**2,(th*Zbatch[k,:]*Hbatch[k]).transpose())/self.batch_size
                data_indep=-numpy.dot(tx*XS**2,(th*ZS[k,:]*HS[k]).transpose())/self.NS
                grad_W2k=data_dep - data_indep
                grad_W[k]=[grad_W1k,grad_W2k]
                
            else:
                data_dep=-numpy.dot(tx*Xbatch,(th*Zbatch[k,:]*Hbatch[k]).transpose())/self.batch_size
                data_indep=-numpy.dot(tx*XS,(th*ZS[k,:]*HS[k]).transpose())/self.NS
                grad_Wk=data_dep - data_indep
                grad_W[k]=grad_Wk
        
        # W for XWZ
        if self.xz_interaction:
            data_dep=-numpy.dot(tx*Xbatch,Zbatch.transpose())/self.batch_size
            data_indep=-numpy.dot(tx*XS,ZS.transpose())/self.NS
            grad_W[self.K]=data_dep - data_indep
        
        self.grad_a=grad_a
        self.grad_b=grad_b
        self.grad_c=grad_c
        self.grad_W=grad_W
        
    
    def update_param(self):
        
        print("in update_param...")
        tol=1e-8
        tol_negbin_min=-20
        tol_poisson_max=self.tol_poisson_max#16 #numpy.log(255)
        tol_gamma_min=1e-3
        tol_gamma_max=1e3
        if self.if_fix_vis_bias:
            fix_a_log_ind=self.fix_a_log_ind
            not_fix_a_log_ind=numpy.logical_not(fix_a_log_ind)
            not_fix_a_log_ind=numpy.array(not_fix_a_log_ind, dtype=int)
            not_fix_a_log_ind.shape=(len(not_fix_a_log_ind),1)
            
        # update a
        if self.visible_type=="Bernoulli" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision1" or self.visible_type=="Gaussian_FixPrecision2":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else: # fix some a
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            print("in update_param ...")
            print(self.if_fix_vis_bias)
            print(self.a[0:10].transpose())

        elif self.visible_type=="Poisson":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            self.a[self.a>tol_poisson_max]=tol_poisson_max
            print("in update_param ...")
            print(self.a[0:10])
            
        elif self.visible_type=="NegativeBinomial":
            #print "before update ..."
            #print self.a[0:10]
            #print self.grad_a
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            # a not too small, not positive,s [-20,0)
            self.a[self.a>=0]=-tol # project a to negative
            self.a[self.a<tol_negbin_min]=tol_negbin_min
            #self.W=self.W - self.learn_rate_W * self.grad_W
            #self.W[self.W>0]=0 # project W to negative
            print("in update_param ...")
            print(self.a[0:10])
            
        elif self.visible_type=="Multinoulli":
            for m in range(self.M):
                if not self.if_fix_vis_bias:
                    self.a[m]=self.a[m] - self.learn_rate_a * self.grad_a[m]
                    
        elif self.visible_type=="Gaussian" or self.visible_type=="Gaussian2":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * self.grad_a[0]
                #self.a[0][self.a[0]<0]=0                
                self.a[1]=self.a[1] - self.learn_rate_a[1] * self.grad_a[1]
            else:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * (not_fix_a_log_ind*self.grad_a[0]) # update mean
                self.a[1]=self.a[1] - self.learn_rate_a[1] * (not_fix_a_log_ind*self.grad_a[1])
            tol=0.005
            self.a[1][self.a[1]>=-tol]=-tol
            self.a[1][self.a[1]<-50]=-50
            #self.a[1][self.a[1]<-numpy.pi]=-numpy.pi
            #self.a[1][self.a[1]<-20]=-20
            print("in update_param ...")
            print(self.a[0][0:10].transpose())
            print(self.a[1][0:10].transpose())
            
        elif self.visible_type=="Gaussian_Hinton":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * self.grad_a[0]
                #self.a[0][self.a[0]<0]=0                
                self.a[1]=self.a[1] - self.learn_rate_a[1] * self.grad_a[1]
            else:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * (not_fix_a_log_ind*self.grad_a[0]) # update mean
                self.a[1]=self.a[1] - self.learn_rate_a[1] * (not_fix_a_log_ind*self.grad_a[1]) # update precision
                self.a[1][self.a[1]<=0]=tol # precision>0
                
        elif self.visible_type=="Gamma":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] -self.learn_rate_a[0] * self.grad_a[0]
            else:
                self.a[0]=self.a[0] -self.learn_rate_a[0] * (not_fix_a_log_ind*self.grad_a[0])
                self.a[0][self.a[0]<1]= 1
                self.a[0][self.a[0]>tol_gamma_max]=tol_gamma_max
                self.a[1]=self.a[1] -self.learn_rate_a[1] * (not_fix_a_log_ind*self.grad_a[1])
                self.a[1][self.a[1]>=0]=-tol_gamma_min
                self.a[1][self.a[1]<-tol_gamma_max]=-tol_gamma_max
            #self.W=self.W - self.learn_rate_W * self.grad_W
            #self.W[self.W>0]=0
            print("in update_param ...")
            print(self.a[0][0:10])
            print(self.a[1][0:10])
            
        # update c
        self.c = self.c - self.learn_rate_c * self.grad_c
        print("c[0:10]")
        print(self.c[0:10].transpose())
        
        # update b
        if self.hidden_type=="Bernoulli" or self.hidden_type=="Binomial" or self.hidden_type=="Multinomial" or self.hidden_type=="Gaussian_FixPrecision1" or self.hidden_type=="Gaussian_FixPrecision2":
            for k in range(self.K):
                self.b[k]=self.b[k] - self.learn_rate_b * self.grad_b[k]
            print(self.b[0].transpose())
                
        elif self.hidden_type=="Gaussian":
            for k in range(self.K):
                self.b[k][0]= self.b[k][0] - self.learn_rate_b[0] * self.grad_b[k][0]
                self.b[k][1]= self.b[k][1] - self.learn_rate_b[1] * self.grad_b[k][1]
                self.b[k][1][self.b[k][1]>=0]=-tol
                
        elif self.hidden_type=="Gaussian_Hinton":
            for k in range(self.K):
                self.b[k][0]= self.b[k][0] - self.learn_rate_b[0] * self.grad_b[k][0]
                self.b[k][1]= self.b[k][1] - self.learn_rate_b[1] * self.grad_b[k][1]
                self.b[k][1][self.b[k][1]<=0]=tol    

        # update W
        if self.xz_interaction:
            KK=self.K+1
        else:
            KK=self.K
        for k in range(KK):
            if self.visible_type=="Multinoulli":
                for m in range(self.M):
                    self.W[k][m]=self.W[k][m] - self.learn_rate_W * self.grad_W[k][m]
            elif self.visible_type=="Gaussian2":
                self.W[k][0] = self.W[k][0] - self.learn_rate_W[0] * self.grad_W[k][0]
                self.W[k][1] = self.W[k][1] - self.learn_rate_W[1] * self.grad_W[k][1]
            else:
                self.W[k]=self.W[k] - self.learn_rate_W * self.grad_W[k]
        print(self.W[0])


    def sample_minibatch(self, batch_size=20):
        ind_batch=self.rng.choice(self.N,size=batch_size,replace=False)
        if self.visible_type=="Multinoulli":
            Xbatch=[None]*self.M
            for m in range(self.M):
                Xbatch[m]=self.X[m][:,ind_batch]
                if self.batch_size==1:
                    Xbatch[m].shape=(self.I[m],1)
        else:
            Xbatch=self.X[:,ind_batch]
            if self.batch_size==1:
                Xbatch.shape=(self.M,1)
        return Xbatch


    def change_learning_rate(self, current_learn_rate, change_rate, current_iter, change_every_many_iters):
        if current_iter!=0 and current_iter%change_every_many_iters==0:
            if numpy.isscalar(current_learn_rate):
                return current_learn_rate * change_rate
            else:
                new_learn_rate=[c*change_rate for c in current_learn_rate]
                #R=len(current_learn_rate)
                #new_learn_rate=[None]*R
                #for r in range(R):
                #    new_learn_rate[r]=current_learn_rate[r]*change_rate
                return new_learn_rate
        else:
            return current_learn_rate
            

    def reinit_a(self): 
        if self.visible_type=="Bernoulli":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            self.a=mean
        elif self.visible_type=="Gaussian" or self.visible_type=="Gaussian2":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/(var+0.0001)
            #precision[precision>100]=100
            #precision[precision>numpy.pi]=numpy.pi
            precision[precision>10]=10
            self.a[0]=mean*precision
            self.a[1]=-0.5*precision
        elif self.visible_type=="Gaussian_Hinton":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/(var+0.0001)
            #precision[precision>100]=100
            #precision[precision>numpy.pi]=numpy.pi
            precision[precision>10]=10
            self.a[0]=mean
            self.a[1]=precision
            #print self.a[0]
            #print self.a[1]
        elif self.visible_type=="Gaussian_FixPrecision1":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            self.a=mean
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/(var+0.0001)
            precision[precision>10]=10
            self.visible_type_fixed_param=precision
        elif self.visible_type=="Gaussian_FixPrecision2":
#            var=100*numpy.var(self.X, axis=1)
#            var.shape=(var.size,1)
#            precision=1/var
#            precision[precision>10]=10
#            self.visible_type_fixed_param=precision
#            mean=numpy.mean(self.X, axis=1)
#            mean.shape=(mean.size,1)
#            self.a=mean*precision
#            print(self.a)
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/(var+0.0001)
            #precision[precision>100]=100
            #precision[precision>numpy.pi]=numpy.pi
            precision[precision>10]=10
            self.visible_type_fixed_param=precision
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            self.a=mean*precision
            #print(self.a)
        elif self.visible_type=="Poisson":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)
            #var.shape=(var.size,1)
            self.a=numpy.log(mean)
            self.a[self.a<-2.3]=-2.3 # threshold log(0.1)
        elif self.visible_type=="NegativeBinomial":
            #max_X=numpy.max(self.X, axis=1)
            #max_X.shape=(max_X.size,1)
            #self.visible_type_fixed_param=max_X
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)
            #var.shape=(var.size,1)
            self.a=numpy.log(mean/(self.visible_type_fixed_param+mean))
            
        elif self.visible_type=="Multinoulli":
            for m in range(self.M):
                mean=numpy.mean(self.X[m], axis=1)
                mean.shape=(mean.size,1)
                #var=numpy.var(self.X[m], axis=1)
                #var.shape=(var.size,1)
                self.a[m]=numpy.log(mean/mean.sum())
        elif self.visible_type=="Multinomial":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)
            #var.shape=(var.size,1)
            #self.a=numpy.log(mean/mean.sum())
            self.a=mean/mean.sum()
        elif self.visible_type=="Gamma":
            mean=numpy.mean(self.X+1, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X+1, axis=1)
            var.shape=(var.size,1)
            self.a[0]=mean**2/var - 1
            self.a[1]=-mean/var
    

    def fix_vis_bias(self, a=None, fix_a_log_ind=None):
        """
        Fix the visible bias in training.
        For Gaussian, a is a 2-d tuple, a[0] for a1 and a[1] for a2.
        For Multinoulli, a is a M-d tuple or list.
        """
        self.if_fix_bis_bias=True
        self.fix_a_log_ind=fix_a_log_ind
        if self.fix_a_log_ind is None:
            self.fix_a_log_ind=numpy.array([True]*self.M)
        if a is not None:
            if self.visible_type in ["Bernoulli","Multinomial","Poisson","Gaussian_FixPrecision1","Gaussian_FixPrecision2"]:
                if isinstance(a,list):
                    a=a[0]
                self.a=a
            elif self.visible_type in ["Gaussian","Gaussian2"]:
                self.a=a
            print("I will fix a using the new a in this CRBM." )
        else:
            print("I will fix the existing a in this CRBM.")
            
            
    def get_param(self):
        """
        Get model parameters.
        """
        return self.a,self.b,self.c,self.W


    def set_param(self,a, b, c, W):
        """
        Set model parameters.
        """
        self.a=a
        self.b=b
        self.c=c
        self.W=W
        
        
    def make_dir_save(self,parent_dir_save, prefix, learn_rate_a, learn_rate_b, learn_rate_c, learn_rate_W, visible_type_fixed_param, hidden_type_fixed_param, maxiter, normalization_method="None"):
        
        if self.visible_type=="Gaussian" or self.visible_type=="Gamma": 
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + self.hidden_type + ":" + str(self.K) + "_learnrateabcW:" + str(learn_rate_a[0]) + "_" + str(learn_rate_a[1]) + "_" + str(learn_rate_b) + "_" + str(learn_rate_c) + "_" + str(learn_rate_W) + "_visfix:" + str(visible_type_fixed_param) + "_hidfix:" + str(hidden_type_fixed_param) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        else:
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + self.hidden_type + ":" + str(self.K) + "_learnrateabcW:" + str(learn_rate_a) + "_" + str(learn_rate_b) + "_" + str(learn_rate_c) + "_" + str(learn_rate_W) + "_visfix:" + str(visible_type_fixed_param) + "_hidfix:" + str(hidden_type_fixed_param) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        dir_save=parent_dir_save+foldername
        self.dir_save=dir_save
        try:
            os.makedirs(dir_save)
        except OSError:
            #self.dir_save=parent_dir_save
            pass
        print("The results will be saved in " + self.dir_save)
        return self.dir_save
        
    
    def train(self, X=None, X_validate=None, batch_size=20, NMF=10, increase_NMF_at=None, increased_NMF=[20], pcdk=20, NS=20, maxiter=100, learn_rate_a=0.1, learn_rate_b=0.1, learn_rate_c=0.1, learn_rate_W=0.1, change_rate=0.9, adjust_change_rate_at=None, adjust_coef=1.02, reg_lambda_a=0, reg_alpha_a=1, reg_lambda_b=0, reg_alpha_b=1, reg_lambda_W=0, reg_alpha_W=1,  change_every_many_iters=100, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=False, if_plot_error_free_energy=False, dir_save="./", prefix="CRBM", figwidth=5, figheight=3):
        """
        X: numpy 2d array of size M by N, each column is a sample. 
        If train_subset_size_for_compute_error and valid_subset_size_for_compute_error are Nones, use all available training and validation samples.
        """
        start_time=time.clock()
        print("training CRBM ...")
        if self.visible_type=="Multinoulli": # convert to binary
            self.X=[None]*self.M
            if X_validate is not None:
                self.X_validate=[None]*self.M
            else:
                self.X_validate=None
            for m in range(self.M):
                Z,_=cl.membership_vector_to_indicator_matrix(X[m,:],z_unique=range(self.I[m]))
                self.X[m]=Z.transpose()
                if X_validate is not None:
                    Z,_=cl.membership_vector_to_indicator_matrix(X_validate[m,:],z_unique=range(self.I[m]))
                    self.X_validate[m]=Z.transpose()
            self.N=self.X[0].shape[1]
            if X_validate is not None:
                self.N_validate=self.X_validate[0].shape[1] # number of validation samples
        else: # not multinoulli variables
            self.X= X
            self.N=self.X.shape[1] # number of training samples
            self.X_validate=X_validate
            if X_validate is not None:
                self.N_validate=self.X_validate.shape[1] # number of validation samples
            
        self.batch_size=batch_size
        if self.batch_size>self.N:
            self.batch_size=1

        # re-initiate the bias term of visible variables using the statistics of data
        if reinit_a_use_data_stat:
            self.reinit_a()

        # initialize Markov chains
        print("initializing Markov chains...")
        _,_,_,_,_,_,_=self.pcd_sampling(NS=NS, pcdk=init_chain_time*pcdk, rand_init=False, init=True, Zto1=False) # initialize pcd
        self.maxiter=maxiter
        self.learn_rate_a=learn_rate_a
        self.learn_rate_b=learn_rate_b
        self.learn_rate_c=learn_rate_c
        self.learn_rate_W=learn_rate_W
        # regularization coefficient
        self.reg_lambda_a=reg_lambda_a
        self.reg_alpha_a=reg_alpha_a
        self.reg_lambda_b=reg_lambda_b
        self.reg_alpha_b=reg_alpha_b
        self.reg_lambda_W=reg_lambda_W
        self.reg_alpha_W=reg_alpha_W

        self.rec_errors_train=[]
        self.rec_errors_valid=[]
        self.mfes_train=[]
        self.mfes_valid=[]
        
        for i in range(self.maxiter):
            Zto1=False
            #if i<120:
            #    Zto1=True
                
                
            # get mini-batch
            Xbatch=self.sample_minibatch(self.batch_size)
            
            # mean-field approximation here!
            if increase_NMF_at is not None:
                if i==increase_NMF_at[0]:
                    NMF=increased_NMF[0]
                    if len(increase_NMF_at)>1:
                        increase_NMF_at=increase_NMF_at[1:]
                        increased_NMF=increased_NMF[1:]
                        
            XbatchM,Hbatch,Zbatch=self.mean_field_approximate_inference(Xbatch, NMF=NMF, Zto1=Zto1)

            print("in the training of CRBM... Hbatch:")
            for k in range(self.K):
                if Zbatch[k,:].sum()<0.1:
                    print("inactive capsule: Hbatch[k].sum(axis=1)=")
                    print(Hbatch[k].sum(axis=1))
                    break
            for k in range(self.K):
                if Zbatch[k,:].sum()>5:
                    print("active capsule: Hbatch[k].sum(axis=1)=")
                    print(Hbatch[k].sum(axis=1))
                    break
            #print Hbatch[0]
                    
            print("Zbatch.sum(axis=1)=")
            print(Zbatch.sum(axis=1))
            #print Zbatch

            # pcd-k sampling
            XS,HS,ZS,_,_,_,_=self.pcd_sampling(pcdk, init=False, Zto1=Zto1)
            print("in the training of CRBM... HS:")
            for k in range(self.K):
                if ZS[k,:].sum()==0:
                    print("inactive capsule: HS[k].sum(axis=1)=")
                    print(HS[k].sum(axis=1))
                    break
            for k in range(self.K):
                if Zbatch[k,:].sum()>5:
                    print("active capsule: HS[k].sum(axis=1)=")
                    print(HS[k].sum(axis=1))
                    break
            #print HS[0]
                    
            print("ZS.sum(axis=1)=")
            print(ZS.sum(axis=1))            
            #print ZS
            
            # cd-k sampling
            #_,_,XS,HS,_=self.pcd_sampling(pcdk=pcdk,X0=Xbatch,persistent=False,init=True) # use probabilities insead of binaries
            #self.NS=self.batch_size # for CD-k, they must be equal.

            # compute gradient
            self.compute_gradient(Xbatch, Hbatch, Zbatch, XS, HS, ZS)
            # update parameters
            if adjust_change_rate_at is not None:
                if i==adjust_change_rate_at[0]:
                    change_rate=change_rate*adjust_coef # increast change_rate
                    change_rate=1.0 if change_rate>1.0 else change_rate # make sure not greater than 1
                    if len(adjust_change_rate_at)>1:
                        adjust_change_rate_at=adjust_change_rate_at[1:] # delete the first element
                    else:
                        adjust_change_rate_at=None
                
            self.learn_rate_a=self.change_learning_rate(current_learn_rate=self.learn_rate_a, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_b=self.change_learning_rate(current_learn_rate=self.learn_rate_b, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_c=self.change_learning_rate(current_learn_rate=self.learn_rate_c, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_W=self.change_learning_rate(current_learn_rate=self.learn_rate_W, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            
            self.update_param()
            
            
            # backup parameters
            self.backup_param(i, dif=500)
            
            # compute reconstruction error of the training samples
            if track_reconstruct_error:
                rec_error_train=self.compute_reconstruction_error(X0=Xbatch, X0RM=XbatchM)
                self.rec_errors_train.append(rec_error_train)
                
                if rec_error_train>self.rec_error_max or math.isnan(rec_error_train):
                    self.reset_param_use_backup(i, dif=250)
                    # reinitialize pcd
                    _,_,_,_,_,_,_=self.pcd_sampling(NS=NS, pcdk=init_chain_time*pcdk, rand_init=False, init=True, Zto1=False)
                    
            if track_free_energy:
                mfe_train,_=self.compute_free_energy(X=Xbatch, H=Hbatch)
                self.mfes_train.append(mfe_train)
                
            

            if self.X_validate is not None:
                if valid_subset_size_for_compute_error is not None:
                    valid_subset_ind=self.rng.choice(numpy.arange(self.N_validate,dtype=int),size=valid_subset_size_for_compute_error)
                    if self.visible_type=="Multinoulli":
                        X_validate_subset=[None]*self.M
                        for m in range(self.M):
                            X_validate_subset[m]=self.X_validate[m][:,valid_subset_ind]
                    else:
                        X_validate_subset=self.X_validate[:,valid_subset_ind]
                    if track_reconstruct_error:
                        print(X_validate_subset.shape)
                        rec_error_valid=self.compute_reconstruction_error(X0=X_validate_subset, X0RM=None)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X_validate_subset)
                        self.mfes_valid.append(mfe_validate)
                else:
                    if track_reconstruct_error:                    
                        rec_error_valid=self.compute_reconstruction_error(X0=self.X_validate, X0RM=None)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(self.X_validate)
                        self.mfes_valid.append(mfe_validate)
                # compute difference of free energy between training set and validation  set
                # the log-likelihood(train_set) - log-likelihood(validate_set) = F(validate_set) - F(train_set), the log-partition function, logZ is cancelled out
            print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W))
            if track_reconstruct_error:
                if self.X_validate is None:
                    print("train_rec_error: {0}".format(rec_error_train))
                else:
                    print("train_rec_error: {0}, valid_rec_error: {1}".format(rec_error_train, rec_error_valid))
            if track_free_energy:
                if self.X_validate is None:
                    print("free_energy_train:{0}".format(mfe_train))
                else:
                    free_energy_dif=mfe_train - mfe_validate
                    print("free_energy_train: {0}, free_energy_valid: {1}, free_energy_dif: {2}".format(mfe_train, mfe_validate, free_energy_dif))
                
        if if_plot_error_free_energy:
            self.plot_error_free_energy(dir_save, prefix, figwidth=5, figheight=3)
            
        print("The training of CRBM is finished!")
        end_time = time.clock()
        self.train_time=end_time-start_time
        return self.train_time
        print("It took {0} seconds.".format(self.train_time))

    
    def smooth(self,x, mean_over=5):
        """
        Smooth a vector of numbers.
        x: list of vector.
        mean_over: scalar, the range of taking mean.
        """
        num=len(x)
        x_smooth=numpy.zeros((num,))
        for n in range(num):
            start=n-mean_over+1
            if start<0:
                start=0
            x_smooth[n]=numpy.mean(x[start:n+1])
        return x_smooth
        
        
    def plot_error_free_energy(self, dir_save="./", prefix="CRBM", mean_over=5, figwidth=5, figheight=3):
        
        if len(self.rec_errors_train)>0:
            num_iters=len(self.rec_errors_train)
            if mean_over>0:
                self.rec_errors_train=self.smooth(self.rec_errors_train, mean_over=mean_over)
            else:
                self.rec_errors_train=numpy.array(self.rec_errors_train)

        if len(self.rec_errors_valid)>0:
            num_iters=len(self.rec_errors_valid)
            if mean_over>0:
                self.rec_errors_valid=self.smooth(self.rec_errors_valid, mean_over=mean_over)
            else:
                self.rec_errors_valid=numpy.array(self.rec_errors_valid)

        if len(self.mfes_train)>0:
            num_iters=len(self.mfes_train)
            if mean_over>0:
                self.mfes_train=self.smooth(self.mfes_train, mean_over=mean_over)
            else:
                self.mfes_train=numpy.array(self.mfes_train)

        if len(self.mfes_valid)>0:
            num_iters=len(self.mfes_valid)
            if mean_over>0:
                self.mfes_valid=self.smooth(self.mfes_valid, mean_over=mean_over)
            else:
                self.mfes_valid=numpy.array(self.mfes_valid)

        iters=numpy.array(range(num_iters),dtype=int)
        
        # ignore the first five results as they are not stable
        iters=iters[5:]
        if len(self.rec_errors_train)>0:
            self.rec_errors_train=self.rec_errors_train[5:]
        if len(self.rec_errors_valid)>0:
            self.rec_errors_valid=self.rec_errors_valid[5:]
        if len(self.mfes_train)>0:
            self.mfes_train=self.mfes_train[5:]
        if len(self.mfes_valid)>0:
            self.mfes_valid=self.mfes_valid[5:]
        
        #plt.ion()
        fig=plt.figure(num=1,figsize=(figwidth,figheight))
        ax1=fig.add_subplot(1,1,1)
        if len(self.rec_errors_train)>0:
            ax1.plot(iters,self.rec_errors_train,linestyle="-",color="red",linewidth=0.5, label="RCE:Train")
        if len(self.rec_errors_valid)>0:
            ax1.plot(iters,self.rec_errors_valid,linestyle=":",color="darkgoldenrod",linewidth=0.5, label="RCE:Test")
        ax1.set_xlabel("Iteration",fontsize=8)
        ax1.set_ylabel("Reconstruction Error (RCE)", color="red",fontsize=8)
        for tl in ax1.get_yticklabels():
            tl.set_color("r")
        if self.rec_errors_train.max()>1:
            ax1.set_ylim(0.0,1.0)
        plt.setp(ax1.get_yticklabels(), fontsize=8)
        plt.setp(ax1.get_xticklabels(), fontsize=8)
        ax1.legend(loc="upper right",fontsize=8)
        prefix=prefix+"_error"
            
        #ax.legend(loc="lower left",fontsize=8)
        if len(self.mfes_train)>0 or len(self.mfes_valid)>0:
            ax2=ax1.twinx()
            if len(self.mfes_train)>0:
                ax2.plot(iters,self.mfes_train,linestyle="-", color="blue", linewidth=0.5, label="FE:Train")
            if len(self.mfes_valid)>0:
                ax2.plot(iters,self.mfes_valid,linestyle=":",color="blueviolet",linewidth=0.5, label="FE:Test")
            ax2.set_xlabel("Iteration",fontsize=8)
            ax2.set_ylabel("Free Energy (FE)",color="blue",fontsize=8)
            for tl in ax2.get_yticklabels():
                tl.set_color("b")
            plt.setp(ax2.get_yticklabels(), fontsize=8)
            plt.setp(ax2.get_xticklabels(), fontsize=8)
            ax1.legend(loc="upper left",fontsize=8)
            ax2.legend(loc="upper right",fontsize=8)
            prefix=prefix+"_free_energy"
        
        filename=dir_save+prefix+".pdf"
        plt.tight_layout()
        fig.savefig(filename,bbox_inches='tight',format="pdf",dpi=600)
        plt.close(fig)
        #plt.close("all")

    
    def save_sampling(self, X, dir_save="./", prefix="CRBM"):
        """
        Save the sampling results.
        """
        filename=dir_save + prefix + ".txt"
        numpy.savetxt(filename, X, fmt="%.2f", delimiter="\t")
            
    
#    def generate_samples(self, NS=100, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, dir_save="./", prefix="CDBM"):
#        
#        if reinit or NS!=self.NS:
#            self.pcd_sampling(pcdk=pcdk, NS=NS, X0=None, persistent=True, rand_init=rand_init, init=True)
#            
#        for s in range(sampling_time):
#            chainX,chainH,chainZ,chainXM,chainHM,chainZM,chain_length=self.pcd_sampling(pcdk=pcdk, init=False)
#            # plot sampled data
#            sample_set_x_3way=numpy.reshape(chainXM,newshape=(28,28,NS))
#            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)
            

    def generate_samples(self, NS=100, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, row=28, col=28, dir_save="./", prefix="CRBM"):
        """
        Use Gibbs sampling.
        """
        
        if reinit or NS!=self.NS:
            self.pcd_sampling(pcdk=pcdk, NS=NS, X0=None, persistent=True, rand_init=rand_init, init=True)
            
        for s in range(sampling_time):
            chainX,chainH,chainZ,chainXM,chainHM,chainZM,chain_length=self.pcd_sampling(pcdk=pcdk, init=False)
            
            # plot sampled data
            sample_set_x_3way=numpy.reshape(chainXM,newshape=(row,col,NS))
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)
            # plot ZM
            ZM_3way=self.make_Z_matrix(chainZM)
            self.ZM_3way=ZM_3way
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_ZM.pdf", data=ZM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
            # save the Z code
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_ZM.txt", chainZM.transpose(),fmt="%.4f",delimiter="\t")
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_Z.txt", chainZ.transpose(),fmt="%s",delimiter="\t")
            
            # sorted results
            ind=numpy.argsort(chainZM.sum(axis=1))
            ind=ind[::-1]
            chainZM_sorted=chainZM[ind,:]
            chainZ_sorted=chainZ[ind,:]
            ZM_3way_sorted=self.make_Z_matrix(chainZM_sorted)
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_ZM_sorted.pdf", data=ZM_3way_sorted, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
            # save the Z code
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_ZM_sorted.txt", chainZM_sorted.transpose(),fmt="%.4f",delimiter="\t")
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_Z_sorted.txt", chainZ_sorted.transpose(),fmt="%s",delimiter="\t")
        return chainZM,chainZM_sorted,ind
        

    def generate_samples_given_z(self, Z0=None, NS=100, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, row=28, col=28, num_col=None, figwidth=6, figheight=6, dir_save="./", prefix="CRBM"):
        """
        Use Gibbs sampling.
        """
        
        if reinit or NS!=self.NS:
            self.pcd_sampling_fix_z(pcdk=10*pcdk, NS=NS, X0=None, Z0=Z0, persistent=True, rand_init=rand_init, init=True)
            
        for s in range(sampling_time):
            chainX,chainH,chainXM,chainHM,chain_length=self.pcd_sampling_fix_z(pcdk=pcdk, init=False)
            
            # plot sampled data
            sample_set_x_3way=numpy.reshape(chainXM,newshape=(row,col,NS))
            if num_col is None:
                num_col=int(numpy.ceil(numpy.sqrt(NS)))
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_given_z_randinit_"+str(rand_init)+"_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=figwidth, figheight=figheight, colormap="gray", aspect="equal", num_col=num_col, wspace=0.01, hspace=0.001)
        

    def make_Z_matrix(self, Z):
        ZT=numpy.reshape(Z,newshape=(1,Z.shape[0],Z.shape[1]))
        return ZT
        

    def infer_z_given_x(self, train_set_x_sub=None, NMF=100, num_col=10, figwidth=6, figheight=6, plotimg=True, dir_save="./", prefix="CRBM"):
        """
        Use mean-field approximation.
        """
        
        NS=train_set_x_sub.shape[1]
        _,_,ZM=self.mean_field_approximate_inference(train_set_x_sub, NMF=NMF, Zto1=False)
        if plotimg:
            ZM_3way=self.make_Z_matrix(ZM)
            self.ZM_3way=ZM_3way
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_ZM.pdf", data=ZM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
        # save the Z code
        numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_ZM.txt", ZM.transpose(),fmt="%.4f",delimiter="\t")
        
        # sorted results
        ind=numpy.argsort(ZM.sum(axis=1))
        ind=ind[::-1]
        ZM_sorted=ZM[ind,:]
        if plotimg:
            ZM_3way_sorted=self.make_Z_matrix(ZM_sorted)
            if num_col is None:
                num_col=int(numpy.ceil(numpy.sqrt(NS)))
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_ZM_sorted.pdf", data=ZM_3way_sorted, figwidth=figwidth, figheight=figheight, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=num_col, wspace=0.1, hspace=0.1)
        numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_ZM_sorted.txt", ZM_sorted.transpose(),fmt="%.4f",delimiter="\t")
        return ZM,ZM_sorted,ind


    def per_capsule_pattern_given_x(self, train_set_x_sub=None, NMF=100, threshold_z=0, row=28, col=28, figwidth=6, figheight=6, dir_save="./", prefix="CRBM"):
        NS=train_set_x_sub.shape[1]
        # approximate H, Z
        _,HM,ZM=self.mean_field_approximate_inference(train_set_x_sub, NMF=NMF, Zto1=False)
        # save ZM
        ZM_3way=self.make_Z_matrix(ZM)
        self.ZM_3way=ZM_3way
        cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_x_generate_ZM.pdf", data=ZM_3way, figwidth=figwidth, figheight=figwidth/NS, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=NS, wspace=0.1, hspace=0.1)
        # save the Z code
        numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_x_generate_ZM.txt", ZM.transpose(),fmt="%.4f",delimiter="\t")
        
        for n in range(NS):
            #ZMncopy=copy.deepcopy(ZM[:,n])
            #ZMncopy[ZMncopy<=threshold_z]=0
            #ZMn=numpy.hstack( (ZM[:,[n]], numpy.diag(ZMncopy)) ) # the first column is the Z code for the n-the sample, and the following are dismentaled
            ZMn=numpy.hstack( (ZM[:,[n]], numpy.diag(ZM[:,n])) )
            #ZMall.extend(ZMn)
            
            if isinstance(HM[0],numpy.ndarray): # Bernoulli etc.
                HMn=[]
                for k in range(self.K):
                    HMn.append( numpy.tile(HM[k][:,[n]],self.K+1) )
            else: # Gaussian etc.
                HMn=[None]*self.K
                for k in range(self.K):
                    HMn[k]=[None,None]
                    HMn[k][0]=numpy.tile(HM[k][0][:,[n]],self.K+1)
                    HMn[k][1]=numpy.tile(HM[k][1][:,[n]],self.K+1)
                
            #HMall.extend(HMn)
        
            # generate per-capsule patterns
            _,XMn=self.sample_x_given_hz(HMn, ZMn, a_hat=None, value_or_meanfield="mean")
            
            # change prior background to zeros
            ind=ZMn.sum(axis=0)<=threshold_z
            XMn[:,ind]=0
            
            XMn=numpy.hstack((train_set_x_sub[:,[n]],XMn)) # put actual sample to the first
            if n==0:
                XMall=XMn
                ZMall=ZMn
            else:
                XMall=numpy.hstack((XMall,XMn))
                ZMall=numpy.hstack((ZMall,ZMn))
        # plot
        sample_set_x_3way=numpy.reshape(XMall,newshape=(row,col,NS*(self.K+2)))
        cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type + "_per_capsule_pattern.pdf", data=sample_set_x_3way, figwidth=figwidth, figheight=NS/(self.K+2)*figwidth, colormap="gray", aspect="equal", num_col=self.K+2, wspace=0.01, hspace=0.001) 
        
        return XMall,ZMall
        

    def reorder_capsules(self, ind):
        self.c=self.c[ind]
        b_old=copy.deepcopy(self.b)
        self.b=[ b_old[ind[k]] for k in range(self.K)]
        W_old=copy.deepcopy(self.W)
        self.W=[ W_old[ind[k]] for k in range(self.K)]
        self.W.append( W_old[self.K] )
        self.chainZ=self.chainZ[ind,:]
        self.chainZM=self.chainZM[ind,:]
        chainH_old=copy.deepcopy(self.chainH)
        self.chainH=[chainH_old[ind[k]] for k in range(self.K)]
        chainHM_old=copy.deepcopy(self.chainHM)
        self.chainHM=[chainHM_old[ind[k]] for k in range(self.K)]


#ind=numpy.argsort(ZM.sum(axis=1))
#ind=ind[::-1]
#ZM_sorted=ZM[ind,:]
#ZM_3way=model_crbm.make_Z_matrix(ZM_sorted)
#cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+visible_type +"_given_sample_generate_ZM_sorted.pdf", data=ZM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
#numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+visible_type +"_given_sample_generate_ZM_sorted.txt", ZM_sorted.transpose(),fmt="%.4f",delimiter="\t")            