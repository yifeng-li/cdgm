
from __future__ import division
import numpy
import restricted_boltzmann_machine
import capsule_rbm3
#import capsule_rbm23
import classification as cl
import copy
import os
import time
import math

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt

class capsule_dgm:
    def __init__(self, features=None, M=None, K=None, J=16, visible_type="Bernoulli", visible_type_fixed_param=1, hidden_type="Bernoulli", hidden_type_fixed_param=1, if_fix_vis_bias=False, a=None, fix_a_log_ind=None, tol_poisson_max=8, rec_error_max=0.5, tie_gen_recog_param=False, xz_interaction=True, rng=numpy.random.RandomState(100)):
        """
        M: scalar integer, the dimension of input, i.e. the number of input features.
        K: list of integers, the numbers of hidden units in each hidden layer. 
        J: scalar, the number of units per capsule.
        hidden_type can be Bernoulli, Poisson, Binomial, NegativeBinomial,Multinomial, or Gaussian_FixPrecision1 or Gaussian_FixPrecision2.
        """
        self.features=features
        self.M=M
        self.K=K
        self.NK=len(K) # number of hidden layers
        self.L=self.NK # equivalent to self.NK
        self.J=J # number of units in a capsule
        if numpy.isscalar(self.J):
            self.J=[self.J]*self.L
            for l in range(self.L):
                self.J[l]=[self.J[l]]*self.K[l]
        self.visible_type=visible_type
        self.visible_type_fixed_param=visible_type_fixed_param
        if numpy.isscalar(hidden_type):
            hidden_type=[hidden_type]*self.L
        self.hidden_type=hidden_type
        if numpy.isscalar(hidden_type_fixed_param):
            hidden_type_fixed_param=[hidden_type_fixed_param]*self.L
        self.hidden_type_fixed_param=hidden_type_fixed_param
        self.a=None # generative
        self.b=[None]*self.L # generative
        self.b2=[None]*self.L # generative
        self.c=[None]*self.L # generative
        self.d=[None]*self.L # generative
        self.W=[None]*self.L # generative, self.W[l] is the weights for the l-th layer
        self.W2=[None]*self.L # generative, if interactions affect variances or the second parameter
        self.br=[None]*self.L # recognition
        self.b2r=[None]*self.L # recognition
        self.cr=[None]*self.L # recognition
        self.dr=[None]*self.L # recognition
        self.Wr=[None]*self.L # recognition
        self.W2r=[None]*self.L # recognition
        self.crbms=[None]*self.L #  self.crbms[l] is the capsule RBM for the l-th layer
        self.rng=rng
        self.tie_gen_recog_param=tie_gen_recog_param
        self.xz_interaction=xz_interaction
        self.tol_poisson_max=tol_poisson_max
        self.rec_error_max=rec_error_max
        
        # whether fix a if this HM is a joint component in multimodal HM
        self.if_fix_vis_bias=if_fix_vis_bias
        self.fix_a_log_ind=fix_a_log_ind

        self.W[0]=[None]*self.K[0]
        self.W2[0]=[None]*self.K[0]
        self.Wr[0]=[None]*self.K[0]
        self.W2r[0]=[None]*self.K[0]
        # initialize visible types
        if self.visible_type=="Bernoulli":
            self.a=numpy.zeros(shape=(self.M,1))
            for k in range(self.K[0]):
                #self.W[0][k]=self.rng.normal(loc=0, scale=0.0001, size=(self.M,self.J[0][k]))
                #self.Wr[0][k]=self.rng.normal(loc=0, scale=0.0001, size=(self.M,self.J[0][k]))
                self.W[0][k]=numpy.zeros(shape=(self.M,self.J[0][k]),dtype=float)
                self.Wr[0][k]=numpy.zeros(shape=(self.M,self.J[0][k]),dtype=float)
                #self.W[k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
            # for the interaction between X and Z
            #self.W[0][self.K]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K))
        elif self.visible_type=="Gaussian":
            self.a=[None]*2
            #self.a[0]=numpy.abs(self.rng.normal(loc=0, scale=1, size=(self.M,1))) # M X 1  a1=mu*lambda  # coefficient for x
            self.a[0]=self.rng.normal(loc=0, scale=1, size=(self.M,1))
            self.a[1]=-0.5*numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=-lambda/2<0 coefficient for x^2
            for k in range(self.K[0]):
                #self.W[0][k]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.J[0][k])) # M by K, initialize weight matrix
                #self.Wr[0][k]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.J[0][k]))
                self.W[0][k]=numpy.zeros(shape=(self.M,self.J[0][k]),dtype=float)
                self.Wr[0][k]=numpy.zeros(shape=(self.M,self.J[0][k]),dtype=float)
            #self.W[0][K]=numpy.zeros(shape=(self.M,self.K),dtype=float)
        elif self.visible_type=="Gaussian2":
            self.a=[None]*2
            #self.a[0]=numpy.abs(self.rng.normal(loc=0, scale=1, size=(self.M,1))) # M X 1  a1=mu*lambda  # coefficient for x
            self.a[0]=self.rng.normal(loc=0, scale=1, size=(self.M,1))
            self.a[1]=-5*numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=-lambda/2<0 coefficient for x^2
            for k in range(self.K[0]):
                self.W[0][k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
                self.W2[0][k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
                self.Wr[0][k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
                self.W2r[0][k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
        elif self.visible_type=="Gaussian_FixPrecision1": 
            #self.a=numpy.ones(shape=(self.M,1)) # a=mu, statistics is lambda*x
            self.a=self.rng.random_sample(size=(self.M,1)) # a=mu, statistics is lambda*x
            for k in range(self.K[0]):
                #self.W[k]=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.J[k]))
                self.W[0][k]=numpy.zeros(shape=(self.M,self.J[k]),dtype=float)
            # self.visible_type_fixed_param is precision lambda
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        
        # hidden bais b
        for l in range(self.L):
            self.b[l]=[None]*self.K[l]
            self.b2[l]=[None]*self.K[l]
            self.br[l]=[None]*self.K[l]
            self.b2r[l]=[None]*self.K[l]
            if self.hidden_type[l]=="Bernoulli":
                for k in range(self.K[l]):
                    self.b[l][k]=self.rng.normal(loc=0, scale=1, size=(self.J[l][k],1))#numpy.zeros(shape=(self.J[l][k],1))
                    self.br[l][k]=self.rng.normal(loc=0, scale=1, size=(self.J[l][k],1))#numpy.zeros(shape=(self.J[l][k],1))
            elif self.hidden_type[l] in ["Gaussian","Gaussian2"]:
                for k in range(self.K[l]):
                    self.b[l][k]=self.rng.normal(loc=0, scale=1, size=(self.J[l][k],1))
                    self.b2[l][k]=-0.5*numpy.ones(shape=(self.J[l][k],1),dtype=float)
                    self.br[l][k]=self.rng.normal(loc=0, scale=1, size=(self.J[l][k],1))
                    self.b2r[l][k]=-0.5*numpy.ones(shape=(self.J[l][k],1),dtype=float)
        
        if self.tie_gen_recog_param:
            self.br=self.b
            self.b2r=self.b2
        
        # initialize W[1], ..., W[L-1]
        for l in range(1,self.L):
            self.W[l]=[[None]*self.K[l]]*self.K[l-1]
            self.W2[l]=[[None]*self.K[l]]*self.K[l-1]
            self.Wr[l]=[[None]*self.K[l]]*self.K[l-1]
            self.W2r[l]=[[None]*self.K[l]]*self.K[l-1]
            if self.hidden_type[l] in ["Bernoulli","Gaussian"]:
                for k1 in range(self.K[l-1]):
                    for k2 in range(self.K[l]):
                        #self.W[l][k1][k2]=self.rng.normal(loc=0, scale=0.01, size=(self.J[l-1][k1],self.J[l][k2])) 
                        #self.Wr[l][k1][k2]=self.rng.normal(loc=0, scale=0.01, size=(self.J[l-1][k1],self.J[l][k2]))
                        self.W[l][k1][k2]=numpy.zeros(shape=(self.J[l-1][k1],self.J[l][k2]),dtype=float)
                        self.Wr[l][k1][k2]=numpy.zeros(shape=(self.J[l-1][k1],self.J[l][k2]),dtype=float)
            elif self.hidden_type[l] in ["Gaussian2"]:
                for k1 in range(self.K[l-1]):
                    for k2 in range(self.K[l]):
                        self.W[l][k1][k2]=numpy.zeros(shape=(self.J[l-1][k1],self.J[l][k2]),dtype=float)
                        self.W2[l][k1][k2]=numpy.zeros(shape=(self.J[l-1][k1],self.J[l][k2]),dtype=float)
                        self.Wr[l][k1][k2]=numpy.zeros(shape=(self.J[l-1][k1],self.J[l][k2]),dtype=float)
                        self.W2r[l][k1][k2]=numpy.zeros(shape=(self.J[l-1][k1],self.J[l][k2]),dtype=float)
        
        if self.tie_gen_recog_param:
            self.Wr=self.W
            self.W2r=self.W2
            
        # z bias c, follows Bernoulli distribution, z[L] is Multinoulli
        for l in range(self.L):
            #self.c[l]=self.rng.normal(loc=0, scale=1, size=(self.K[l],1))#numpy.zeros(shape=(self.K[l],1),dtype=float)
            #self.cr[l]=self.rng.normal(loc=0, scale=1, size=(self.K[l],1))#numpy.zeros(shape=(self.K[l],1),dtype=float)
            self.c[l]=numpy.zeros(shape=(self.K[l],1),dtype=float)
            self.cr[l]=numpy.zeros(shape=(self.K[l],1),dtype=float)
            
        if self.tie_gen_recog_param:
            self.cr=self.c
            
        # d, follows Multinoulli distribution
        #for l in range(self.L-1):
        #    self.d[l]=[None]*self.K[l]
        #    self.dr[l]=[None]*self.K[l]
        #    for k in range(self.K[l]):
        #        self.d[l][k]=numpy.zeros(shape=(self.K[l+1],1))
        #        self.dr[l][k]=numpy.zeros(shape=(self.K[l+1],1))
                
        for l in range(self.L-1):
            #self.d[l]= self.rng.normal(loc=0, scale=0.1, size=(self.K[l],self.K[l+1])) #numpy.zeros((self.K[l],self.K[l+1]))
            #self.dr[l]=self.rng.normal(loc=0, scale=0.1, size=(self.K[l],self.K[l+1])) #numpy.zeros((self.K[l],self.K[l+1]))
            self.d[l]= numpy.zeros((self.K[l],self.K[l+1]))
            self.dr[l]= numpy.zeros((self.K[l],self.K[l+1]))
            
        if self.tie_gen_recog_param:
            self.dr=self.d
        
        self.if_fix_vis_bias=if_fix_vis_bias
        self.fix_a_log_ind=fix_a_log_ind
        if self.if_fix_vis_bias:
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
            if a is not None:
                self.a=a
                print("I will fix a using the new a in this CHM.")
            else:
                print("I will fix the existing a in this CHM.")
            
        if if_fix_vis_bias:
            self.fix_vis_bias(a,fix_a_log_ind)
            
        self.backup_param(0)
            

    def backup_param(self, i, dif=240):
        if i==0:
            self.a_backup1=copy.deepcopy(self.a)
            self.b_backup1=copy.deepcopy(self.b)
            self.b2_backup1=copy.deepcopy(self.b2)
            self.c_backup1=copy.deepcopy(self.c)
            self.d_backup1=copy.deepcopy(self.d)
            self.W_backup1=copy.deepcopy(self.W)
            self.W2_backup1=copy.deepcopy(self.W2)
            self.br_backup1=copy.deepcopy(self.br)
            self.b2r_backup1=copy.deepcopy(self.b2r)
            self.cr_backup1=copy.deepcopy(self.cr)
            self.dr_backup1=copy.deepcopy(self.dr)
            self.Wr_backup1=copy.deepcopy(self.Wr)
            self.W2r_backup1=copy.deepcopy(self.W2r)
            
            self.a_backup2=copy.deepcopy(self.a)
            self.b_backup2=copy.deepcopy(self.b)
            self.b2_backup2=copy.deepcopy(self.b2)
            self.c_backup2=copy.deepcopy(self.c)
            self.d_backup2=copy.deepcopy(self.d)
            self.W_backup2=copy.deepcopy(self.W)
            self.W2_backup2=copy.deepcopy(self.W2)
            self.br_backup2=copy.deepcopy(self.br)
            self.b2r_backup2=copy.deepcopy(self.b2r)
            self.cr_backup2=copy.deepcopy(self.cr)
            self.dr_backup2=copy.deepcopy(self.dr)
            self.Wr_backup2=copy.deepcopy(self.Wr)
            self.W2r_backup2=copy.deepcopy(self.W2r)
            self.backup_iter=i
        elif i-self.backup_iter==dif:
            self.a_backup1=copy.deepcopy(self.a_backup2)
            self.b_backup1=copy.deepcopy(self.b_backup2)
            self.b2_backup1=copy.deepcopy(self.b2_backup2)
            self.c_backup1=copy.deepcopy(self.c_backup2)
            self.d_backup1=copy.deepcopy(self.d_backup2)
            self.W_backup1=copy.deepcopy(self.W_backup2)
            self.W2_backup1=copy.deepcopy(self.W2_backup2)
            self.br_backup1=copy.deepcopy(self.br_backup2)
            self.b2r_backup1=copy.deepcopy(self.b2r_backup2)
            self.cr_backup1=copy.deepcopy(self.cr_backup2)
            self.dr_backup1=copy.deepcopy(self.dr_backup2)
            self.Wr_backup1=copy.deepcopy(self.Wr_backup2)
            self.W2r_backup1=copy.deepcopy(self.W2r_backup2)
            
            self.a_backup2=copy.deepcopy(self.a)
            self.b_backup2=copy.deepcopy(self.b)
            self.b2_backup2=copy.deepcopy(self.b2)
            self.c_backup2=copy.deepcopy(self.c)
            self.d_backup2=copy.deepcopy(self.d)
            self.W_backup2=copy.deepcopy(self.W)
            self.W2_backup2=copy.deepcopy(self.W2)
            self.br_backup2=copy.deepcopy(self.br)
            self.b2r_backup2=copy.deepcopy(self.b2r)
            self.cr_backup2=copy.deepcopy(self.cr)
            self.dr_backup2=copy.deepcopy(self.dr)
            self.Wr_backup2=copy.deepcopy(self.Wr)
            self.W2r_backup2=copy.deepcopy(self.W2r)
            self.backup_iter=i
        
        
    def reset_param_use_backup(self, i, dif=120):
        if i-self.backup_iter >= dif:
            self.a=copy.deepcopy(self.a_backup2)
            self.b=copy.deepcopy(self.b_backup2)
            self.b2=copy.deepcopy(self.b2_backup2)
            self.c=copy.deepcopy(self.c_backup2)
            self.d=copy.deepcopy(self.d_backup2)
            self.W=copy.deepcopy(self.W_backup2)
            self.W2=copy.deepcopy(self.W2_backup2)
            self.br=copy.deepcopy(self.br_backup2)
            self.b2r=copy.deepcopy(self.b2r_backup2)
            self.cr=copy.deepcopy(self.cr_backup2)
            self.dr=copy.deepcopy(self.dr_backup2)
            self.Wr=copy.deepcopy(self.Wr_backup2)
            self.W2r=copy.deepcopy(self.W2r_backup2)
            self.backup_iter=i
        else:
            self.a=copy.deepcopy(self.a_backup1)
            self.b=copy.deepcopy(self.b_backup1)
            self.b2=copy.deepcopy(self.b2_backup1)
            self.c=copy.deepcopy(self.c_backup1)
            self.d=copy.deepcopy(self.d_backup1)
            self.W=copy.deepcopy(self.W_backup1)
            self.W2=copy.deepcopy(self.W2_backup1)
            self.br=copy.deepcopy(self.br_backup1)
            self.b2r=copy.deepcopy(self.b2r_backup1)
            self.cr=copy.deepcopy(self.cr_backup1)
            self.dr=copy.deepcopy(self.dr_backup1)
            self.Wr=copy.deepcopy(self.Wr_backup1)
            self.W2r=copy.deepcopy(self.W2r_backup1)
            self.backup_iter=i

   
    def transpose_Wl(self, Wl, l, K1, K2, xz_interaction):
        """
        Transpose W[l] to get Wr[l].
        """
        if l==0:
            Wrl=[copy.deepcopy(w.transpose()) for w in Wl]
        else:
            Wrl=copy.deepcopy(Wl)
            for k1 in range(K1):
                for k2 in range(K2):
                    Wrl[k1][k2] = Wrl[k1][k2].T
            if xz_interaction:
                Wrl[K1]=Wrl[K1].T
        return Wrl
    
    
    def merge_bl(self, bl, dist):
        if dist in ["Bernoulli", "Poisson"]:
            blstack=numpy.vstack(bl)
            blstack=blstack
        elif dist in ["Gaussian"]:
            blstack=[None,None]
            K=len(bl)
            bl0=[bl[k][0] for k in range(K)]
            bl1=[bl[k][1] for k in range(K)]
            blstack[0]=numpy.vstack(bl0)
            blstack[1]=numpy.vstack(bl1)
        return blstack


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
            
    
    def fix_vis_bias(self,a=None,fix_a_log_ind=None):
        """
        Fix the visible bias. Do not update them in learning.
        a: a numpy array of shape M by 1.
        fix_a_log_ind: a bool numpy vector of length M, fixed_log_ind[m]==True means fix self.a[m]
        """
        if a is not None:
            self.a=a # reset a
            if len(self.crbms)>0:
                self.crbms[0].a=a # reset the first crbm's visiable bias
            self.if_fix_vis_bias=True
            self.fix_a_log_ind=fix_a_log_ind
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
        else: # do not reset a
            self.if_fix_vis_bias=True
            self.fix_a_log_ind=fix_a_log_ind
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
        

    def pretrain(self, X=None, X_validate=None, batch_size=20, NMF=20, increase_NMF_at=None, increased_NMF=[20], pcdk=20, NS=20, maxiter=100, learn_rate_a=0.01, learn_rate_b=0.01, learn_rate_c=0.01, learn_rate_W=0.01, change_rate=0.8, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=100, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=False, reinit_a_use_data_stat=False, if_plot_error_free_energy=False, dir_save="./", prefix="Pretrain_using_CRBM", figwidth=5, figheight=3):
        """
        Pretraining CHM using CRBMs.
        Different layers can have different learning rates.
        """
        only_pretrain_first_layer=False
        
        start_time=time.clock()
        # different layers can have different learning rates         
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*self.L
        if numpy.isscalar(learn_rate_c):
            learn_rate_c=[learn_rate_c]*self.L
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*self.L        
            
        self.X=X
        self.X_validate = X_validate
        crbm_X=self.X
        crbm_X_validate=self.X_validate
        self.H_pretrain=[None]*self.L
        self.Z_pretrain=[None]*self.L 
        print("Start pretraining CDGM...")
        for l in range(self.L):
            if l==0:
                self.crbms[l]=capsule_rbm3.capsule_rbm3(features=self.features, M=self.M, K=self.K[0], J=self.J[0], visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type[0], hidden_type_fixed_param=self.hidden_type_fixed_param[0], tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=self.if_fix_vis_bias, a=self.a, fix_a_log_ind=self.fix_a_log_ind, xz_interaction=self.xz_interaction, rec_error_max=self.rec_error_max, tol_poisson_max=self.tol_poisson_max, rng=self.rng)
                
                self.crbms[l].train(X=crbm_X, X_validate=crbm_X_validate, batch_size=batch_size, NMF=NMF, increase_NMF_at=increase_NMF_at, increased_NMF=increased_NMF, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=learn_rate_a, learn_rate_b=learn_rate_b[l], learn_rate_c=learn_rate_c[l], learn_rate_W=learn_rate_W[l], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, reg_lambda_a=0, reg_alpha_a=1, reg_lambda_b=0, reg_alpha_b=1, reg_lambda_W=0, reg_alpha_W=1, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, reinit_a_use_data_stat=reinit_a_use_data_stat, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_"+str(l), figwidth=figwidth, figheight=figheight)
                a,b,c,W=self.crbms[l].get_param()
                self.a=a
                bl=self.merge_bl(b, self.hidden_type[l])
                b12l=[None,None]
                if self.hidden_type[l] in ["Gaussian","Gaussian2"]:
                    b12l[0]=bl[0]
                    b12l[1]=bl[1]
                    self.b[l]=[b[k][0] for k in range(self.K[l])]
                    self.b2[l]=[b[k][1] for k in range(self.K[l])]
                else:
                    b12l[0]=bl
                    self.b[l]=b
                self.c[l]=c
                del W[self.K[l]] # delte last item
                self.W[l]=W
                self.br[l]=copy.deepcopy(self.b[l])
                self.b2r[l]=copy.deepcopy(self.b2[l])
                self.cr[l]=copy.deepcopy(self.c[l])
                self.Wr[l]=copy.deepcopy(self.W[l])
                if l<self.L-1 and not only_pretrain_first_layer:
                    Xl,Hl,Zl=self.crbms[l].mean_field_approximate_inference(crbm_X, NMF=NMF)
                    Xl_validate,Hl_validate,Zl_validate=self.crbms[l].mean_field_approximate_inference(crbm_X_validate, NMF=NMF)
                    #crbm_X=Hl
                    #crbm_V=Zl
                    crbm_X=self.merge_ZlHl(Zl, Hl)
                    crbm_X_validate=self.merge_ZlHl(Zl_validate, Hl_validate)
                    self.H_pretrain[l]=Hl
                    self.Z_pretrain[l]=Zl
            else:
                if not only_pretrain_first_layer:
                    self.crbms[l]=capsule_rbm3.capsule_rbm3(features=None, M=self.K[l-1]*self.J[l-1][0], K=self.K[l], J=self.J[l], visible_type=self.hidden_type[l-1], visible_type_fixed_param=self.hidden_type_fixed_param[l-1], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=True, a=b12l, fix_a_log_ind=None, xz_interaction=self.xz_interaction, rec_error_max=10*self.rec_error_max, tol_poisson_max=self.tol_poisson_max, rng=self.rng)
                    
                    self.crbms[l].fix_vis_bias( a=b12l )
                    self.crbms[l].train(X=crbm_X, X_validate=crbm_X_validate, batch_size=batch_size, NMF=NMF, increase_NMF_at=increase_NMF_at, increased_NMF=increased_NMF, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=learn_rate_b[l-1], learn_rate_b=learn_rate_b[l], learn_rate_c=learn_rate_c[l], learn_rate_W=learn_rate_W[l], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, reg_lambda_a=0, reg_alpha_a=1, reg_lambda_b=0, reg_alpha_b=1, reg_lambda_W=0, reg_alpha_W=1, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, reinit_a_use_data_stat=False, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_"+str(l), figwidth=figwidth, figheight=figheight)
                    _,b,c,W=self.crbms[l].get_param()
                    bl=self.merge_bl(b, self.hidden_type[l])
                    b12l=[None,None]
                    if self.hidden_type[l] in ["Gaussian","Gaussian2"]:
                        b12l[0]=bl[0]
                        b12l[1]=bl[1]
                        self.b[l]=[b[k][0] for k in range(self.K[l])]
                        self.b2[l]=[b[k][1] for k in range(self.K[l])]
                    else:
                        b12l[0]=bl
                        self.b[l]=b
                    self.c[l]=c
                    self.W[l]=self.split_Wl(W, self.J[l-1][0],l)
                    self.br[l]=copy.deepcopy(self.b[l])
                    self.b2r[l]=copy.deepcopy(self.b2[l])
                    self.cr[l]=copy.deepcopy(self.c[l])
                    self.Wr[l]=copy.deepcopy(self.W[l])
                    if l<self.L-1:
                        Xl,Hl,Zl=self.crbms[l].mean_field_approximate_inference(crbm_X, NMF=NMF)
                        Xl_validate,Hl_validate,Zl_validate=self.crbms[l].mean_field_approximate_inference(crbm_X_validate, NMF=NMF)
                        crbm_X=self.merge_ZlHl(Zl, Hl)
                        crbm_X_validate=self.merge_ZlHl(Zl_validate, Hl_validate)
                        self.H_pretrain[l]=Hl
                        self.Z_pretrain[l]=Zl
        end_time = time.clock()
        self.pretrain_time=end_time-start_time
        return self.pretrain_time
        print("It took {0} seconds.".format(self.pretrain_time))
        self.backup_param(0)
        
        
#    def split_Wl(self, Wl, I):
#        """
#        Split a list of weight matrices to a list of list of matrices: W[l][k] -> W[0][k], W[0][k], ..., W[k1][k], ..., W[K1-1][k]. W[k1][k] is of size I X J
#        Wl: list of matrices.
#        I: scalar, the row dimension of a split matrix.
#        """
#        K=len(Wl)
#        K=K-1
#        
#        num_input_feat,J=Wl[0].shape
#        K1=int(num_input_feat/I)
#        
#        Wls=[[None]*K for k1 in range(K1)]
#        for k in range(K):
#            for k1 in range(K1):
#                Wls[k1][k] = Wl[k][k1*I:(k1+1)*I,:]
##        if self.xz_interaction:
##            Wls.append(Wl[K])
##        else:
##            Wls.append(None)
#        Wls.append(Wl[K]) # the pretraining does not learning  x-z and z-z interactions. Thus, keep the initialized values to the fine-tuning.
#        return Wls
    

    def split_Wl(self, Wl, I, l):
        """
        Split a list of weight matrices to a list of list of matrices: W[l][k] -> W[0][k], W[0][k], ..., W[k1][k], ..., W[K1-1][k]. W[k1][k] is of size I X J
        Wl: list of matrices.
        I: scalar, length of a capsule, the row dimension of a split matrix.
        l: layer index. Here l>0. That is not for x-z interaction. It is for z-z interaction.
        """
        K=len(Wl) # number of capsules at the upper layer
        
        num_input_feat,J=Wl[0].shape
        K1=int(num_input_feat/I) # number of capsules at the bottom layer
        
        Wls=[[None]*K for k1 in range(K1)]
        for k in range(K):
            for k1 in range(K1):
                Wls[k1][k] = Wl[k][k1*I:(k1+1)*I,:]
        #Wls.append(Wl[K]) # the pretraining does not learning  x-z and z-z interactions. Thus, keep the initialized values to the fine-tuning.
        return Wls
    

    def merge_ZlHl(self, Zl, Hl):
        K=Zl.shape[0]
        ZlHl_list=[ Zl[k,:]*Hl[k] for k in range(K)]
        ZlHl_arr=numpy.vstack(ZlHl_list)
        return ZlHl_arr


    def train(self, X=None, X_validate=None, batch_size=10, NMF=20, increase_NMF_at=None, increased_NMF=[20], maxiter=100, learn_rate_a=0.01, learn_rate_b=0.01, learn_rate_c=0.01, learn_rate_d=0.01, learn_rate_W=0.01, change_rate=0.8, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=10, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, if_plot_error_free_energy=False, dir_save="./", prefix="CDGM", figwidth=5, figheight=5):
        """
        Wake-sleep algorithm to train Capsule-DGM.
        Different layers can have different learning rates.
        """
        start_time=time.clock()
        print("Start training CDGM...")
        if self.visible_type=="Multinoulli": # convert to binary
            self.X=[None]*self.M
            self.X_validate=[None]*self.M
            for m in range(self.M):
                Z,_=cl.membership_vector_to_indicator_matrix(X[m,:],z_unique=range(self.Ms[m]))
                self.X[m]=Z.transpose()
                self.N=self.X[0].shape[1]
                if X_validate is not None:
                    Z,_=cl.membership_vector_to_indicator_matrix(X_validate[m,:],z_unique=range(self.Ms[m]))
                    self.X_validate[m]=Z.transpose()
                    self.N_validate=self.X_validate[0].shape[1] # number of validation samples
        else: # not multinoulli variables
            self.X=X
            self.N=self.X.shape[1] # number of training samples
            self.X_validate=X_validate
            if X_validate is not None:
                self.N_validate=self.X_validate.shape[1] # number of validation samples
            else:
                self.N_validate=0

        #self.reinit_a()

        self.batch_size=batch_size
        if self.batch_size>self.N:
            self.batch_size=1
                
        if self.L==1:
            print("There is only one hidden layer. This is just a CRBM, a pretraining is thus enough. I decide to exit.")
            return 0
    
        # different layers have different learning rates         
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*self.L
        if numpy.isscalar(learn_rate_c):
            learn_rate_c=[learn_rate_c]*self.L
        if numpy.isscalar(learn_rate_d):
            learn_rate_c=[learn_rate_d]*self.L
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*self.L       
        
        self.maxiter=maxiter
        self.learn_rate_a=learn_rate_a
        self.learn_rate_b=learn_rate_b
        self.learn_rate_c=learn_rate_c
        self.learn_rate_d=learn_rate_d
        self.learn_rate_W=learn_rate_W
        self.change_rate=change_rate
        self.change_every_many_iters=change_every_many_iters
        
        self.rec_errors_train=[]
        self.rec_errors_valid=[]
        self.mfes_train=[]
        self.mfes_valid=[]

        for i in range(self.maxiter):
            
            if adjust_change_rate_at is not None:
                if i==adjust_change_rate_at[0]:
                    change_rate=change_rate*adjust_coef # increast change_rate
                    change_rate=1.0 if change_rate>1.0 else change_rate # make sure not greater than 1
                    if len(adjust_change_rate_at)>1:
                        adjust_change_rate_at=adjust_change_rate_at[1:] # delete the first element
                    else:
                        adjust_change_rate_at=None
                        
            # change learning rates
            self.learn_rate_a=self.change_learning_rate(current_learn_rate=self.learn_rate_a, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_b=self.change_learning_rate(current_learn_rate=self.learn_rate_b, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_c=self.change_learning_rate(current_learn_rate=self.learn_rate_c, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_d=self.change_learning_rate(current_learn_rate=self.learn_rate_d, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_W=self.change_learning_rate(current_learn_rate=self.learn_rate_W, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            #print "starting the {0}-th iteration, the learning rate of a, b, W: {1}, {2}, {3}".format(i,self.learn_rate_a,self.learn_rate_b,self.learn_rate_W)
            # get mini-batch
            
            ## wake phase
            Xbatch=self.sample_minibatch(self.batch_size)
            XbatchMg,HbatchMg,ZbatchMg,AbatchMg,Hbatchr,Zbatchr,Abatchr,a_hat_gen,b_hat_gen,c_hat_gen,d_hat_gen=self.sample_xhza_wake(Xbatch, NMF=NMF, value_or_meanfield="meanfield", compute_HMg=True)   
            self.compute_gradient_wake(Xbatch, Hbatchr, Zbatchr, Abatchr, XbatchMg, HbatchMg, ZbatchMg, AbatchMg)
            self.update_param_wake()
            
            # update the parameters for CRBMs
            #self.update_crbms()
            
            ## sleep phase 
            #Xfantacy,XfantacyM,Hfantacy,Zfantacy,Afantacy,HfantacyMr,ZfantacyMr,AfantacyMr=self.sample_xhza_sleep(NS=self.batch_size, Hg=None, Zg=None, Ag=None, warm_start=False, NMF=NMF, value_or_meanfield="meanfield", compute_HMr=True)
            
            Xfantacy,XfantacyM,Hfantacy,Zfantacy,Afantacy,HfantacyMr,ZfantacyMr,AfantacyMr=self.sample_xhza_sleep(NS=self.batch_size, Hg=copy.deepcopy(Hbatchr), Zg=copy.deepcopy(Zbatchr), Ag=copy.deepcopy(Abatchr), warm_start=False, NMF=NMF, value_or_meanfield="meanfield", compute_HMr=True)
            
            
            self.compute_gradient_sleep(Xfantacy, Hfantacy, Zfantacy, Afantacy,  HfantacyMr, ZfantacyMr, AfantacyMr) # try to replace Xfantacy by XfantacyM
            self.update_param_sleep()
            
            print("Zbatchr[0].sum(axis=1)=")
            print(Zbatchr[0].sum(axis=1))
            #print(Abatchr[0].sum(axis=2))
            #print(Zbatchr[0])
            if self.L>=2:            
                print("Zbatchr[1].sum(axis=1)=")
                print(Zbatchr[1].sum(axis=1))
                #print(Abatchr[1].sum(axis=2))
                #print(Zbatchr[1])
            if self.L>=3:            
                print("Zbatchr[2].sum(axis=1)=")
                print(Zbatchr[2].sum(axis=1))
                #print(Zbatchr[2])
                
            print("Zfantacy[0].sum(axis=1)=")
            print(Zfantacy[0].sum(axis=1))
            #print(Afantacy[0].sum(axis=2))
            #print(Zfantacy[0])
            if self.L>=2:
                print("Zfantacy[1].sum(axis=1)=")
                print(Zfantacy[1].sum(axis=1))
                #print(Afantacy[1].sum(axis=2))
                #print(Zfantacy[1])
            if self.L>=3:
                print("Zfantacy[2].sum(axis=1)=")
                print(Zfantacy[2].sum(axis=1))
                #print(Zfantacy[2])
            
            # backup parameters
            self.backup_param(i, dif=500)
            
            # compute reconstruction error of the training samples
            # sample some training samples, rather than use all training samples which is time-consuming
            if track_reconstruct_error:
                rec_error_train,_,_,_,_,_,_,_=self.compute_reconstruction_error(X0=Xbatch, X0RM=XbatchMg )
                self.rec_errors_train.append(rec_error_train)
                
                if rec_error_train>self.rec_error_max or math.isnan(rec_error_train):
                    self.reset_param_use_backup(i, dif=250)
             
            #we can monitor the lower bound for each iteration    
                
            if track_free_energy:
                mfe_train,_=self.compute_free_energy(X=Xbatch, HMr=Hbatchr, a_hat_gen=a_hat_gen, b_hat_gen=b_hat_gen)
                self.mfes_train.append(mfe_train)
                
            if self.X_validate is not None:
                if valid_subset_size_for_compute_error is not None:
                    self.valid_subset_ind=self.rng.choice(numpy.arange(self.N_validate,dtype=int),size=valid_subset_size_for_compute_error)
                    if self.visible_type=="Multinoulli":
                        X_validate_subset=[None]*self.M
                        for m in range(self.M):
                            X_validate_subset[m]=self.X_validate[m][:,self.valid_subset_ind]
                    else:
                        X_validate_subset=self.X_validate[:,self.valid_subset_ind]
                    if track_reconstruct_error:
                        rec_error_valid,HMr_valid,ZMr_valid,AMr_valid,a_hat_gen_valid,b_hat_gen_valid,c_hat_gen_valid,d_hat_gen_valid=self.compute_reconstruction_error(X0=X_validate_subset, X0RM=None)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=X_validate_subset, HMr=HMr_valid, a_hat_gen=a_hat_gen_valid, b_hat_gen=b_hat_gen_valid)
                        self.mfes_valid.append(mfe_validate)
                else:
                    if track_reconstruct_error:                  
                        rec_error_valid,HMr_valid,ZMr_valid,AMr_valid,a_hat_gen_valid,b_hat_gen_valid,c_hat_gen_valid,d_hat_gen_valid=self.compute_reconstruction_error(X0=self.X_validate, X0RM=None)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=self.X_validate, HMr=HMr_valid, a_hat_gen=a_hat_gen_valid, b_hat_gen=b_hat_gen_valid)
                        self.mfes_valid.append(mfe_validate)
                # compute difference of free energy between training set and validation  set
                # the log-likelihood(train_set) - log-likelihood(validate_set) = F(validate_set) - F(train_set), the log-partition function, logZ is cancelled out
#                if track_reconstruct_error and track_free_energy:
#                    free_energy_dif=mfe_train - mfe_validate
#                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}, free_energy_train: {4}, free_energy_valid: {5}, free_energy_dif: {6}".format(i, self.learn_rate_W, rec_error_train, rec_error_valid, mfe_train, mfe_validate, free_energy_dif))
#                elif not track_reconstruct_error and track_free_energy:
#                    free_energy_dif=mfe_train - mfe_validate
#                    print("{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}, free_energy_valid: {3}, free_energy_dif: {4}".format(i, self.learn_rate_W, mfe_train, mfe_validate, free_energy_dif))
#                elif track_reconstruct_error and not track_free_energy:
#                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}".format(i, self.learn_rate_W, rec_error_train, rec_error_valid))
#                elif not track_reconstruct_error and not track_free_energy:
#                    print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W))
#            else:
#                if track_reconstruct_error and track_free_energy:
#                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, free_energy_train: {3}".format(i, self.learn_rate_W, rec_error_train, mfe_train))
#                elif not track_reconstruct_error and track_free_energy:
#                    print("{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}".format(i, self.learn_rate_W, mfe_train))
#                elif track_reconstruct_error and not track_free_energy:
#                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}".format(i, self.learn_rate_W, rec_error_train))
#                elif not track_reconstruct_error and not track_free_energy:
#                    print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W))
                    
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
            self.plot_error_free_energy(dir_save, prefix=prefix, figwidth=figwidth, figheight=figheight)

        print("The (fine-tuning) training of CDGM is finished!")
        end_time = time.clock()
        self.train_time=end_time-start_time
        return self.train_time
        print("It took {0} seconds.".format(self.train_time))
    
    
    def sample_xhza_wake(self, X, NMF=20, value_or_meanfield="meanfield", compute_HMg=True):
        """
        Use the recognition parameters to sample hidden states.
        """
        NS=X.shape[1] # number of samples
        # sample mean of h,z
        H=[[None]*self.K[l] for l in range(self.L)]
        Z=[None]*self.L
        A=[None]*(self.L-1) # A[l] is a 3-way array A[l][k1,k2,n] 
        b_hat=copy.deepcopy(self.br)
        b2_hat=copy.deepcopy(self.b2r)
        c_hat=copy.deepcopy(self.cr)
        d_hat=copy.deepcopy(self.dr)
        for l in range(self.L):
            if l<self.L-1:
                hidden_type_Zl="Bernoulli"
            else:
                hidden_type_Zl="Bernoulli"
            if l==0:
                # mean-field
                for n in range(NMF):
                    # reset c_hat
                    if self.xz_interaction:
                        c_hat[l] = self.cr[l] + self.Wr[l][self.K[l]].T @ X
                    else:
                        c_hat[l] = numpy.tile(self.cr[l], NS)
                        
                    for k in range(self.K[l]):
                        if n==0: # # initialize H[0]
                            H[l][k],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias([b_hat[l][k],b2_hat[l][k]],NS,self.hidden_type[l]), hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
                        c_hat[l][k,:] = c_hat[l][k,:] + numpy.diag( X.T @ self.Wr[l][k] @ H[l][k] )
                    
                    # sample Z
                    #print("c_hat[l]=")
                    #print(c_hat[l])
                    #print("Wr[0][0]=")
                    #print(self.Wr[0][0])
                    #print("H[0][0]=")
                    #print(H[0][0])
                    Z[l],_=self.sample_h_given_b_hat(b_hat=c_hat[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield) #value_or_meanfield
                    
                    # b_hat
                    b_hat[l]=copy.deepcopy(self.br[l])
                    for k in range(self.K[l]):
                        if self.hidden_type[l]=="Bernoulli":
                            b_hat[l][k]= b_hat[l][k] + Z[l][k,:] * (self.Wr[l][k].T @ X)
                        elif self.hidden_type[l]=="Gaussian_FixPrecision1":
                            pass
                        elif self.hidden_type[l]=="Gaussian":
                            b_hat[l][k] = b_hat[l][k] + Z[l][k,:] * (self.Wr[l][k].T @ X)
                        # sample H
                        H[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat[l][k],b2_hat[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
            
            else: # middle or last hidden layer
                # mean-field
                for n in range(NMF):
                    
                    if n==0: # initialize A and H
                        A[l-1]=numpy.zeros((self.K[l-1],self.K[l],NS))
                        for k1 in range(self.K[l-1]):
                            A[l-1][k1,:,:],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias(self.dr[l-1][k1,:].reshape(self.K[l],1),NS,"Multinomial"), hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield)
                        for k in range(self.K[l]):
                            H[l][k],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias([self.br[l][k],self.b2r[l][k]],NS,self.hidden_type[l]), hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
                    
                    # reset c_hat
                    if self.xz_interaction:
                        c_hat[l] = self.cr[l] +  self.Wr[l][self.K[l-1]].T @ Z[l-1]
                    else:
                        c_hat[l] = numpy.tile(self.cr[l], NS)
                    # sample Z[l]
                    for k in range(self.K[l]):
                        for k1 in range(self.K[l-1]):
                            c_hat[l][k,:] = c_hat[l][k,:] + A[l-1][k1,k,:]*Z[l-1][k1,:]*numpy.diag( H[l-1][k1].T @ self.Wr[l][k1][k] @ H[l][k] )
                    Z[l],_=self.sample_h_given_b_hat(b_hat=c_hat[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield)
                    
                    # sample A[l]
                    #d_hat[l]=copy.deepcopy(self.dr[l])
                    d_hat[l-1]=numpy.zeros((self.K[l-1],self.K[l],NS))
                    for k1 in range(self.K[l-1]):
                        for k in range(self.K[l]):
                            d_hat[l-1][k1,k,:] = self.dr[l-1][k1,k] +  Z[l-1][k1,:]*Z[l][k,:]*numpy.diag( H[l-1][k1].T @ self.Wr[l][k1][k] @ H[l][k])
                        A[l-1][k1,:,:],_ = self.sample_h_given_b_hat(b_hat=d_hat[l-1][k1,:,:], hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield) #value_or_meanfield
                    
                    # b_hat[l] for H[l]
                    b_hat[l]=copy.deepcopy(self.br[l])
                    b2_hat[l]=copy.deepcopy(self.b2r[l])
                    for k in range(self.K[l]):
                        if self.hidden_type[l]=="Bernoulli":
                            for k1 in range(self.K[l-1]):
                                b_hat[l][k] = b_hat[l][k] + A[l-1][k1,k,:]*Z[l][k,:]*Z[l-1][k1,:] * (self.Wr[l][k1][k].T @ H[l-1][k1])
                        elif self.hidden_type[l]=="Gaussian_FixPrecision1":
                            pass
                        elif self.hidden_type[l]=="Gaussian2":
                            pass
                        elif self.hidden_type[l]=="Gaussian":
                            for k1 in range(self.K[l-1]):
                                b_hat[l][k] = b_hat[l][k] + A[l-1][k1,k,:]*Z[l][k,:]*Z[l-1][k1,:] * (self.Wr[l][k1][k].T @ H[l-1][k1])
                        # sample H[l]
                        H[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat[l][k],b2_hat[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
                        
                
        # compute mean of x,h,z using generative parameters and recognition data
        HMg=[[None]*self.K[l] for l in range(self.L)]
        ZMg=[None]*self.L
        AMg=[None]*self.L
        a_hat_gen=copy.deepcopy(self.a)
        b_hat_gen=copy.deepcopy(self.b)#[None]*self.L #copy.deepcopy(self.b)
        b2_hat_gen=copy.deepcopy(self.b2)#[None]*self.L #copy.deepcopy(self.b2)
        c_hat_gen=copy.deepcopy(self.c)#[None]*self.L #copy.deepcopy(self.c)
        d_hat_gen=copy.deepcopy(self.d)#[None]*(self.L-1)
        if self.visible_type in ["Bernoulli", "Poisson", "NegativeBinomial", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
            for k in range(self.K[0]):
                a_hat_gen =  a_hat_gen + Z[0][k,:] * (self.W[0][k] @ H[0][k])
            if self.xz_interaction:
                a_hat_gen = a_hat_gen + self.W[0][self.K[0]] @ Z[0]
                
        elif self.visible_type in ["Gaussian", "Gaussian_Hinton"]:
            for k in range(self.K[0]):
                a_hat_gen[0] =  a_hat_gen[0] + Z[0][k,:] * (self.W[0][k] @ H[0][k])          
            if self.xz_interaction:
                a_hat_gen[0] = a_hat_gen[0] + self.W[0][self.K[0]] @ Z[0]
                
        _,XMg,_=self.sample_visible(visible_type=self.visible_type, a=a_hat_gen, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, rng=self.rng)
        
        if compute_HMg:
            for l in range(self.L-1,-1,-1): 
                if l<self.L-1:
                    hidden_type_Zl="Bernoulli"
                else:
                    hidden_type_Zl="Bernoulli"                      
                if l==self.L-1: # last hidden layer:
                    # c_hat_gen
                    c_hat_gen[l] = numpy.tile(self.c[l], NS)
                    # sample ZMg
                    ZMg[l],_=self.sample_h_given_b_hat(b_hat=c_hat_gen[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield="meanfield")
                    # b_hat_gen
                    # generate HMg
                    for k in range(self.K[l]):
                        b_hat_gen_lk=self.repeat_bias([self.b[l][k],self.b2[l][k]], NS, self.hidden_type[l])
                        if isinstance(b_hat_gen_lk,list):
                            b_hat_gen[l][k]=b_hat_gen_lk[0]
                            b2_hat_gen[l][k]=b_hat_gen_lk[1]
                        else:
                            b_hat_gen[l][k]=b_hat_gen_lk
                            b2_hat_gen[l][k]=None
                        HMg[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat_gen[l][k],b2_hat_gen[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="meanfield")
                    
                else: # not last layer
                    for n in range(NMF): # mean-field
                                                
                        # initialize HMg and AMg
                        if n==0: 
                            AMg[l]=numpy.zeros((self.K[l],self.K[l+1],NS))
                            for k in range(self.K[l]):
                                HMg[l][k],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias([self.b[l][k],self.b2[l][k]],NS,self.hidden_type[l]), hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="meanfield")
                                AMg[l][k,:,:], _ = self.sample_h_given_b_hat(b_hat=self.repeat_bias(self.d[l][k,:].reshape(self.K[l+1],1),NS,"Multinomial"), hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield)
                        
                        # c_hat_gen
                        if self.xz_interaction:
                            c_hat_gen[l] = self.c[l] + self.W[l+1][self.K[l]] @ Z[l+1]
                        else:
                            c_hat_gen[l] = numpy.tile(self.c[l], NS)
                                
                        # sample ZMg
                        for k in range(self.K[l]):
                            for k2 in range(self.K[l+1]):
                                c_hat_gen[l][k,:]= c_hat_gen[l][k,:] + AMg[l][k,k2,:]*Z[l+1][k2,:]*numpy.diag( HMg[l][k].T @ self.W[l+1][k][k2] @ H[l+1][k2] )
                        ZMg[l],_=self.sample_h_given_b_hat(b_hat=c_hat_gen[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield="meanfield")
                        
                        # sample AMg
                        d_hat_gen[l]=numpy.zeros((self.K[l],self.K[l+1],NS))
                    for k in range(self.K[l]):
                        for k2 in range(self.K[l+1]):
                            d_hat_gen[l][k,k2,:] = self.d[l][k,k2] + ZMg[l][k,:]*Z[l+1][k2,:]*numpy.diag( HMg[l][k].T @ self.W[l+1][k][k2] @ H[l+1][k2])
                        AMg[l][k,:,:],_ = self.sample_h_given_b_hat(b_hat=d_hat_gen[l][k,:,:], hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield) # value_or_meanfield
                        
                        # b_hat_gen
                        b_hat_gen[l]=copy.deepcopy(self.b[l])
                        b2_hat_gen[l]=copy.deepcopy(self.b2[l])
                        for k in range(self.K[l]):
                            if self.hidden_type[l]=="Bernoulli":
                                for k2 in range(self.K[l+1]):
                                    b_hat_gen[l][k] = b_hat_gen[l][k] + AMg[l][k,k2,:]*ZMg[l][k,:]*Z[l+1][k2,:]*(self.W[l+1][k][k2] @ H[l+1][k2])
                            elif self.hidden_type[l]=="Gaussian_FixPrecision1":
                                pass
                            elif self.hidden_type[l]=="Gaussian":
                                for k2 in range(self.K[l+1]):
                                    b_hat_gen[l][k]= b_hat_gen[l][k] + AMg[l][k,k2,:]*ZMg[l][k,:]*Z[l+1][k2,:]*(self.W[l+1][k][k2] @ H[l+1][k2])
                            # generate HMg
                            HMg[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat_gen[l][k],b2_hat_gen[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="meanfield")
        
        
        return XMg,HMg,ZMg,AMg,H,Z,A,a_hat_gen,b_hat_gen,c_hat_gen,d_hat_gen


    def repeat_bias(self, b, N, dist):
        # repeat b N times column-wise, for visible bias or hidden bias. b is a vector. or a list of two vectors.
        if dist in ["Bernoulli", "Poisson", "NegativeBinomial", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
            if isinstance(b, list):
                b=b[0] # peel off
            b_rep=numpy.tile(b,N)
        elif dist in ["Gaussian", "Gaussian_Hinton"]:
            b_rep=[None,None]
            b_rep[0]=numpy.tile(b[0], N)
            b_rep[1]=b[1]
        return b_rep
    
    def repeat_bias_new(self, b, N, dist):
        # repeat b N times column-wise, for visible bias or hidden bias. b is a vector. or a list of two vectors.
        if dist in ["Bernoulli", "Poisson", "NegativeBinomial", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
            b_rep0=numpy.tile(b,N)
            b_rep1=None
        elif dist in ["Gaussian", "Gaussian_Hinton"]:
            #b_rep=[None,None]
            b_rep0=numpy.tile(b[0], N)
            b_rep1=b[1]
        return b_rep0,b_rep1

    def repeat_b(self, b, N, dist):
        # repeat b N times column-wise, for a layer of capsules.
        K=len(b) # number of capsules
        b_rep=[None]*K
        if dist in ["Bernoulli", "Poisson", "NegativeBinomial", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
            for k in range(K):
                b_rep[k]=numpy.tile(b[k],N)
        elif dist in ["Gaussian", "Gaussian_Hinton"]:
            for k in range(K):
                b_rep[k]=[None,None]
                b_rep[k][0]=numpy.tile(b[k][0], N)
                b_rep[k][1]=b[k][1]
        return b_rep


    # Not ready for use.
    def compute_posterior_bias_use_recognition_param(self, X, H):
        b_hat=[None]*self.NK
        for nk in range(self.NK):
            if nk==0:
                if self.visible_type=="Multinoulli":
                    b_hat0=self.br[nk]
                    for m in range(self.M):
                        b_hat0 = b_hat0 + numpy.dot( self.Wr[nk][m], X[m] )
                    b_hat[nk]=b_hat0
                else:
                    b_hat[nk]=self.br[nk] + numpy.dot( self.Wr[nk], X )
            else:
                b_hat[nk]=self.br[nk] + numpy.dot( self.Wr[nk], H[nk-1] )
                
        return b_hat


    # Not ready for use.    
    def compute_posterior_bias_use_generative_param(self, H, a_hat_only=False):
        
        # a_hat
        if self.visible_type=="Multinoulli":
            a_hat=[None]*self.M
            for m in range(self.M):
                a_hat[m]= self.a[m] + numpy.dot(self.W[0][m],H[0])
        elif self.visible_type=="Gaussian":
            a_hat=[None]*2
            a1=self.a[0]
            a2=self.a[1]
            a_hat[0]=a1 + numpy.dot(self.W[0],H[0]) 
            a_hat[1]=a2
        elif self.visible_type=="Gaussian_Hinton":
            a1=self.a[0]
            a2=self.a[1]
            a_hat[0]=a1 + 1/a2*numpy.dot(self.W[0],H[0]) 
            a_hat[1]=a2
        else:
            a_hat=self.a + numpy.dot(self.W[0],H[0])

        # b_hat
        b_hat=[None]*self.NK
        if a_hat_only:
            return a_hat,b_hat
            
        for nk in range(self.NK):
            if nk==self.NK-1:
                b_hat[nk]=self.b[nk]
            else:
                b_hat[nk]=self.b[nk] + numpy.dot( self.W[nk+1], H[nk+1] )
        return a_hat,b_hat
        
        
    def compute_gradient_wake(self, Xbatch, Hbatchr, Zbatchr, Abatchr, XbatchMg, HbatchMg, ZbatchMg, AbatchMg):
        """
        Compute gradient in the wake phase to update the generative parameters.
        """
        # a
        if self.visible_type in ["Bernoulli", "Poisson", "NegativeBinomial", "Multinomial", "Gaussian_FixPrecision2"]:
            self.grad_a=-numpy.mean(Xbatch-XbatchMg,axis=1)
            self.grad_a.shape=(self.M,1)
        elif self.visible_type=="Gaussian":
            grad_a1=-numpy.mean(Xbatch-XbatchMg,axis=1)
            XbatchMg2=XbatchMg**2 - 1/(2*self.a[1])
            grad_a2=-numpy.mean(Xbatch**2-XbatchMg2,axis=1)
            grad_a1.shape=(self.M,1)
            grad_a2.shape=(self.M,1)
            self.grad_a=[grad_a1,grad_a2]
        elif self.visible_type=="Gaussian_FixPrecision1":
            self.grad_a=-numpy.mean(self.visible_type_fixed_param*(Xbatch-XbatchMg),axis=1)
            self.grad_a.shape=(self.M,1)
        elif self.visible_type=="Multinoulli":
            self.grad_a=[None]*self.M
            for m in range(self.M):
                grad_am=-numpy.mean(Xbatch[m]-XbatchMg[m])
                grad_am.shape=(self.Ms[m],1)
                self.grad_a[m]=grad_am
        
        # b
        self.grad_b=[[None]*self.K[l] for l in range(self.L) ]
        self.grad_b2=[[None]*self.K[l] for l in range(self.L) ]
        for l in range(self.L):
            if self.hidden_type[l] in ["Bernoulli", "Poisson", "Binomial", "Multinomial", "Gaussian_FixPrecision2"]:
                for k in range(self.K[l]):
                    grad_b_lk=-numpy.mean(Hbatchr[l][k] - HbatchMg[l][k], axis=1)
                    grad_b_lk.shape=(self.J[l][k],1)
                    self.grad_b[l][k]=grad_b_lk
            elif self.hidden_type[l] =="Gaussian_FixPrecision1":
                for k in range(self.K[l]):
                    grad_b_lk=-numpy.mean(self.hidden_type_fixed_param[l][k]*Hbatchr[l][k] - self.hidden_type_fixed_param[l][k]*HbatchMg[l][k], axis=1)
                    grad_b_lk.shape=(self.J[l][k],1)
                    self.grad_b[l][k]=grad_b_lk
            elif self.hidden_type[l] == "Gaussian":
                for k in range(self.K[l]):
                    grad_b_lk=-numpy.mean(Hbatchr[l][k] - HbatchMg[l][k], axis=1)
                    grad_b_lk.shape=(self.J[l][k],1)
                    HbatchMg_lk2=HbatchMg[l][k]**2 - 1/(2*self.b2[l][k])
                    grad_b2_lk=-numpy.mean(Hbatchr[l][k]**2 - HbatchMg_lk2, axis=1) 
                    grad_b2_lk.shape=(self.J[l][k],1)
                    self.grad_b[l][k]=grad_b_lk
                    self.grad_b2[l][k]=grad_b2_lk

        # c
        self.grad_c=[None]*self.L
        for l in range(self.L):
            grad_cl=-numpy.mean(Zbatchr[l]-ZbatchMg[l], axis=1)
            grad_cl.shape=(self.K[l],1)
            self.grad_c[l]=grad_cl
            
        # d
        self.grad_d=[None]*(self.L-1)
        for l in range(self.L-1):
            self.grad_d[l]=-numpy.mean(Abatchr[l]-AbatchMg[l],axis=2)
            
        # W
        self.grad_W=copy.deepcopy(self.W) # just want the shape of W
        self.grad_W2=copy.deepcopy(self.W2) # just want the shape of W2
        for l in range(self.L):
            if l==0:
                for k in range(self.K[l]):
                    grad_Wlk = - ( (Xbatch - XbatchMg) @ (Zbatchr[l][k,:]*Hbatchr[l][k]).T )/self.batch_size
                    self.grad_W[l][k]=grad_Wlk
                # W for X Omega_0 Z_0, W[0][K[0]] = Omega_0 
                if self.xz_interaction:
                    self.grad_W[l][self.K[l]]= - ( (Xbatch - XbatchMg) @ Zbatchr[l].T )/self.batch_size
            else:# not first layer
                for k1 in range(self.K[l-1]):
                    for k in range(self.K[l]):
                        grad_Wlk1k = - ( ((Abatchr[l-1][k1,k,:]*Zbatchr[l-1][k1,:]*Hbatchr[l-1][k1]) - (AbatchMg[l-1][k1,k,:]*ZbatchMg[l-1][k1,:]*HbatchMg[l-1][k1])) @ (Zbatchr[l][k,:]*Hbatchr[l][k]).T )/self.batch_size
                        self.grad_W[l][k1][k]=grad_Wlk1k                                  
                # W for Z_l-1 Omega_l Z_l, W[l][K[l-1]] is Omega_l
                if self.xz_interaction:
                    self.grad_W[l][self.K[l-1]] = -( (Zbatchr[l-1] - ZbatchMg[l-1]) @ Zbatchr[l].T )/self.batch_size
    
    
    def update_param_wake(self):
        """
        Update the generative parameters.
        """
        #print("in update_param...")
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
        if self.visible_type in ["Bernoulli", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else: # fix some a
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            #print "in update_param ..."
            #print self.if_fix_vis_bias
            #print self.a[0:10].transpose()

        elif self.visible_type=="Poisson":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            self.a[self.a>tol_poisson_max]=tol_poisson_max
            
        elif self.visible_type=="NegativeBinomial":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            # a not too small, not positive,s [-20,0)
            self.a[self.a>=0]=-tol # project a to negative
            self.a[self.a<tol_negbin_min]=tol_negbin_min

            
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
            self.a[1][self.a[1]>=0]=-tol
            
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
        
        # update b
        for l in range(self.L):
            if self.hidden_type[l] in ["Bernoulli", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
                for k in range(self.K[l]):
                    self.b[l][k]=self.b[l][k] - self.learn_rate_b[l] * self.grad_b[l][k]
                #print self.b[l][0].transpose()
                    
            elif self.hidden_type[l]=="Gaussian":
                for k in range(self.K[l]):
                    self.b[l][k]= self.b[l][k] - self.learn_rate_b[l][0] * self.grad_b[l][k]
                    self.b2[l][k]= self.b2[l][k] - self.learn_rate_b[l][1] * self.grad_b2[l][k]
                    self.b2[l][k][self.b2[l][k]>=0]=-tol
                    
            elif self.hidden_type[l]=="Gaussian_Hinton":
                for k in range(self.K[l]):
                    self.b[l][k][0]= self.b[l][k][0] - self.learn_rate_b[l][0] * self.grad_b[l][k][0]
                    self.b[l][k][1]= self.b[l][k][1] - self.learn_rate_b[l][1] * self.grad_b[l][k][1]
                    self.b[l][k][1][self.b[l][k][1]<=0]=tol    

        # update c
        for l in range(self.L):
            self.c[l] = self.c[l] - self.learn_rate_c[l] * self.grad_c[l]
            
        # update d
        for l in range(self.L-1):
            self.d[l] = self.d[l] - self.learn_rate_d[l] * self.grad_d[l]

        # update W
        for l in range(self.L):
            if l==0:
                for k in range(self.K[l]):
                    if self.visible_type=="Multinoulli":
                        for m in range(self.M):
                            self.W[l][m][k]=self.W[l][m][k] - self.learn_rate_W[l] * self.grad_W[l][m][k]
                    elif self.visible_type=="Gaussian2":
                        self.W[l][k] = self.W[l][k] - self.learn_rate_W[l][0] * self.grad_W[l][k]
                        self.W2[l][k] = self.W2[l][k] - self.learn_rate_W[l][1] * self.grad_W2[l][k]
                    else:
                        self.W[l][k]=self.W[l][k] - self.learn_rate_W[l] * self.grad_W[l][k]
                if self.xz_interaction:
                    self.W[l][self.K[l]]=self.W[l][self.K[l]] - self.learn_rate_W[l] * self.grad_W[l][self.K[l]]
                    
            else: # not first layer
                for k1 in range(self.K[l-1]):
                    for k in range(self.K[l]):
                        if self.hidden_type[l]=="Gaussian2":
                            self.W[l][k1][k] = self.W[l][k1][k] - self.learn_rate_W[l][0] * self.grad_W[l][k1][k]
                            self.W2[l][k1][k] = self.W2[l][k1][k] - self.learn_rate_W[l][1] * self.grad_W2[l][k1][k]
                        else:
                            self.W[l][k1][k]=self.W[l][k1][k] - self.learn_rate_W[l] * self.grad_W[l][k1][k]
                if self.xz_interaction:
                    self.W[l][self.K[l-1]]=self.W[l][self.K[l-1]] - self.learn_rate_W[l] * self.grad_W[l][self.K[l-1]]


    def sample_xhza_sleep(self, NS=100, Hg=None, Zg=None, Ag=None, warm_start=False, NMF=20, value_or_meanfield="meanfield", compute_HMr=True):
        """
        Use the generative parameters to sample hidden states and visible states.
        Hg: None or a list of length of NK. If Hg is a list, Hg[-1] is a matrix, the rest may be None's. This is used in MDBN etc.
        warm_start: logical, if Ture, use Hr/HMr and Zr/ZMr to initialize Hg, Zg and Ag. This requires Hg, Zg and Ag are not None as inputs.
        """
        if Hg is None:
            #Hg=[None]*self.L
            #for l in range(self.L):
            #    Hg[l]=[None]*self.K[l]
            Hg=[[None]*self.K[l] for l in range(self.L)]
            Zg=[None]*self.L
            last=self.L-1
        else:
            last=self.L-2 # in this case, NS is not used.
        
        if Ag is None:
            Ag=[None]*(self.L-1)
            
        b_hat_gen = copy.deepcopy(self.b)
        b2_hat_gen = copy.deepcopy(self.b2)
        c_hat_gen = copy.deepcopy(self.c)
        d_hat_gen = copy.deepcopy(self.d)
        
        for l in range(last,-1,-1):
            
            hidden_type_Zl="Bernoulli"
            if l==self.L-1:
                hidden_type_Zl="Bernoulli"
                
            if l==self.L-1:
                # generate Zg
                c_hat_gen[l] = numpy.tile(self.c[l], NS)
                Zg[l],_=self.sample_h_given_b_hat(b_hat=c_hat_gen[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield="value")
                # generate Hg
                for k in range(self.K[l]):
                    b_hat_gen_lk=self.repeat_bias([self.b[l][k],self.b2[l][k]], NS, self.hidden_type[l])
                    if isinstance(b_hat_gen_lk,list):
                        b_hat_gen[l][k]=b_hat_gen_lk[0]
                        b2_hat_gen[l][k]=b_hat_gen_lk[1]
                    else:
                        b_hat_gen[l][k]=b_hat_gen_lk
                    Hg[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat_gen[l][k],b2_hat_gen[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="value")
                    
            else: # not last layer
                if Hg[l] is None:
                    Hg[l]=[None]*self.K[l]
                for n in range(NMF):
                    # initialize Hg and Ag
                    if n==0 and not warm_start:
                        Ag[l]=numpy.zeros((self.K[l],self.K[l+1],NS))
                        for k in range(self.K[l]):
                            Ag[l][k,:,:], _ = self.sample_h_given_b_hat(b_hat=self.repeat_bias(self.d[l][k,:].reshape(self.K[l+1],1),NS,"Multinomial"), hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield)
                        
                            Hg[l][k],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias([self.b[l][k],self.b2[l][k]],NS,self.hidden_type[l]), hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
                    
                    # initialize c_hat_gen
                    if self.xz_interaction:
                        c_hat_gen[l] = self.c[l] + self.W[l+1][self.K[l]] @ Zg[l+1]
                    else:
                        c_hat_gen[l] = numpy.tile(self.c[l], NS)
                    
                    # sample Zg
                    for k in range(self.K[l]):
                        for k2 in range(self.K[l+1]):
                            c_hat_gen[l][k,:]= c_hat_gen[l][k,:] + Ag[l][k,k2,:]*Zg[l+1][k2,:]*numpy.diag( Hg[l][k].T @ self.W[l+1][k][k2] @ Hg[l+1][k2] )
                    # generate Zg
                    Zg[l],_=self.sample_h_given_b_hat(b_hat=c_hat_gen[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield) 
                    
                    # sample Ag
                    d_hat_gen[l]=numpy.zeros((self.K[l],self.K[l+1],NS))
                    for k in range(self.K[l]):
                        for k2 in range(self.K[l+1]):
                            d_hat_gen[l][k,k2,:] = self.d[l][k,k2] + Zg[l][k,:]*Zg[l+1][k2,:]*numpy.diag( Hg[l][k].T @ self.W[l+1][k][k2] @ Hg[l+1][k2])
                        Ag[l][k,:,:],_ = self.sample_h_given_b_hat(b_hat=d_hat_gen[l][k,:,:], hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield) #value_or_meanfield
                    
                    # sample Hg
                    b_hat_gen[l]=copy.deepcopy(self.b[l])
                    for k in range(self.K[l]):
                        if self.hidden_type[l]=="Bernoulli":
                            for k2 in range(self.K[l+1]):
                                b_hat_gen[l][k] = b_hat_gen[l][k] + Ag[l][k,k2,:]*Zg[l][k,:]*Zg[l+1][k2,:]*(self.W[l+1][k][k2] @ Hg[l+1][k2])
                        elif self.hidden_type[l]=="Gaussian":
                            for k2 in range(self.K[l+1]):
                                b_hat_gen[l][k] = b_hat_gen[l][k] + Ag[l][k,k2,:]*Zg[l][k,:]*Zg[l+1][k2,:]*(self.W[l+1][k][k2] @ Hg[l+1][k2])
                        # generate Hg
                        Hg[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat_gen[l][k],b2_hat_gen[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
            
        # generate Xg
        a_hat_gen=copy.deepcopy(self.a)
        for k in range(self.K[0]):
            if self.visible_type in ["Bernoulli", "Poisson", "NegativeBinomial", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
                a_hat_gen = a_hat_gen + Zg[0][k,:] * (self.W[0][k] @ Hg[0][k])
            elif self.visible_type in ["Gaussian", "Gaussian_Hinton"]:
                a_hat_gen[0] = a_hat_gen[0] + Zg[0][k,:] * (self.W[0][k] @ Hg[0][k])
        if self.xz_interaction:
            if  self.visible_type in ["Bernoulli", "Poisson", "NegativeBinomial", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
                a_hat_gen = a_hat_gen + self.W[0][self.K[0]] @ Zg[0]
            elif self.visible_type in ["Gaussian", "Gaussian_Hinton"]:
                a_hat_gen[0] = a_hat_gen[0] + self.W[0][self.K[0]] @ Zg[0]
        Xg,XMg,_=self.sample_visible(visible_type=self.visible_type, a=a_hat_gen, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, rng=self.rng)
        #if value_or_meanfield=="meanfield":
        #    Xg=XMg
            
        # compute expected h and z using recognition parameters and Hg and Zg
        HMr=None
        ZMr=None
        AMr=None
        if compute_HMr:
            if warm_start:
                HMr=copy.deepcopy(Hg)
                ZMr=copy.deepcopy(Zg)
                AMr=copy.deepcopy(Ag)
            else:
                HMr=[[None]*self.K[l] for l in range(self.L)]
                ZMr=[None]*self.L
                AMr=[None]*(self.L-1)
            b_hat=copy.deepcopy(self.br)
            b2_hat=copy.deepcopy(self.b2r)
            c_hat=copy.deepcopy(self.cr)
            d_hat=copy.deepcopy(self.dr)
            for l in range(self.L):
                hidden_type_Zl="Bernoulli"
                if l==self.L-1:
                    hidden_type_Zl="Bernoulli" #Multinomial
                if l==0:
                    # mean-field
                    for n in range(NMF):
                        # reset c_hat
                        if self.xz_interaction:
                            c_hat[l] = self.cr[l] + self.Wr[l][self.K[l]].T @ Xg
                        else:
                            c_hat[l] = numpy.tile(self.cr[l], NS)
                            
                        for k in range(self.K[l]):
                            if n==0 and not warm_start: # # initialize HMr[l]
                                HMr[l][k],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias([b_hat[l][k],b2_hat[l][k]],NS,self.hidden_type[l]), hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="meanfield")
                            c_hat[l][k,:] = c_hat[l][k,:] + numpy.diag( Xg.T @ self.Wr[l][k] @ HMr[l][k] )
                        # sample Z
                        ZMr[l],_=self.sample_h_given_b_hat(b_hat=c_hat[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield="meanfield")
                        # b_hat
                        b_hat[l]=copy.deepcopy(self.br[l])
                        for k in range(self.K[l]):
                            if self.hidden_type[l]=="Bernoulli":
                                b_hat[l][k]= b_hat[l][k] + ZMr[l][k,:] * (self.Wr[l][k].T @ Xg)
                            elif self.hidden_type[l]=="Gaussian":
                                b_hat[l][k] = b_hat[l][k] + ZMr[l][k,:] * (self.Wr[l][k].T @ Xg)
                            # sample H
                            HMr[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat[l][k],b2_hat[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="meanfield")
                        
                else: # middle or last hidden layer
                    # mean-field
                    for n in range(NMF):
                        # reset c_hat
                        if self.xz_interaction:
                            c_hat[l] = self.cr[l] +  self.Wr[l][self.K[l-1]].T @ Zg[l-1]
                        else:
                            c_hat[l] = numpy.tile(self.cr[l], NS)
                            
                        # initialize HMr[l] and AMr[l-1]
                        if n==0 and not warm_start:
                            AMr[l-1]=numpy.zeros((self.K[l-1],self.K[l],NS))
                            for k1 in range(self.K[l-1]):
                                AMr[l-1][k1,:,:],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias(self.dr[l-1][k1,:].reshape(self.K[l],1),NS,"Multinomial"), hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield) #value_or_meanfield
                            for k in range(self.K[l]):
                                HMr[l][k],_ = self.sample_h_given_b_hat(b_hat=self.repeat_bias([self.br[l][k],self.b2r[l][k]],NS,self.hidden_type[l]), hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield="meanfield")
                            
                        # c_hat
                        for k in range(self.K[l]):
                            for k1 in range(self.K[l-1]):
                                c_hat[l][k,:]= c_hat[l][k,:] + AMr[l-1][k1,k,:]*Zg[l-1][k1,:]*numpy.diag( Hg[l-1][k1].T @ self.Wr[l][k1][k] @ HMr[l][k])
                        # sample Z
                        ZMr[l],_=self.sample_h_given_b_hat(b_hat=c_hat[l], hidden_type=hidden_type_Zl, hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield)
                        
                        # sample AMr[l]
                        #d_hat[l]=copy.deepcopy(self.dr[l])
                        d_hat[l-1]=numpy.zeros((self.K[l-1],self.K[l],NS))
                        for k1 in range(self.K[l-1]):
                            for k in range(self.K[l]):
                                d_hat[l-1][k1,k,:] = self.dr[l-1][k1,k] +  Zg[l-1][k1,:]*ZMr[l][k,:]*numpy.diag( Hg[l-1][k1].T @ self.Wr[l][k1][k] @ HMr[l][k])
                            AMr[l-1][k1,:,:],_ = self.sample_h_given_b_hat(b_hat=d_hat[l-1][k1,:,:], hidden_type="Multinomial", hidden_type_fixed_param=1, hidden_value_or_meanfield=value_or_meanfield)
                        
                        # b_hat
                        b_hat[l]=copy.deepcopy(self.br[l])
                        for k in range(self.K[l]):
                            if self.hidden_type[l]=="Bernoulli":
                                for k1 in range(self.K[l-1]):
                                    b_hat[l][k] = b_hat[l][k] + AMr[l-1][k1,k,:]*ZMr[l][k,:]*Zg[l-1][k1,:] * (self.Wr[l][k1][k].T @ Hg[l-1][k1])
                            elif self.hidden_type[l]=="Gaussian":
                                for k1 in range(self.K[l-1]):
                                    b_hat[l][k] = b_hat[l][k] + AMr[l-1][k1,k,:]*ZMr[l][k,:]*Zg[l-1][k1,:] * (self.Wr[l][k1][k].T @ Hg[l-1][k1])
                            # sample H
                            HMr[l][k],_ = self.sample_h_given_b_hat(b_hat=[b_hat[l][k],b2_hat[l][k]], hidden_type=self.hidden_type[l], hidden_type_fixed_param=self.hidden_type_fixed_param[l], hidden_value_or_meanfield=value_or_meanfield)
        
        return Xg,XMg,Hg,Zg,Ag,HMr,ZMr,AMr


    def generate_x(self, NS=100, NMF=20, num_iter=1000, method="Gibbs", init=True, X_init=None):
        """
        Generate samples from the learned exp-HM using Gibbs sampling or ancestral sampling.
        method: either "Gibbs" or "ancestral".
        """
        if method=="ancestral":
            self.Xg,self.XMg,self.Hg,self.Zg,self.Ag,_,_,_=self.sample_xhza_sleep(NS=NS, Hg=None, Zg=None, Ag=None, warm_start=False, NMF=NMF, value_or_meanfield="meanfield", compute_HMr=False)

        elif method=="Gibbs":
            if init:
                if X_init is None:
                    self.Xg,self.XMg,self.Hg,self.Zg,self.Ag,_,_,_=self.sample_xhza_sleep(NS=NS, Hg=None, Zg=None, Ag=None, warm_start=False, NMF=NMF, value_or_meanfield="meanfield", compute_HMr=False)
                else:
                    self.Xg=X_init
            
            for i in range(num_iter):
                _,_,_,_,self.Hr,self.Zr,self.Ar,_,_,_,_=self.sample_xhza_wake(self.Xg, NMF=NMF, value_or_meanfield="meanfield", compute_HMg=False)
                self.Xg,self.XMg,self.Hg,self.Zg,self.Ag,_,_,_=self.sample_xhza_sleep(Hg=self.Hr, Zg=self.Zr, Ag=self.Ar, warm_start=False, NMF=NMF, value_or_meanfield="meanfield", compute_HMr=False)
        
        return self.Xg,self.XMg,self.Hg,self.Zg,self.Ag


    def generate_samples(self, X_init=None, NS=100, NMF=100, method="Gibbs", sampling_time=4, sampling_num_iter=100, row=28, col=28, dir_save="./", prefix="CDGM"):
        
        for s in range(sampling_time):
            if s==0:
                init=True
            else:
                init=False
            chainX,chainXM,chainHM,chainZM,chainAM=self.generate_x(NS=NS, num_iter=sampling_num_iter, NMF=NMF, method=method, init=init, X_init=X_init)
            # plot sampled data
            sample_set_x_3way=numpy.reshape(chainXM,newshape=(row,col,NS))
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_method_"+method+"_generated_samples_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)
            # plot ZM
            ZM_3way=self.make_Z_matrix(chainZM)
            self.ZM_3way=ZM_3way
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type + "_method_"+method+"_generated_samples_"+str(s)+"_ZM.pdf", data=ZM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
            for l in range(self.L):
                # save the Z code
                numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_method_"+method+"_generated_samples_"+str(s)+"_ZM_layer"+ str(l) +".txt", chainZM[l].transpose(),fmt="%.4f",delimiter="\t")
                #numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_"+str(s)+"_Z_"+ str(l) +".txt", chainZ[l].transpose(),fmt="%s",delimiter="\t")
            
            # sort ZM
            chainZM_sorted=[None]*self.L
            #chainZ_sorted=[None]*self.L
            for l in range(self.L):
                ind=numpy.argsort(chainZM[l].sum(axis=1))
                ind=ind[::-1]
                chainZM_sorted[l]=chainZM[l][ind,:]
                #chainZ_sorted=chainZ[l][ind,:]
                numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type + "_method_"+ method + "_generated_samples_"+str(s)+"_ZM_sorted_layer"+str(l)+".txt", chainZM_sorted[l].transpose(),fmt="%.4f",delimiter="\t")
                #numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_"+str(s)+"_Z_sorted_layer"+str(l)+".txt", chainZ_sorted[l].transpose(),fmt="%s",delimiter="\t")
            ZM_3way_sorted=self.make_Z_matrix(chainZM_sorted)
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type + "_method_" + method+ "_generated_samples_"+str(s)+"_ZM_sorted.pdf", data=ZM_3way_sorted, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
        return chainZM,chainZM_sorted
            

    def make_Z_matrix(self, Z):
        row=self.L
        col=numpy.max(self.K)
        NS=Z[0].shape[1]
        ZT=numpy.zeros((row,col,NS))
        for n in range(NS):
            for r in range(row):
                cstart=0
                if self.K[r]<col:
                    cstart=int(numpy.floor((col-self.K[r])/2))
                cend=cstart+self.K[r]
                ZT[r,cstart:cend,n]=Z[r][:,n]
        return ZT
        

    def infer_hidden_states_given_x(self, train_set_x_sub=None, NMF=100, dir_save="./", prefix="CHM"):
        """
        Use mean-field approximation.
        """
        
        NS=train_set_x_sub.shape[1]
        _,_,_,_,HM,ZM,_,_,_,_,_=self.sample_xhza_wake(train_set_x_sub, NMF=NMF, value_or_meanfield="meanfield", compute_HMg=False)
        ZM_3way=self.make_Z_matrix(ZM)
        self.ZM_3way=ZM_3way
        cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_x_infer_ZM.pdf", data=ZM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
        # save the Z code
        for l in range(self.L):
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_x_infer_ZM_layer_"+ str(l) +".txt", ZM[l].transpose(),fmt="%.4f",delimiter="\t")
            
        # sort ZM
        ZM_sorted=[None]*self.L
        for l in range(self.L):
            ind=numpy.argsort(ZM[l].sum(axis=1))
            ind=ind[::-1]
            ZM_sorted[l]=ZM[l][ind,:]
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_x_infer_ZM_sorted_layer_"+ str(l) +".txt", ZM_sorted[l].transpose(),fmt="%.4f",delimiter="\t")
        ZM_3way_sorted=self.make_Z_matrix(ZM_sorted)
        cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_x_infer_ZM_sorted.pdf", data=ZM_3way_sorted, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
        
        return ZM,ZM_sorted
        

    def compute_gradient_sleep(self, Xfantacy, Hfantacy, Zfantacy, Afantacy,  HfantacyMr, ZfantacyMr, AfantacyMr):
        """
        Compute gradient in the sleep phase to update the recognition parameters.
        """
        # br
        self.grad_br=[[None]*self.K[l] for l in range(self.L) ]
        self.grad_b2r=[[None]*self.K[l] for l in range(self.L) ]
        for l in range(self.L):
            if self.hidden_type[l] in ["Bernoulli", "Poisson", "Binomial", "Multinomial", "Gaussian_FixPrecision2"]:
                for k in range(self.K[l]):
                    grad_br_lk=-numpy.mean(Hfantacy[l][k] - HfantacyMr[l][k], axis=1)
                    grad_br_lk.shape=(self.J[l][k],1)
                    self.grad_br[l][k]=grad_br_lk
            elif self.hidden_type[l] == "Gaussian_FixPrecision1":
                for k in range(self.K[l]):
                    grad_br_lk=-numpy.mean(self.hidden_type_fixed_param[l][k]*Hfantacy[l][k] - self.hidden_type_fixed_param[l][k]*HfantacyMr[l][k], axis=1)
                    grad_br_lk.shape=(self.J[l][k],1)
                    self.grad_br[l][k]=grad_br_lk
            elif self.hidden_type[l] == "Gaussian":
                for k in range(self.K[l]):
                    grad_br_lk=-numpy.mean(Hfantacy[l][k] - HfantacyMr[l][k], axis=1)
                    grad_br_lk.shape=(self.J[l][k],1)
                    HfantacyMr_lk2=HfantacyMr[l][k]**2 - 1/(2*self.b2[l][k])
                    grad_b2r_lk=-numpy.mean(Hfantacy[l][k]**2 - HfantacyMr_lk2, axis=1) 
                    grad_b2r_lk.shape=(self.J[l][k],1)
                    self.grad_br[l][k]=grad_br_lk
                    self.grad_b2r[l][k]=grad_b2r_lk

        # cr
        self.grad_cr=[None]*self.L
        for l in range(self.L):
            grad_crl=-numpy.mean(Zfantacy[l]-ZfantacyMr[l], axis=1)
            grad_crl.shape=(self.K[l],1)
            self.grad_cr[l]=grad_crl
            
        # dr
        self.grad_dr=[None]*(self.L-1)
        for l in range(self.L-1):
            self.grad_dr[l]=-numpy.mean(Afantacy[l]-AfantacyMr[l],axis=2)
              
        # Wr
        self.grad_Wr=copy.deepcopy(self.Wr) # just want the shape of Wr
        self.grad_W2r=copy.deepcopy(self.W2r) # just want the shape of W2r
        for l in range(self.L):
            if l==0:
                for k in range(self.K[l]):
                    grad_Wrlk = - ( Xfantacy @ (Zfantacy[l][k,:]*Hfantacy[l][k] - ZfantacyMr[l][k,:]*HfantacyMr[l][k]).T )/self.batch_size
                    self.grad_Wr[l][k]=grad_Wrlk
                # W for X Omega_0 Z_0, W[0][K[0]] = Omega_0 
                if self.xz_interaction:
                    self.grad_Wr[l][self.K[l]]= - ( Xfantacy @ (Zfantacy[l] - ZfantacyMr[l]).T )/self.batch_size
            else:# not first layer
                for k1 in range(self.K[l-1]):
                    for k in range(self.K[l]):
                        grad_Wrlk1k= - ( (Zfantacy[l-1][k1,:]*Hfantacy[l-1][k1]) @ (Afantacy[l-1][k1,k,:]*Zfantacy[l][k,:]*Hfantacy[l][k] - AfantacyMr[l-1][k1,k,:]*ZfantacyMr[l][k,:]*HfantacyMr[l][k]).T )/self.batch_size
                        self.grad_Wr[l][k1][k]=grad_Wrlk1k                              
                # W for Z_l-1 Omega_l Z_l, W[l][K[l-1]] is Omega_l
                if self.xz_interaction:
                    self.grad_Wr[l][self.K[l-1]]=- ( Zfantacy[l-1] @ (Zfantacy[l] - ZfantacyMr[l]).T )/self.batch_size
    
    
    def update_param_sleep(self):
        """
        Update the recognition parameters.
        """
        #print("in update_param...")
        tol=1e-8
        
        # update br
        for l in range(self.L):
            if self.hidden_type[l] in ["Bernoulli", "Binomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
                for k in range(self.K[l]):
                    self.br[l][k]=self.br[l][k] - self.learn_rate_b[l] * self.grad_br[l][k]
                #print self.b[l][0].transpose()
                    
            elif self.hidden_type[l]=="Gaussian":
                for k in range(self.K[l]):
                    self.br[l][k]= self.br[l][k] - self.learn_rate_b[l][0] * self.grad_br[l][k]
                    self.b2r[l][k]= self.b2r[l][k] - self.learn_rate_b[l][1] * self.grad_b2r[l][k]
                    self.b2r[l][k][self.b2r[l][k]>=0]=-tol
                    
            elif self.hidden_type[l]=="Gaussian_Hinton":
                for k in range(self.K[l]):
                    self.br[l][k]= self.br[l][k] - self.learn_rate_b[l][0] * self.grad_br[l][k]
                    self.b2r[l][k]= self.b2r[l][k] - self.learn_rate_b[l][1] * self.grad_b2r[l][k]
                    self.b2r[l][k][self.b2r[l][k]<=0]=tol    

        # update cr
        for l in range(self.L):
            self.cr[l] = self.cr[l] - self.learn_rate_c[l] * self.grad_cr[l]

        #update dr
        for l in range(self.L-1):
            self.dr[l] = self.dr[l] - self.learn_rate_d[l] * self.grad_dr[l]
            
        # update Wr
        for l in range(self.L):
            if l==0:
                for k in range(self.K[l]):
                    if self.visible_type=="Multinoulli":
                        for m in range(self.M):
                            self.Wr[l][m][k]=self.Wr[l][m][k] - self.learn_rate_W[l] * self.grad_Wr[l][m][k]
                    elif self.visible_type=="Gaussian2":
                        self.Wr[l][k] = self.Wr[l][k] - self.learn_rate_W[l][0] * self.grad_Wr[l][k]
                        self.W2r[l][k] = self.W2r[l][k] - self.learn_rate_W[l][1] * self.grad_W2r[l][k]
                    else:
                        self.Wr[l][k]=self.Wr[l][k] - self.learn_rate_W[l] * self.grad_Wr[l][k]
                if self.xz_interaction:
                    self.Wr[l][self.K[l]]=self.Wr[l][self.K[l]] - self.learn_rate_W[l] * self.grad_Wr[l][self.K[l]]
                    
            else: # not first layer
                for k1 in range(self.K[l-1]):
                    for k in range(self.K[l]):
                        if self.hidden_type[l]=="Gaussian2":
                            self.Wr[l][k1][k] = self.Wr[l][k1][k] - self.learn_rate_W[l][0] * self.grad_Wr[l][k1][k]
                            self.W2r[l][k1][k][1] = self.W2r[l][k1][k] - self.learn_rate_W[l][1] * self.grad_W2r[l][k1][k]
                        else:
                            self.Wr[l][k1][k]=self.Wr[l][k1][k] - self.learn_rate_W[l] * self.grad_Wr[l][k1][k]
                if self.xz_interaction:
                    self.Wr[l][self.K[l-1]]=self.Wr[l][self.K[l-1]] - self.learn_rate_W[l] * self.grad_Wr[l][self.K[l-1]]


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


    def plot_error_free_energy(self, dir_save="./", prefix="CHM", mean_over=5, figwidth=5, figheight=3):
        
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
        ax1.set_ylabel("Reconstruction Error (RCE)", color="red",fontsize=8)
        for tl in ax1.get_yticklabels():
            tl.set_color("r")
        if self.rec_errors_train.max()>1:
            ax1.set_ylim(0.0,1.0)
        ax1.set_xlabel("Iteration",fontsize=8)
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


    def compute_reconstruction_error(self, X0, X0RM=None):
        """
        Compute the difference between the real sample X0 and the recoverd sample X0RM by mean-field.
        """
        if X0RM is None:
            X0RM,_,_,_,HMr,ZMr,AMr,a_hat_gen,b_hat_gen,c_hat_gen,d_hat_gen=self.sample_xhza_wake(X0, NMF=20, value_or_meanfield="meanfield", compute_HMg=False) # HMr,a_hat_gen,b_hat_gen may be used to compute the free energy as well
        else:
            HMr=None
            ZMr=None
            AMr=None
            a_hat_gen=None
            b_hat_gen=None
            c_hat_gen=None
            d_hat_gen=None
        if self.visible_type=="Multinoulli":
            self.rec_error=0
            for m in range(self.M):
                self.rec_error= self.rec_error + self.rec_error+numpy.mean(numpy.abs(X0RM[m]-X0[m]))
        else:
            self.rec_error=numpy.mean(numpy.abs(X0RM-X0))
        return self.rec_error,HMr,ZMr,AMr,a_hat_gen,b_hat_gen,c_hat_gen,d_hat_gen

    
    # This function is not ready for use.
    def compute_free_energy(self,X=None, HMr=None, a_hat_gen=None, b_hat_gen=None, in_mdbn=False): 
        """
        Compute "free" energy - E_q[log p(x,h)] - H(q). 
        """
        if X is None:
            X=self.X
        if HMr is None:
            _,_,_,HMr,a_hat_gen,b_hat_gen=self.sample_xh_wake(X,compute_HM=False)
            
        # compute E_q[log p(x,h)]
        mean_logpxy,_=self.compute_Eq_log_pxh(X, HMr, a_hat_gen, b_hat_gen, in_mdbn=in_mdbn)
        
        # compute entropy
        mean_entropy,_=self.compute_entropy(HMr, in_mdbn=in_mdbn)
        
        fes= - mean_logpxy - mean_entropy
        
        mfe=numpy.mean(fes) # average over N samples
        return mfe,fes
        
    
    # this function is not ready for use.
    def compute_Eq_log_pxh(self, X, HMr, a_hat_gen, b_hat_gen, in_mdbn=False):
        """
        Compute E [ log p(z) ] = E[ log h(z) + theta^T s(z) - A(theta) ] = E_q [ zeta(z,theta) - A(theta) ] where is theta is posterior generative parameter. 
        """
        
        rbm=restricted_boltzmann_machine.restricted_boltzmann_machine(M=100, K=100) # create a rbm for call its zeta function, the initial parameters of this RBM does not matter
        z=rbm.zeta(a_hat_gen, X, fixed_param=self.visible_type_fixed_param, distribution=self.visible_type)
        logPar=rbm.A(a_hat_gen, fixed_param=self.visible_type_fixed_param, distribution=self.visible_type)

        if in_mdbn:
            last=self.NK-1 # do not consider the last hidden layer, as it will be considered in the joint MDBM
        else:
            last=self.NK
            
        for nk in range(last):
            z = z + rbm.zeta(b_hat_gen[nk], HMr[nk], fixed_param=self.hidden_type_fixed_param[nk], distribution=self.hidden_type[nk])
            logPar = logPar + rbm.A(b_hat_gen[nk], fixed_param=self.hidden_type_fixed_param[nk], distribution=self.hidden_type[nk])
    
        logpxh = z - logPar # for all samples
        
        mean_logpxh=numpy.mean(logpxh)
        
        return mean_logpxh,logpxh
        
        
    def change_learning_rate(self, current_learn_rate, change_rate, current_iter, change_every_many_iters):
        if current_iter!=0 and current_iter%change_every_many_iters==0:
            if numpy.isscalar(current_learn_rate): # scalar
                return current_learn_rate * change_rate
            elif numpy.isscalar(current_learn_rate[0]): # list of learning rates
                new_learn_rate=[c*change_rate for c in current_learn_rate]
                #R=len(current_learn_rate)
                #new_learn_rate=[None]*R
                #for r in range(R):
                #    new_learn_rate[r]=current_learn_rate[r]*change_rate
                return new_learn_rate
            else: # list of lists
                new_learn_rate=[[c*change_rate for c in cs] for cs in current_learn_rate]
                return new_learn_rate
        else:
            return current_learn_rate


    def sample_minibatch(self, batch_size=20):
        ind_batch=self.rng.choice(self.N,size=batch_size,replace=False)
        if self.visible_type=="Multinoulli":
            Xbatch=[None]*self.M
            for m in range(self.M):
                Xbatch[m]=self.X[m][:,ind_batch]
                if batch_size==1:
                    Xbatch[m].shape=(self.Ms[m],1)
        else:
            Xbatch=self.X[:,ind_batch]
            if batch_size==1:
                Xbatch.shape=(self.M,1)
        return Xbatch
        
    # This function is not ready for use.
    def compute_entropy(self, HP, in_mdbn=False):
        """
        Compute the entropy of approximate distribution q(h).
        Only work for Bernoulli and Multinoulli distributions.
        HP: each column of HP[l] is a sample.
        """
        print("I am computing entropy...")
        entropies=0
        num_samples=HP[0].shape[1]
        #print "There are {} samples".format(num_samples)

        if in_mdbn:
            last=self.NK-1 # do not consider the last hidden layer, as it will be considered in the joint MDBM
        else:
            last=self.NK
            
        for n in range(num_samples):
            entropies_n=0
            #print "there are {} hidden layers".format(self.NK)
            for l in range(last):
                if self.hidden_type[l]=="Bernoulli":
                    HPln=HP[l][:,n]
                    for h in HPln:
                        if h!=0 and h!=1:
                            entropies_n= entropies_n -h*numpy.log(h) - (1-h)*numpy.log(1-h)
                elif self.hidden_type[l]=="Multinomial": # only applicable for count is 1, = multinoulli
                    HPln=HP[l][:,n]
                    for h in HPln:
                        if h!=0 and h!=1:
                            entropies_n= entropies_n - h*numpy.log(h)
                elif self.hidden_type[l]=="Binomial":
                    HPln=HP[l][:,n]
                    for h in HPln:
                        if h!=0 and h!=1:
                            entropies_n= entropies_n + 0.5*numpy.log( 2*numpy.pi*numpy.e*self.hidden_type_fixed_param[l]*h*(1-h))
                elif self.hidden_type[l]=="Gaussian_FixPrecision1" or self.hidden_type[l]=="Gaussian_FixPrecision2":
                    for k in range(self.K[l]):
                        entropies_n= entropies_n + 0.5*numpy.log( 2*numpy.pi*numpy.e/self.hidden_type_fixed_param[l])
                elif self.hidden_type[l]=="Gaussian":
                    for k in range(self.K[l]):
                        entropies_n= entropies_n + 0.5*numpy.log( 2*numpy.pi*numpy.e/(-2*self.b[l][1]))
                else:
                    print("The entropy for {0} distrubution is not implemented yet.".format(self.hidden_type[l]))
                    entropies_n= entropies_n + 0
            entropies= entropies + entropies_n
        mean_entropy=entropies/num_samples
        #print "The mean entropy is {}".format(mean_entropy)
        return mean_entropy,entropies
        

    def sample_h_given_b_hat(self, b_hat=None, hidden_type="Bernoulli", hidden_type_fixed_param=1, hidden_value_or_meanfield="value"):
        """
        In a shallow set.
        """
        #H=X # initialize H
        #num=X.shape[1]
        #for n in range(num):
        #    b=numpy.copy(self.b)
        #    b.shape=(self.K,)
        #    h_prob=cl.sigmoid(b + numpy.dot(self.W.transpose(),X[:,n])) 
        #    h=numpy.zeros(shape=(self.K,),dtype=int)
        #    for k in range(self.K):
        #        h[k]=self.rng.binomial(n=1,h_prob[k],size=1)
        #    H[:,n]=h
            
        # sampling
        if hidden_type=="Bernoulli":
            if isinstance(b_hat, list):
                b_hat=b_hat[0] # peel off
            b_hat[b_hat<-200]=-200
            P=cl.sigmoid(b_hat)
            HM=P # mean of hidden variables
            if hidden_value_or_meanfield=="value":
                H=cl.Bernoulli_sampling(P,rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif hidden_type=="Binomial":
            if isinstance(b_hat, list):
                b_hat=b_hat[0] # peel off
            P=cl.sigmoid(b_hat) # probability
            HM=hidden_type_fixed_param*P # mean
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Binomial_sampling(hidden_type_fixed_param, P, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif hidden_type=="Gaussian":
            HM=-b_hat[0]/(2*b_hat[1])
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Gaussian_sampling(HM, -2*b_hat[1], rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif hidden_type=="Gaussian2":
            pass
        elif hidden_type=="Multinomial":
            if isinstance(b_hat, list):
                b_hat=b_hat[0] # peel off
            P=numpy.exp(b_hat) # probability
            P=cl.normalize_probability(P)
            HM=hidden_type_fixed_param*P # mean
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Multinomial_sampling(hidden_type_fixed_param, P=P, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
                
        return H,HM


    def sample_visible(self, visible_type="Bernoulli", a=None, W=None, H=None, visible_type_fixed_param=10, tie_W_for_pretraining_DBM_top=False, NS=None, rng=numpy.random.RandomState(100)):
        if H is not None:
            if visible_type=="Bernoulli":
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                P=cl.sigmoid( a_hat )
                XM=P
                X=cl.Bernoulli_sampling(P,rng=rng)
            elif visible_type=="Gaussian":
                a1=a[0]
                a2=a[1]
                if tie_W_for_pretraining_DBM_top:
                    a_hat1=a1 + 2*numpy.dot(W,H) # mean
                else:
                    a_hat1=a1 + numpy.dot(W,H) 
                a_hat2=a2
                XM=-a_hat1/(2*a_hat2)
                P=None
                X=cl.Gaussian_sampling(XM,-2*a_hat2,rng=rng)
            elif visible_type=="Gaussian_FixPrecision1":
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                XM=a_hat
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_FixPrecision2":
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                XM=a_hat/visible_type_fixed_param
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_Hinton":
                a1=a[0]
                a2=a[1]
                if tie_W_for_pretraining_DBM_top:
                    a_hat1=a1 + 2/a2*numpy.dot(W,H) # mean
                else:
                    a_hat1=a1 + 1/a2*numpy.dot(W,H) 
                XM=a_hat1
                P=None
                X=cl.Gaussian_sampling(a_hat1,a2,rng=rng)
            elif visible_type=="Poisson": 
                tol_poisson_max=self.tol_poisson_max
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                a_hat[a_hat>tol_poisson_max]=tol_poisson_max
                XM=numpy.exp(a_hat)
                P=None
                X=cl.Poisson_sampling(XM,rng=rng)
            elif visible_type=="NegativeBinomial": 
                if tie_W_for_pretraining_DBM_top:
                    a_hat= a + 2*numpy.dot(W,H) 
                else:
                    a_hat= a + numpy.dot(W,H) # a_hat should be negative
                tol_negbin_max=-1e-8
                tol_negbin_min=-100
                a_hat[a_hat>=0]=tol_negbin_max
                a_hat[a_hat<tol_negbin_min]=tol_negbin_min
                P_failure=numpy.exp(a_hat)
                P=P_failure
                P_success=1-P_failure
                #print "max: {}".format(numpy.max(P_failure))
                #print "min: {}".format(numpy.min(P_failure))
                XM=visible_type_fixed_param*(P_failure/P_success)
                X=cl.NegativeBinomial_sampling(K=visible_type_fixed_param,P=P_success,rng=rng)
            elif visible_type=="Multinomial":
                if tie_W_for_pretraining_DBM_top:
                    a_hat= a + 2*numpy.dot(W,H) 
                else:
                    a_hat= a + numpy.dot(W,H)
                P=numpy.exp(a_hat)
                P=cl.normalize_probability(P)
                #print "max: {}".format(numpy.max(P))
                #print "min: {}".format(numpy.min(P))
                XM=visible_type_fixed_param*(P)
                X=cl.Multinomial_sampling(N=visible_type_fixed_param,P=P,rng=rng)
            elif visible_type=="Multinoulli":
                P=[None]*self.M
                XM=[None]*self.M
                X=[None]*self.M
                for m in range(self.M):
                    if tie_W_for_pretraining_DBM_top:
                        a_hat= a[m] + 2*numpy.dot(W[m],H) 
                    else:
                        a_hat= a[m] + numpy.dot(W[m],H)
                    P[m]=numpy.exp(a_hat)
                    P[m]=cl.normalize_probability(P[m])
                    #print "max: {}".format(numpy.max(P))
                    #print "min: {}".format(numpy.min(P))
                    XM[m]=P[m]
                    X[m]=cl.Multinomial_sampling(N=1,P=P[m],rng=rng)
            elif visible_type=="Gamma":
                a1=a[0]
                a2=a[1]
                a_hat1=a1
                if tie_W_for_pretraining_DBM_top:
                    a_hat2=a2 + 2*numpy.dot(W,H) # mean
                else:
                    a_hat2=a2 + numpy.dot(W,H)
                P=None
                X=cl.Gamma_sampling(a_hat1+1, -a_hat2,rng=rng)
            else:
                print("Please choose a correct visible type!")
                
        if H is None: # randomly generate some visible samples without using H and W
            if NS is None:
                NS=1
            if visible_type=="Bernoulli":
                P=cl.sigmoid(numpy.repeat(a,NS,axis=1))
                XM=P
                X=cl.Bernoulli_sampling(P, rng=rng)
            elif visible_type=="Gaussian":
                a1=a[0]
                a2=a[1]
                XM=-numpy.repeat(a1,NS,axis=1)/(2*numpy.repeat(a2,NS,axis=1))
                P=None
                X=cl.Gaussian_sampling(XM, -2*a2, rng=rng)
            elif visible_type=="Gaussian_FixPrecision1":
                XM=a
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_FixPrecision2":
                XM=a/visible_type_fixed_param
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_Hinton":
                a1=a[0]
                a2=a[1]
                XM=numpy.repeat(a1,NS,axis=1)
                P=None
                X=cl.Gaussian_sampling(XM, a2, rng=rng)
            elif visible_type=="Poisson":
                tol_poisson_max=self.tol_poisson_max
                a[a>tol_poisson_max]=tol_poisson_max
                XM=numpy.exp(numpy.repeat(a,NS,axis=1))
                P=None
                X=cl.Poisson_sampling(XM, rng=rng)
            elif visible_type=="NegativeBinomial":
                P_failure=numpy.exp(numpy.repeat(a,NS,axis=1))
                P=P_failure
                P_success=1-P_failure
                XM=visible_type_fixed_param * (1-P_success)/P_success
                X=cl.NegativeBinomial_sampling(visible_type_fixed_param, P_success, rng=rng)
            elif visible_type=="Multinomial":
                p_normalized=cl.normalize_probability(numpy.exp(a))
                P=numpy.repeat(p_normalized,NS,axis=1) # DO I NEED TO NORMALIZE IT?
                XM=visible_type_fixed_param * P
                X=cl.Multinomial_sampling(N=visible_type_fixed_param,P=P,rng=rng)
            elif visible_type=="Multinoulli":
                P=[None]*self.M
                XM=[None]*self.M
                X=[None]*self.M
                for m in range(self.M):
                    p_normalized=cl.normalize_probability(numpy.exp(a[m]))
                    P[m]=numpy.repeat(p_normalized,NS,axis=1) # DO I NEED TO NORMALIZE IT?
                XM[m]=P[m]
                X[m]=cl.Multinomial_sampling(N=1,P=P[m],rng=rng)
            elif visible_type=="Gamma":
                a1=a[0]
                a2=a[1]
                a2_rep=numpy.repeat(a2,NS,axis=1)
                P=None
                X=cl.Gamma_sampling(a1+1, -a2_rep,rng=rng)
            else:
                print("Please choose a correct visible type!")
                
        return X,XM,P


    def update_crbms(self):
        """ 
        Update the parameters of separate CRBMs.
        """
        
        for l in range(self.L):
            if l==0: # first CRBM
                a=self.a
                b=self.b[l]
                c=self.c[l]
                W=self.W[l]
                self.crbms[l].set_param(a,b,c,W)
            elif l==self.L-1 and self.L>1: # last CRBM
                a=self.b[l-1]
                b=self.b[l]
                d=self.c[l-1]
                c=self.c[l]
                W=self.W[l]
                self.crbms[l].set_param(a,b,c,d,W)
            else: # CRBMs in the middle
                a=self.b[l-1]
                b=self.b[l]
                d=self.c[l-1]
                c=self.c[l]
                W=2*self.W[l]
                self.crbms[l].set_param(a,b,c,d,W)


    def get_param(self):
        return self.a,self.b,self.c,self.W,self.br,self.cr,self.Wr


    def set_param(self, a=None, b=None, c=None, W=None, br=None, cr=None, Wr=None, update_crbms=True):
        """
        param is a dict type.
        """
        if a is not None:
            self.a=a
        if b is not None:
            self.b=b
        if c is not None:
            self.c=c
        if W is not None:
            self.W=W
        if br is not None:
            self.br=br
        if cr is not None:
            self.cr=cr
        if Wr is not None:
            self.Wr=Wr
        if update_crbms:
            self.update_crbms()
        
        
    def make_dir_save(self,parent_dir_save, prefix, learn_rate_a, learn_rate_b, learn_rate_c, learn_rate_W, maxiter=None, normalization_method="None"):
        
        print("start making dir...")
        # different layers can have different learning rates         
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*self.L
        if numpy.isscalar(learn_rate_c):
            learn_rate_c=[learn_rate_c]*self.L
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*self.L
            
            
#        if self.visible_type=="Gaussian" or self.visible_type=="Gamma": 
#            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a[0]) + "_" + str(learn_rate_a[1]) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_visfix:" + str(self.visible_type_fixed_param) + "_hidfix:" + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
#        elif self.visible_type=="Multinoulli":
#            foldername=prefix + "_X"+self.visible_type+":" + str(len(self.M)) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_visfix:" + str(self.visible_type_fixed_param) + "_hidfix:" + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
#        else:
#            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_visfix:" + str(self.visible_type_fixed_param[0]) + "_hidfix:" + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
            
        if self.visible_type=="Gaussian" or self.visible_type=="Gaussian2" or self.visible_type=="Gamma": 
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
            #foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a[0]) + "_" + str(learn_rate_a[1]) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_c,dtype=str)) + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        else:
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
            #foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabcW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_c,dtype=str)) + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        
        dir_save=parent_dir_save+foldername
        self.dir_save=dir_save
            
        try:
            os.makedirs(dir_save)
        except OSError:
            #self.dir_save=parent_dir_save
            pass
        print("The results will be saved in " + self.dir_save)
        return self.dir_save


    def save_sampling(self, XM, ifsort=True, dir_save="./", prefix="CHM"):
        """
        Save the sampling results for bag of word data.
        """
        if ifsort:
            num_features=XM.shape[0]
            num_samples=XM.shape[1]
            XM_sorted=numpy.zeros(shape=(num_features,num_samples),dtype=float)
            features_sorted=numpy.zeros(shape=(num_features,num_samples),dtype=object)
            for n in range(num_samples):
                x=XM[:,n]
                ind=numpy.argsort(x,kind="mergesort")
                ind=ind[::-1]
                XM_sorted[:,n]=x[ind]
                features_sorted[:,n]=self.features[ind]
                
            filename=dir_save + prefix + "_sampled_XM_sorted.txt"
            numpy.savetxt(filename,XM_sorted, fmt="%.2f", delimiter="\t")
            filename=dir_save + prefix + "_sampled_features_sorted.txt"
            numpy.savetxt(filename,features_sorted, fmt="%s", delimiter="\t")
        else:
            filename=dir_save + prefix + "_sampled_XM.txt"
            numpy.savetxt(filename,XM, fmt="%.2f", delimiter="\t")
            filename=dir_save + prefix + "_features.txt"
            numpy.savetxt(filename,self.features, fmt="%s", delimiter="\t")
