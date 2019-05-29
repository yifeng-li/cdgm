# test Exp-CRBM on MNIST
#from __future__ import division
import pickle, gzip
import numpy
import capsule_rbm
import classification as cl
import shutil
import os

workdir="/home/yifeng/research/deep/github/cdgm/"
os.chdir(workdir)

dir_data="./data/MNIST/"

parent_dir_save="./results/CRBM/"
prefix="CRBM_MNIST"

# Load the dataset
f = gzip.open(dir_data+"mnist.pkl.gz", "rb")
train_set_x, train_set_y, test_set_x, test_set_y = pickle.load(f, fix_imports=True, encoding="latin1", errors="strict")
f.close()

# train_set_x is a list of 28X28 matrices
train_set_x=numpy.array(train_set_x,dtype=int)
num_train_samples,num_rows,num_cols=train_set_x.shape
train_set_x=numpy.reshape(train_set_x,newshape=(num_train_samples,num_rows*num_cols))
train_set_y=numpy.array(train_set_y,dtype=int)

test_set_x=numpy.array(test_set_x,dtype=int)
num_test_samples,num_rows,num_cols=test_set_x.shape
test_set_x=numpy.reshape(test_set_x,newshape=(num_test_samples,num_rows*num_cols))
test_set_y=numpy.array(test_set_y,dtype=int)

train_set_x=train_set_x.transpose()
test_set_x=test_set_x.transpose()

print(train_set_x.shape)
print(train_set_y.shape)
print(test_set_x.shape)
print(test_set_y.shape)

num_train=train_set_x.shape[1]
num_test=test_set_x.shape[1]
num_cl=len(numpy.unique(train_set_y))

# convert train_set_y to binary codes
train_set_y01,z_unique=cl.membership_vector_to_indicator_matrix(z=train_set_y, z_unique=range(num_cl))
train_set_y01=train_set_y01.transpose()
test_set_y01,_=cl.membership_vector_to_indicator_matrix(z=test_set_y, z_unique=range(num_cl))
test_set_y01=test_set_y01.transpose()


num_feat=train_set_x.shape[0]
visible_type="Bernoulli"
hidden_type="Bernoulli"
rng=numpy.random.RandomState(100)
M=num_feat
normalization_method="None"

if visible_type=="Bernoulli":
    normalization_method="scale"
    # parameter setting
    learn_rate_a=0.01
    learn_rate_c=0.01
    learn_rate_W=0.01
    change_rate=0.95
    adjust_change_rate_at=[3600,6000]#[6000,9600]
    adjust_coef=1.02
    
    batch_size=100
    NMF=10
    increase_NMF_at=[3600]
    increased_NMF=[20]
    pcdk=20
    NS=100
    maxiter=12000
    change_every_many_iters=120
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True
    xz_interaction=False
    
elif visible_type=="Gaussian":
    normalization_method="scale"
    # parameter setting
    learn_rate_a=[0.001,0.001]
    learn_rate_c=0.001
    learn_rate_W=0.001
    change_rate=0.85
    adjust_change_rate_at=[1200,2400,6000,12000,18000]
    adjust_coef=1.05
    
    batch_size=100
    NMF=10
    increase_NMF_at=[3600]
    increased_NMF=[20]
    pcdk=10
    NS=100
    maxiter=96000
    change_every_many_iters=240
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True
    xz_interaction=True

elif visible_type=="Gaussian2":
    normalization_method="scale"
    # parameter setting
    learn_rate_a=[0.001,0.001]
    learn_rate_c=0.001
    learn_rate_W=0.001
    change_rate=0.85
    adjust_change_rate_at=[1200,2400,6000,12000,18000]
    adjust_coef=1.05
    
    batch_size=100
    NMF=10
    increase_NMF_at=[3600]
    increased_NMF=[20]
    pcdk=10
    NS=100
    maxiter=96000
    change_every_many_iters=240
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True
    xz_interaction=True
   
if hidden_type=="Bernoulli":
    J=16
    K=40
    learn_rate_b=0.01
    hidden_type_fixed_param=0

elif hidden_type=="Gaussian":
    J=16
    K=20
    learn_rate_b=[0.001,0.001]
    hidden_type_fixed_param=0

    
# normalization method
if normalization_method=="binary":
    # discret data
    threshold=0
    ind=train_set_x<=threshold
    train_set_x[ind]=0
    train_set_x[numpy.logical_not(ind)]=1
    ind=test_set_x<=threshold
    test_set_x[ind]=0
    test_set_x[numpy.logical_not(ind)]=1

if normalization_method=="logtransform":
    train_set_x=numpy.log2(train_set_x+1)
    test_set_x=numpy.log2(test_set_x+1)

if normalization_method=="scale":
    train_set_x=train_set_x/255
    test_set_x=test_set_x/255

# initialize a model
model_crbm=capsule_rbm.capsule_rbm(M=M,K=K, J=J, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type, hidden_type_fixed_param=hidden_type_fixed_param, tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=False, a=None, xz_interaction=xz_interaction, rng=rng)
# create a folder to save the results
dir_save=model_crbm.make_dir_save(parent_dir_save, prefix, learn_rate_a, learn_rate_b, learn_rate_c, learn_rate_W, visible_type_fixed_param, hidden_type_fixed_param, maxiter, normalization_method)
# make a copy of this script in dir_save
shutil.copy(workdir+"main_test_ExpCRBM_MNIST.py", dir_save)
shutil.copy(workdir+"capsule_rbm.py", dir_save)

# train CRBM
model_crbm.train(X=train_set_x, X_validate=test_set_x, batch_size=batch_size, NMF=NMF, increase_NMF_at=increase_NMF_at, increased_NMF=increased_NMF, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=learn_rate_a, learn_rate_b=learn_rate_b, learn_rate_c=learn_rate_c, learn_rate_W=learn_rate_W, change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=None, valid_subset_size_for_compute_error=batch_size, track_reconstruct_error=True, track_free_energy=False, reinit_a_use_data_stat=reinit_a_use_data_stat, if_plot_error_free_energy=True, dir_save=dir_save, prefix=prefix, figwidth=5, figheight=3)

# sampling using Gibbs sampling
_,_,_=model_crbm.generate_samples(NS=100, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, dir_save=dir_save, prefix=prefix)

# get z code for 100 training samples
num_sample_per_cl=10
train_set_x100,train_set_y100,_,_=cl.truncate_sample_size(train_set_x.transpose(),train_set_y,max_size_given=num_sample_per_cl)
train_set_x100=train_set_x100.transpose()
sample_set_x_3way=numpy.reshape(train_set_x100,newshape=(28,28,100))
cl.plot_image_subplots(dir_save+"fig_"+prefix+"_actual_samples.pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(num_sample_per_cl*num_cl))), wspace=0.01, hspace=0.001)
ZM,ZM_sorted,_=model_crbm.infer_z_given_x(train_set_x_sub=train_set_x100, NMF=1000, dir_save=dir_save, prefix=prefix+"_100")

## given z code, to generate samples
num_cl=10
model_crbm.generate_samples_given_z(Z0=ZM, NS=num_sample_per_cl*num_cl, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, dir_save=dir_save, prefix=prefix+"_given_100ZM")


# given x, infer z and h, then generate image for each active capsule
train_set_x100_split=numpy.hsplit(train_set_x100, num_cl )
for c in range(num_cl):
    _,_=model_crbm.per_capsule_pattern_given_x(train_set_x_sub=train_set_x100_split[c], NMF=1000, threshold_z=0, row=28, col=28, figwidth=24, figheight=6, dir_save=dir_save, prefix=prefix+"_class_"+str(c)+"_per_capsule_thresholdz_"+str(0))

# a different threshold
for c in range(num_cl):
    _,_=model_crbm.per_capsule_pattern_given_x(train_set_x_sub=train_set_x100_split[c], NMF=1000, threshold_z=0.1, row=28, col=28, figwidth=24, figheight=6, dir_save=dir_save, prefix=prefix+"_class_"+str(c)+"_per_capsule_thresholdz_"+str(0.1))

# generate 100 samples per class
for s in range(num_cl):
    Z0=numpy.tile(ZM[:,s*num_sample_per_cl:(s+1)*num_sample_per_cl], num_sample_per_cl)
    model_crbm.generate_samples_given_z(Z0=Z0, NS=num_sample_per_cl*num_cl, sampling_time=1, reinit=False, pcdk=1000, rand_init=True, dir_save=dir_save, prefix=prefix+"_"+str(s))


# only activate one capsule each time
num_samples_per_capsule=40
Z=numpy.zeros((K,num_samples_per_capsule))
Z[0,:]=1
for k in range(1,K):
    Zk=numpy.zeros((K,num_samples_per_capsule))
    Zk[k,:]=1
    Z=numpy.hstack((Z,Zk))
model_crbm.generate_samples_given_z(Z0=Z, NS=K*num_samples_per_capsule, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, num_col=num_samples_per_capsule, figwidth=24, figheight=24*K/num_samples_per_capsule, dir_save=dir_save, prefix=prefix+"_only_one_capsule_active")


# only one sample per capsule
num_samples_per_capsule=1
Z=numpy.zeros((K,num_samples_per_capsule))
Z[0,:]=1
for k in range(1,K):
    Zk=numpy.zeros((K,num_samples_per_capsule))
    Zk[k,:]=1
    Z=numpy.hstack((Z,Zk))
model_crbm.generate_samples_given_z(Z0=Z, NS=K*num_samples_per_capsule, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, num_col=10, figwidth=5, figheight=2, dir_save=dir_save, prefix=prefix+"_only_one_capsule_active_")


print("result saved in: " + dir_save)



