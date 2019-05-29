# test Exp-Capsule HM on MNIST
#import pickle, gzip
import numpy
import capsule_hm
import classification as cl
import shutil
import os

workdir="/home/yifeng/research/deep/github/cdgm/"
os.chdir(workdir)

dir_data="./data/FASHIONMNIST/"

parent_dir_save="./results/CHM/"
prefix="CHM_FASHIONMNIST"
# load data
train_set_x=numpy.loadtxt(dir_data+"fashion-mnist_train.csv", dtype=int, delimiter=",",skiprows=1)
train_set_y=train_set_x[:,0]
train_set_x=train_set_x[:,1:]
train_set_x=train_set_x.transpose()

test_set_x=numpy.loadtxt(dir_data+"fashion-mnist_test.csv", dtype=int, delimiter=",",skiprows=1)
test_set_y=test_set_x[:,0]
test_set_x=test_set_x[:,1:]
test_set_x=test_set_x.transpose()

print(train_set_x.shape)
print(train_set_y.shape)
print(test_set_x.shape)
print(test_set_y.shape)




# limit the number of training set
#train_set_x=train_set_x[:,0:10000]
#train_set_y=train_set_y[0:10000]
#test_set_x=test_set_x[:,0:10000]
#test_set_y=test_set_y[0:10000]

num_train=train_set_x.shape[1]
num_test=test_set_x.shape[1]
num_cl=len(numpy.unique(train_set_y))

# convert train_set_y to binary codes
train_set_y01,z_unique=cl.membership_vector_to_indicator_matrix(z=train_set_y, z_unique=range(num_cl))
train_set_y01=train_set_y01.transpose()
test_set_y01,_=cl.membership_vector_to_indicator_matrix(z=test_set_y, z_unique=range(num_cl))
test_set_y01=test_set_y01.transpose()


num_feat=train_set_x.shape[0]
visible_type="Gaussian"
hidden_type="Gaussian"
hidden_type_fixed_param=0
rng=numpy.random.RandomState(100)
M=num_feat
normalization_method="None"

if visible_type=="Bernoulli":  
    normalization_method="scale" 
    # parameter setting
    learn_rate_a_pretrain=0.02
    learn_rate_c_pretrain=[0.02,0.02,0.02]
    learn_rate_W_pretrain=[0.02,0.02,0.02]
    change_rate_pretrain=0.95
    adjust_change_rate_at_pretrain=[6000]
    adjust_coef_pretrain=1.02
    maxiter_pretrain=6000
    change_every_many_iters_pretrain=120

    learn_rate_a_train=0.01
    learn_rate_c_train=[0.01,0.01,0.01]
    learn_rate_W_train=[0.01,0.01,0.01]
    change_rate_train=0.95
    adjust_change_rate_at_train=[12000]#[6000,9600]#[6000,9600,12000]
    adjust_coef_train=1.05
    maxiter_train=6000
    change_every_many_iters_train=120

    batch_size=100
    NMF=20
    increase_NMF_at=[3600]
    increased_NMF=[20]
    pcdk=20
    NS=100
    
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True#True
    xz_interaction=False

elif visible_type=="Gaussian":
    
    normalization_method="scale" 
    # parameter setting
    rate_pretrain=0.0005
    learn_rate_a_pretrain=[rate_pretrain,rate_pretrain]
    learn_rate_c_pretrain=[rate_pretrain,rate_pretrain,rate_pretrain]
    learn_rate_W_pretrain=[rate_pretrain,rate_pretrain,rate_pretrain]
    change_rate_pretrain=0.95
    adjust_change_rate_at_pretrain=[4800,12000]#[1200,2400,3600,12000,18000]#[4800,12000]
    adjust_coef_pretrain=1.02
    maxiter_pretrain=12000#15000
    change_every_many_iters_pretrain=240

    rate=0.0005
    learn_rate_a_train=[rate,rate]
    learn_rate_c_train=[rate,rate,rate]
    learn_rate_W_train=[rate,rate,rate]
    change_rate_train=0.95
    adjust_change_rate_at_train=[4800,12000]#[1200,2400,3600,12000,18000]#[4800,12000]#[6000,9600,12000]
    adjust_coef_train=1.02
    maxiter_train=24000#15000
    change_every_many_iters_train=240

    batch_size=100
    NMF=10#20#20
    increase_NMF_at=[3600]
    increased_NMF=[20]
    pcdk=20#20#20
    NS=100
    
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True
    xz_interaction=False

# Hyper-parameters for hidden layers
if hidden_type=="Bernoulli":
    J=16
    K=[40,40]
    learn_rate_b_pretrain=[0.02,0.02,0.02]
    learn_rate_b_train=[0.01,0.01,0.01]
    hidden_type_fixed_param=0

elif hidden_type=="Gaussian":
    J=16
    K=[20,20]#[40,40,40]
    learn_rate_b_pretrain=[[rate_pretrain,rate_pretrain],[rate_pretrain,rate_pretrain],[rate_pretrain,rate_pretrain]]
    learn_rate_b_train=[[rate,rate],[rate,rate],[rate,rate]]
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
    

# creat the object
model_chm=capsule_hm.capsule_hm(features=None, M=M, K=K, J=J, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type, hidden_type_fixed_param=hidden_type_fixed_param, tol_poisson_max=8, rec_error_max=0.5, xz_interaction=xz_interaction, rng=rng)
# create a folder to save the results
dir_save=model_chm.make_dir_save(parent_dir_save, prefix, learn_rate_a_pretrain, learn_rate_b_pretrain, learn_rate_W_pretrain, maxiter_pretrain+maxiter_train, normalization_method)
# make a copy of this script in dir_save
shutil.copy(workdir+"main_test_ExpCHM_FASHIONMNIST.py", dir_save)
shutil.copy(workdir+"capsule_rbm3.py", dir_save)
shutil.copy(workdir+"capsule_rbm23.py", dir_save)
shutil.copy(workdir+"capsule_hm.py", dir_save)
shutil.copy(workdir+"classification.py", dir_save)


# pretrain
model_chm.pretrain(X=train_set_x, X_validate=test_set_x, batch_size=batch_size, NMF=NMF, pcdk=pcdk, increase_NMF_at=increase_NMF_at,increased_NMF=increased_NMF, NS=NS, maxiter=maxiter_pretrain, learn_rate_a=learn_rate_a_pretrain, learn_rate_b=learn_rate_b_pretrain, learn_rate_c=learn_rate_c_pretrain, learn_rate_W=learn_rate_W_pretrain, change_rate=change_rate_pretrain, adjust_change_rate_at=adjust_change_rate_at_pretrain, adjust_coef=adjust_coef_pretrain, change_every_many_iters=change_every_many_iters_pretrain, init_chain_time=init_chain_time, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=False, reinit_a_use_data_stat=reinit_a_use_data_stat, if_plot_error_free_energy=True, dir_save=dir_save, prefix="CHM_pretrain", figwidth=5, figheight=3)

# sampling
#_,_=model_chm.generate_samples(NS=100, NMF=100, method="ancestral", sampling_time=4, sampling_num_iter=100, row=28, col=28, dir_save=dir_save, prefix="pretrain_"+prefix)

# train
model_chm.train(X=train_set_x, X_validate=test_set_x, batch_size=batch_size, NMF=NMF, increase_NMF_at=increase_NMF_at, increased_NMF=increased_NMF, maxiter=maxiter_train, learn_rate_a=learn_rate_a_train, learn_rate_b=learn_rate_b_train, learn_rate_c=learn_rate_c_train, learn_rate_W=learn_rate_W_train, change_rate=change_rate_train, adjust_change_rate_at=adjust_change_rate_at_train, adjust_coef=adjust_coef_train, change_every_many_iters=change_every_many_iters_train, init_chain_time=init_chain_time, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=False, if_plot_error_free_energy=True, dir_save=dir_save, prefix="CHM_train", figwidth=5, figheight=3)

# sampling

num_sample_per_cl=10
train_set_x100,train_set_y100,_,_=cl.truncate_sample_size(train_set_x.transpose(),train_set_y,max_size_given=num_sample_per_cl)
train_set_x100=train_set_x100.transpose()
sample_set_x_3way=numpy.reshape(train_set_x100,newshape=(28,28,100))
cl.plot_image_subplots(dir_save+"fig_"+prefix+"_actual_samples.pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(num_sample_per_cl*num_cl))), wspace=0.01, hspace=0.001)
ZM,_=model_chm.infer_hidden_states_given_x(train_set_x_sub=train_set_x100, NMF=100, dir_save=dir_save, prefix=prefix)

_,_=model_chm.generate_samples(NS=100, NMF=100, method="ancestral", sampling_time=4, sampling_num_iter=1000, row=28, col=28, dir_save=dir_save, prefix="finetune_"+prefix)

_,_=model_chm.generate_samples(NS=100, NMF=100, method="Gibbs", sampling_time=4, sampling_num_iter=1000, row=28, col=28, dir_save=dir_save, prefix="finetune_"+prefix)
