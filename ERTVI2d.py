import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import scipy.sparse as sparse

from VIP.vip.prior.transform import trans
from forward.ERT2D.ERT2D import ERT2D
from VIP.vip.pyvi.svgd import SVGD, sSVGD
from VIP.vip.pyvi.advi import ADVI

from datetime import datetime
import time
import configparser
from VIP.vip.prior.prior import prior
from VIP.vip.prior.pdf import Uniform, Gaussian

import argparse
import sys
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
os.environ['OMP_NUM_THREADS']='30'

def init_tomo(config):
    Path(config.get('svgd','outpath')).mkdir(parents=True, exist_ok=True)


def generate_init(n=1,lb=0,ub=1e8, transform=True):
    eps = np.finfo(np.float64).eps
    x = lb + (ub-lb)*np.random.uniform(low=eps,high=1-eps,size=(n,lb.shape[0]))

    if(transform):
        x = trans(x.reshape((-1,lb.shape[0])),lb=lb, ub=ub, trans=1)
        
    return x


def get_init(config, resume=0):
    initfile = os.path.join(Path(config.get('svgd','inputpath')),config['svgd']['init'])
    prior = np.loadtxt(initfile)
    lower_bnd = prior[:,0].astype(np.float64); 
    upper_bnd = prior[:,1].astype(np.float64)
 
    x0 = generate_init(n=config.getint('svgd','nparticles'), lb=lower_bnd, ub=upper_bnd,
        transform=config.getboolean('svgd','transform'))

    if(np.isinf(x0).any()): print("Error: inf occured")
    if(np.isnan(x0).any()): print("Error: nan occured")

    return x0


def create_prior(config):
    
    priorfile = os.path.join(Path(config.get('svgd','inputpath')),config['svgd']['prior'])
    
    p = np.loadtxt(priorfile)
    p1 = p[:,0].astype(np.float64); p2 = p[:,1].astype(np.float64)
    p1 = p1.flatten()
    p2 = p2.flatten()
    ptype = config.get('svgd','priortype')
    if(ptype == 'Uniform'):
        pdf = Uniform(lb=p1, ub=p2)
    if(ptype == 'Gaussian'):
        pdf = Gaussian(mu=p1, sigma=p2)

    smoothness = False
    L = None

    if(ptype=='Uniform'):
        ppdf = prior(pdf=pdf, transform=config.getboolean('svgd','transform'), lb=p1, ub=p2, smooth=smoothness, L=L)
    else:
        ppdf = prior(pdf=pdf, transform=config.getboolean('svgd','transform'), smooth=smoothness, L=L)

    return ppdf


def write_samples(filename, pprior, n=0, chunk=10):

    f = h5py.File(filename,'r+')
    samples = f['samples']
    start = 0
    if(n>0):
        start = samples.shape[0] - n
    if(start<0):
        start = 0
    if(pprior.trans):
        for i in range(start,samples.shape[0]):
            samples[i,:,:] = pprior.adjust(samples[i,:,:])

    # mean = np.mean(samples[:].reshape((-1,samples.shape[2])),axis=0)
    # std = np.std(samples[:].reshape((-1,samples.shape[2])),axis=0)
    # last = samples[-1,:,:]
    f.close()

    # np.save('mean.npy',mean)
    # np.save('std.npy',std)
    # np.save('last_sample.npy',last)
    
    with h5py.File('./results/samples.hdf5', 'r') as f:
        print("Keys in the HDF5 file:", list(f.keys()))
        for key in f.keys():
            print(f"Group/Dataset name: {key}")
            if isinstance(f[key], h5py.Dataset):
                print(f"Shape of {key}: {f[key].shape}")
                print(f"Datatype of {key}: {f[key].dtype}")
        data = f['samples'][:]
    
    samples=data[:,:,:]
    samples=-np.log10(np.exp(samples))
    np.savetxt(os.path.join(config.get('svgd','outpath'),'samples.txt'),samples[-1,:,:])
    
    samples_mean = np.mean(samples[:].reshape((-1,samples.shape[2])),axis=0)
    samples_std = np.std(samples[:].reshape((-1,samples.shape[2])),axis=0)
    output=np.column_stack((samples_mean, samples_std))
    np.savetxt(os.path.join(config.get('svgd','outpath'),'samplemeanstd.txt'),output)
    
    return 0


def clear_file():
    os.remove('data_output.bin')
    os.remove('data_sigma.bin')
    os.remove('lossgrad.bin')


if __name__=="__main__":
    # SVGD
    parser = argparse.ArgumentParser(description='Variational Tomography')
    parser.add_argument("-c", "--config", metavar='config', default='config.ini', help="Configuration file")
    parser.add_argument("-r", "--resume", metavar='resume', default=0, type=float, help="Resume mode (1) or start a new run(0)")

    args = parser.parse_args()
    configfile = args.config
    resume = args.resume

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'Start VTomo at {current_time}...')
    print(f'Config file for VFWI is: {configfile}')

    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(configfile)
    np.random.seed(config.getint('svgd','seeds'))
    
    print('Method for ERT is: ' + config['svgd']['method'])
    init_tomo(config)
    x0 = get_init(config, resume=resume)
    print("Particles size: " + str(x0.shape))

    ppdf = create_prior(config)
    
    # ERT
    data = np.loadtxt(os.path.join(config['svgd']['inputpath'],config['ERT']['obsfilename']))
    print('data:')
    print(data)
    
    weight_a = config.getfloat('ERT','a')
    weight_b = config.getfloat('ERT','b')
    print('a:', weight_a)
    print('b:', weight_b)
    
    data_sigma = np.sqrt(weight_a*weight_a+weight_b*weight_b*data*data) * config.getfloat('ERT','scalingfactor')
    print('data_sigma: ')
    print(data_sigma)
    
    data_output = np.append(data.shape*np.ones(1),data)
    data_output.tofile("data_output.bin")
    data_sigma.tofile("data_sigma.bin")
  
    simulator = ERT2D(config, ppdf, data_sigma, data)

    # svgd sampler
    stepsize = config.getfloat('svgd','stepsize')
    iteration = config.getint('svgd','iter')
    final_decay = config.getfloat('svgd','final_decay')
    gamma = final_decay**(1./iteration)
    if(config['svgd']['method']=='ssvgd'):
        svgd = sSVGD(simulator.dlnprob,
                     kernel=config['svgd']['kernel'],
                     weight=config['svgd']['diag'],
                     out=os.path.join(config.get('svgd','outpath'),'samples.hdf5'))
    elif(config['svgd']['method']=='svgd'):
        svgd = SVGD(simulator.dlnprob,
                    kernel=config['svgd']['kernel'],
                    weight=config['svgd']['diag'],
                    # threshold=0.02,
                    # h=1.0,
                    out=os.path.join(config.get('svgd','outpath'),'samples.hdf5'))
    else:
        print('Not supported method')

    
    print('Start sampling ...')
    print(f'Iteration: {iteration}')
    print(f'Stepsize, decay rate and final decay: {stepsize} {gamma} {final_decay}')
    start = time.time()
    losses, x = svgd.sample(x0,
                    n_iter=config.getint('svgd','iter'),
                    stepsize=stepsize, gamma=gamma,
                    # alpha= 0.5,
                    # metropolis=True,
                    optimizer=config['svgd']['optimizer'],
                    burn_in=config.getint('svgd','burn_in'),
                    thin=config.getint('svgd','thin')
                    )
    end=time.time()
    print('Time taken: '+str(end-start)+' s')

    # write out results
    nsamples = int((config.getint('svgd','iter')-config.getint('svgd','burn_in'))/config.getint('svgd','thin'))
    
    write_samples(os.path.join(config.get('svgd','outpath'),'samples.hdf5'), ppdf, n=nsamples)
    
    with open(os.path.join(config.get('svgd','outpath'),'misfits.txt'),"a") as f:
        np.savetxt(f,losses)
    
    clear_file()
