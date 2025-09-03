import numpy as np
import time
import subprocess
import os
import math

class ERT2D():
    '''
    A class that implements an interface of an external 2D FWI code
    '''
    
    def __init__(self, config, prior ,sigma, data, mask=None, client=None):
        '''
        config: a python configparser.ConfigParser()
        prior: a prior class, see prior/prior.py
        mask: a mask array where the parameters will be fixed, default no mask
        client: a dask client to submit fwi running, must be specified
        '''

        self.config = config
        # self.sigma = sigma
        self.client = client
        self.prior = prior
        self.data = data
        self.isstop = False
        
        
        self.targetRMS = config.getfloat('ERT','targetRMS')
        self.scalingfactor = config.getfloat('ERT','scalingfactor')
        self.nparallel = int(config.getfloat('ERT','nparallel'))
        self.nobs = int(config.getfloat('ERT','nobs'))
        self.nparticles = int(config.getfloat('svgd','nparticles'))
        self.nelement = int(config.getint('ERT','nelement'))
       
        
        if(mask is None):
            mask = np.full(self.nelement,False)
        self.mask = mask


    def ERT_gradient(self, theta):
        
        loss, grad = self.run_R2(theta, self.data, self.config, client=self.client)
        # update grad
        # grad[:,self.mask] = 0
        
        # grad =  grad/self.sigma/self.sigma

        # log likelihood
        # return 0.5*loss/self.sigma/self.sigma, grad
        return 0.5*loss, grad
    
    
    def run_R2(self, theta, data, config, client=None):

        log_con = theta.astype('float64')
        loss = np.zeros((theta.shape[0],))
        grad = np.zeros_like(theta)
        loss_batch = np.zeros((self.nparallel))
        
        nbatch = math.floor( self.nparticles / self.nparallel )
        
        for i in range(nbatch):
            # nparellel should be 
            for j in range(i * self.nparallel, (i+1) * self.nparallel, 1):
                # print(j)
                self._WriteResistivityBatch( math.floor( j % self.nparallel) , log_con[j] )
            # Run MultiR2
            self._RunMultiR2()
            #read data_output.bin
            loss_i, grad_i = self._ReadLossGrad()
            
            for j in range(self.nparallel):
                loss_batch[j] = np.sum(loss_i[j,:]*loss_i[j,:])
                
            loss[range(i * self.nparallel, (i+1) * self.nparallel, 1)] = loss_batch
            grad[range(i * self.nparallel, (i+1) * self.nparallel, 1)] = grad_i
        
        # for i in range(log_con.shape[0]):
        #     rec_i, grad_i = self.SimulationJacobian(log_con[i,:], data)
            
        #     loss_i = np.sum((rec_i-data)**2)
            
        #     loss[i] = loss_i
        #     grad[i,:] = grad_i
        
        # print(loss)
        
        return loss, grad


    def _WriteResistivityBatch(self, parallel_id , log_con):
        '''
        Write forward model (resistivity.dat) for R2
        --------------------------------------------------
        Input:
            self.res         numpy      the resistivity input for the simulation
        '''
        resdata = np.loadtxt("./R2/resistivity.dat")
        resdata[:,2] = 1.0 / np.exp(log_con)
        resdata[:,3] = np.log10(resdata[:,2])
        np.savetxt("./R2_" + str(parallel_id) + "/resistivity.dat", resdata, delimiter="    ", fmt="%15.5e", header='')


    def _RunMultiR2(self):
        '''
        Run R2_J.exe software
        --------------------------------------------------
        '''
        # Linux
        # os.chdir('./R2')
        # # print('pwd: ',os.getcwd())
        # os.system('wine'+' R2_J.exe')
        # # os.system('wine R2_J.exe > {}'.format(os.devnull) + ' 2>&1')
        # os.chdir('../')
        # # print('pwd: ',os.getcwd())
        # # print('R2 completed!')
        # subprocess.run('./MultiR2.out')
        subprocess.run(['./MultiR2.out'], cwd='./', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def _ReadLossGrad(self):
        loss = np.zeros((self.nparallel,self.nobs), dtype=np.float64)
        grad = np.zeros((self.nparallel,self.nelement), dtype=np.float64)
        with open('./lossgrad.bin', 'rb') as f:
            for i in range(self.nparallel):
                loss[i,:] = np.fromfile(f, dtype=np.float64, count=self.nobs)
                grad[i,:] = np.fromfile(f, dtype=np.float64, count=self.nelement)  
        # time.sleep(10) 
        # subprocess.run('rm ./lossgrad.bin')      
        # subprocess.run(['rm ./lossgrad.bin'], cwd='./', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return loss, grad     
    

    def SimulationJacobian(self, log_con, data):
        '''
        HydroERT Simulation Jacobian
        --------------------------------------------------
        Output:
        '''
        # log_con 是一个矩阵，包括很多粒子数，需要多次处理，分批处理
        self._WriteResistivity(log_con)
        self._RunR2_J()
        # self._ReadForward()
        _, _, _, resistance, jacobian = self._ReadForwardandJacobian()
        #######################################################################
        # here
        log_res=np.log(resistance)
        loss = log_res - data
        
        grad = np.dot(jacobian.T, loss)
        #######################################################################
        return log_res, grad
        # num_ind_meas, num_param, elec, self.resistance, self.Jacobian = self.ReadForwardandJacobian()
        # return self.resistance, self.Jacobian
        

    def _RunR2_J(self):
        '''
        Run R2_J.exe software
        --------------------------------------------------
        '''
        # Linux
        # os.chdir('./R2')
        # # print('pwd: ',os.getcwd())
        # os.system('wine'+' R2_J.exe')
        # # os.system('wine R2_J.exe > {}'.format(os.devnull) + ' 2>&1')
        # os.chdir('../')
        # # print('pwd: ',os.getcwd())
        # # print('R2 completed!')
        
        subprocess.run(['wine','R2_J.exe'], cwd='./R2', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def _WriteResistivity(self, log_con):
        '''
        Write forward model (resistivity.dat) for R2
        --------------------------------------------------
        Input:
            self.res         numpy      the resistivity input for the simulation
        '''
        resdata = np.loadtxt('./R2/resistivity.dat')
        resdata[:,2] = 1.0 / np.exp(log_con)
        resdata[:,3] = np.log10(resdata[:,2])
        np.savetxt("./R2/resistivity.dat", resdata, delimiter="    ", fmt="%15.5e", header='')
        

    def _ReadForwardandJacobian(self):
        os.chdir('./R2')
        with open('./R2_forward.bin', 'rb') as f:
            # Read the number of measurements and parameters
            ##### read none #####
            np.fromfile(f, dtype=np.int32, count=1)
            #####################
            num_ind_meas = np.fromfile(f, dtype=np.int32, count=1)[0]
            num_param = np.fromfile(f, dtype=np.int32, count=1)[0]
            
            # Allocate arrays for the data
            fcalc = np.zeros(num_ind_meas, dtype=np.float64)
            elec = np.zeros((4, num_ind_meas), dtype=np.int32)
            Jacobian = np.zeros((num_ind_meas,num_param), dtype=np.float64)

            # Read the 'elec' data
            for j in range(4):
                ##### read none #####
                np.fromfile(f, dtype=np.int32, count=2)
                #####################
                elec[j, :] = np.fromfile(f, dtype=np.int32, count=num_ind_meas)
                
            # Read the 'fcalc' data
            ##### read none #####
            np.fromfile(f, dtype=np.float64, count=1)
            #####################
            fcalc = np.fromfile(f, dtype=np.float64, count=num_ind_meas)
            
            # Read the 'Jacobian' data
            for j in range(num_param):
                ##### read none #####
                np.fromfile(f, dtype=np.float64, count=1)
                #####################
                Jacobian[:,j] = np.fromfile(f, dtype=np.float64, count=num_ind_meas)
                # print(Jacobian[:,j])
                
        # # Save the results to a text file 'R2_forward.dat'
        # with open('R2_forward.dat', 'w') as f:
        #     for i in range(num_ind_meas):
        #         f.write(f"{i+1:8d} {' '.join(f'{elec[j,i]:5d}' for j in range(4))} {fcalc[i]:20.8e}\n")
        
        # # Save the Jacobian to a text file 'R2_Jacobian.dat'
        # with open('R2_Jacobian.dat', 'w') as f:
        #     for j in range(num_param):
        #         f.write(f"{j+1}\n")
        #         np.savetxt(f, Jacobian[:,j], fmt="%20.8e")
                
        # print("all ok")
        os.system('rm R2_forward.bin')
        os.chdir('../')
        return num_ind_meas, num_param, elec, fcalc, Jacobian


    def dlnprob(self, theta):
        '''
        Compute gradient of log posterior pdf
        Input
            theta: 2D array with dimension of nparticles*nparameters
        Return
            lglike: a vector of log likelihood for each particle
            grad: each row contains the gradient for each particle
            mask: an auxilary mask array for SVGD optimization, can be safely ignored
        '''

        # adjust theta such that it is within prior or transformed back to original space
        theta = self.prior.adjust(theta)

        #t = time.time()
        loss, grad = self.ERT_gradient(theta)
        lglike = -loss + self.prior.lnprob(theta)
        #print('Simulation takes '+str(time.time()-t))
        
        # Jacobi_grad = grad
        
        # compute gradient including the prior
        grad, mask = self.prior.grad(theta, grad=grad)
        grad[:,self.mask] = 0
        print(f'Average loss and negative log posterior: {np.mean(loss)} {np.mean(-lglike)}')
        print(f'RMS: {np.sqrt(np.mean(loss)/self.nobs)}')
        with open('./results/RMS.txt',"a") as f:
            np.savetxt(f,[np.sqrt(np.mean(loss)/self.nobs)])
            
        if ( np.sqrt(np.mean(loss) / self.nobs) < (self.targetRMS / self.scalingfactor) ):
            self.isstop = True
             
        print(f'Max. Mean and Median grad: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))}')
        #print(f'max, mean and median grads after transform: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))}')

        # return lglike, grad, mask, self.isstop, Jacobi_grad
        return lglike, grad, mask, self.isstop