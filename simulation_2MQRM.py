# -*- coding: utf-8 -*-
"""
Created on Wed Jun 1 09:00:02 2022

@author: Tabitha
"""

import numpy as np
import qutip as qt
import multiprocess as mp
from multiprocess import Process, Queue
import scipy as sp
import math
import random


def glist_generator(number,uniform=False):
    out = np.zeros([number],dtype = 'f')
    for i in range(number):
        if i == 0:
            out[i] = random.random()
        elif i == number-1:
            out[i] = 1-np.sum(out)
        else:
            out[i] = (1-np.sum(out))*random.random()
    if uniform==False:
        return np.sqrt(out)
    else:
        return np.ones([number])*1/np.sqrt(number)

def glist_gen_dominance(number,position,dominance):
    out = np.ones([number])*(1-dominance**2)/(number-1)
    out[position] = dominance**2
    return np.sqrt(out)
    
def cmat_generator(DG,DE,variation=0.5,uniform=False,symmetrical=False,normalised=True):
    out = np.zeros([DG,DE])
    if uniform==False:
        for i in range(DG):
            for j in range(DE):
                randomval = 2*variation*(random.random()-0.5)+1
                out[i,j] = randomval
        if symmetrical==True:
            dim = min(DG,DE)
            for i in range(dim):
                for j in range(dim):
                    out[i,j]=out[j,i]
            out = 1/np.sqrt(DG*DE) * out
        else:    
            out = 1/np.sqrt(DG*DE) *out
    else:
        for i in range(DG):
            for j in range(DE):
                out[i,j]=1/np.sqrt(DG*DE) 
    
    if normalised == False:
        U,sval,Vt = sp.linalg.svd(out)
        return out, U, sval, Vt
    else:
        U,sval,Vt = sp.linalg.svd(out)
        return 1/sval[0] * out, U, 1/sval[0]*sval, Vt
    
def cmat_perturbation_generator(DG,DE,variation=0.5,uniform=False,symmetrical=False):
    out = np.zeros([DG,DE])
    if uniform==False:
        for i in range(DG):
            for j in range(DE):
                out[i,j] = 2*variation*(random.random()-0.5)
        if symmetrical==True:
            dim = min(DG,DE)
            for i in range(dim):
                for j in range(dim):
                    out[i,j]=out[j,i]
            return 1/np.sqrt(DG*DE) * out
        else:    
            return 1/np.sqrt(DG*DE) * out
    else:
        return out

class Cmatobj:
    def __init__(self, DG, DE, variation, uniform, symmetrical, normalised):
        self.data, self.U, self.svdvals, self.Vt = cmat_generator(DG, DE, variation, uniform, symmetrical, normalised)

class CmatDiagonal:
    def __init__(self, DG, DE):
        self.data = np.diag([1 for k in range(DG)])
        self.U, self.svdvals, self.Vt = sp.linalg.svd(self.data)

class CmatRandomAF:
    def __init__(self, DG, DE, normalised, var_independent=1/np.sqrt(2)):
        self.data = np.random.normal(0,1*var_independent,size=(DG,DE)) + 1j * np.random.normal(0,1*var_independent,size=(DG,DE))
        self.U, self.svdvals, self.Vt = sp.linalg.svd(self.data)
        if normalised == True:
            self.data = 1/self.svdvals[0]*self.data
            self.svdvals = 1/self.svdvals[0] * self.svdvals
        else:
            pass
class Cmat22:
    def __init__(self,offdiagonalval):
        self.data = np.array([[1,offdiagonalval],[offdiagonalval,1]])
        self.U, self.svdvals, self.Vt = sp.linalg.svd(self.data)
        self.data = self.data*1/self.svdvals[0]
        self.svdvals = 1/self.svdvals[0] * self.svdvals

class Cmat22random:
    def __init__(self,offdiagonalval,variation):
        self.data = np.array([[1 + 2*variation*(random.random()-0.5),offdiagonalval+2*variation*(random.random()-0.5)],[offdiagonalval+2*variation*(random.random()-0.5),1+2*variation*(random.random()-0.5)]])
        self.U, self.svdvals, self.Vt = sp.linalg.svd(self.data)
        self.data = self.data*1/self.svdvals[0]
        self.svdvals = 1/self.svdvals[0] * self.svdvals

def randomlydistribute(averageval,spread,number):
    out = np.empty([number],dtype='f')
    for k in range(number):
        out[k] = averageval + spread*random.random()-0.5*spread
    return sorted(out)

def vector2(d):
    out = np.empty([d,d],dtype=object)
    for n in range(d):
        for m in range(d):
            out[n,m] = (qt.states.basis(d,n)*qt.states.basis(d,m).dag())
    return out

def matpower(matrix,n):
    y = (( np.linalg.matrix_power(matrix, n)) / int(math.factorial(n)))
    return y

def expmToN(mat,order):
    result = sum([matpower(mat,k) for k in range(0,order+1)])
    return result

def elevelspacings(eigens, eigvecs, parity_op, parity_val=1, cutoff=0.6):
    parity_expect_list = np.asarray(np.round(qt.expect(parity_op, eigvecs)),dtype=int)[0:int(round(cutoff*eigens.size))]
    print(parity_expect_list)
    array_indices = np.where(parity_expect_list==parity_val)[0]
    energies_of_parity = [eigens[k] for k in array_indices]
    energy_diff = [np.subtract(energies_of_parity[n+1], energies_of_parity[n]) for n in range(len(energies_of_parity)-1)]
    energy_diff_average = np.average(energy_diff)
    normalised_energy_diff = np.divide(energy_diff, energy_diff_average)
    return normalised_energy_diff, energy_diff

class DoubleMultilevel:
    
    def __init__(self, N, D1, D2, Cmat, geff, ep1, ep2 , wc, wa, kappa=0, gamma=0, gamma_d=0, theta=0):
        self.N = N #max cavity population
        self.D1 = D1 #ground atomic levels
        self.D2 = D2 #excited atomic levels
        self.C = Cmat
        self.Cmat = self.C.data*geff #Interaction matrix
        self.C_Ut = np.transpose(self.C.U)
        self.C_svdvals = self.C.svdvals*geff
        self.C_fullsvdvals = np.concatenate((self.C_svdvals,np.zeros(int(abs(D2-D1)))))
        self.C_Vt = self.C.Vt
        self.ep1 = ep1 #atomic energy ground level spacing
        self.ep2 = ep2 #atomic energy excited level soacing
        self.wc = wc #cavity frequency
        self.wa = wa #atom frequency
        self.kappa = kappa #cavity decay
        self.gamma = gamma #radiative devay
        self.gamma_d = gamma_d #dephasing
        self.theta = theta #pumping

        self.totalD = self.D1 + self.D2
        
        self.vectorsmat = vector2(self.totalD) #basis * basis.dag matrix
        
        self.vec = np.empty([self.totalD,self.totalD],dtype=object) #atomic generalised ladder operators vec(n,m) = |n><m|
        
        
        #Adding 0's to make bright ground and bright excited states
        self.BGround = np.concatenate((self.C_Ut, np.zeros((self.D1,self.D2))),axis=1)
        self.BExcited = np.concatenate((np.zeros((self.D2,self.D1)),self.C_Vt),axis=1)
        
        self.BGroundStates = np.empty([len(self.BGround)],dtype=object)
        self.BExcitedStates = np.empty([len(self.BExcited)],dtype=object)
        
        for k in range(len(self.BGround)):
            self.BGroundStates[k] = qt.Qobj(self.BGround[k])
        for k in range(len(self.BExcited)):
            self.BExcitedStates[k] = qt.Qobj(self.BExcited[k])
        
        for n in range(self.totalD):
            for m in range(self.totalD):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
        self.a = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.totalD))
        self.adag = self.a.dag()
        
        if self.D1 == 1:
            self.delta1 = [0]
            self.ep1 = 0
        elif isinstance(self.ep1, list):
            self.delta1 = self.ep1
            self.ep1 = np.abs(self.delta1).max()
        else:
            self.delta1 = np.linspace(-self.ep1/2,self.ep1/2,self.D1)
        
        if self.D2 == 1:
            self.delta2 = [0]
            self.ep2 = 0
        elif isinstance(self.ep2, list):
            self.delta2 = self.ep2
            self.ep2 = np.abs(self.delta2).max()
        else:
            self.delta2 = np.linspace(-self.ep2/2,self.ep2/2,self.D2)
            
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,0]*self.vec[0,n] for n in range(self.D1,self.totalD)])
        self.n_op = self.adag*self.a 
        
        self.Pexp = 1j * np.pi * self.n_op_tot
        self.P = self.Pexp.expm()
        
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.vec[0,n] for n in range(self.D1,self.totalD)]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.vec[n,n] for n in  range(self.D1,self.totalD)]
        self.coop_pumping = [np.sqrt(self.theta/(self.totalD-1))*self.vec[n,0] for n in range(self.D1,self.totalD)]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
    def hamiltonian(self):
        self.H_i = sum([sum([self.Cmat[n,m-self.D1]*(self.adag + self.a)*(self.vec[n,m] + self.vec[m,n]) for n in  range(0,self.D1)]) for m in range(self.D1,self.totalD)])
        self.H = self.wc*self.adag*self.a + sum([(-self.wa/2+self.delta1[i])*self.vec[i,i] for i in range(0,self.D1)]) \
            + sum([(self.wa/2 + self.delta2[i-self.D1])*self.vec[i,i] for i in range(self.D1,self.totalD)]) \
            + self.H_i
        return self.H
    
    def brightness_proportion(self, densitymat):
        svdvals = self.C_fullsvdvals
        brightness = 1/svdvals[0] * sum([svdvals[i]*(densitymat*(qt.tensor(qt.operators.qeye(self.N), self.BGroundStates[i]*self.BGroundStates[i].dag()) \
                                                    + qt.tensor(qt.operators.qeye(self.N), self.BExcitedStates[i]*self.BExcitedStates[i].dag()))).tr() for i in range(min(self.D1,self.D2))])
        return np.real(brightness)
    
    def s_value(self, densitymat):
        svdvals = self.C_fullsvdvals
        svalue = sum([(i+1)*(densitymat*(qt.tensor(qt.operators.qeye(self.N), self.BGroundStates[i]*self.BGroundStates[i].dag()) \
                                                    + qt.tensor(qt.operators.qeye(self.N), self.BExcitedStates[i]*self.BExcitedStates[i].dag()))).tr() for i in range(min(self.D1,self.D2))])
        return np.real(svalue)

    def parity_value(self, densitymat):
        parity = (densitymat*(self.P)).tr()
        return np.real(parity)
    
class MultiLevel:
    
    def __init__(self, N, D, geff, ep, wc, wa, kappa=0, gamma=0, gamma_d=0, theta=0, omega=0, zeta=0, displacement = 0, rwa=True):
        #system variables
        self.N = N #max cavity population
        self.D = D #atomic levels
        self.geff = geff #strength of atom cavity interaction
        self.ep = ep #atomic energy level spacing
        self.wc = wc #cavity frequency
        self.wa = wa #atom frequency
        self.kappa = kappa #cavity decay
        self.gamma = gamma #radiative devay
        self.gamma_d = gamma_d #dephasing
        self.theta = theta #pumping
        self.omega = omega #cavity driving
        self.rwa = rwa
        self.zeta1 = zeta #atomic driving first
        self.zeta2 = zeta.conjugate() #atomic driving second 
        
        self.alpha = displacement
        #multilevel energies
        
        #LOGIC FOR LISTS INPUT

        if isinstance(self.geff, np.ndarray):
            self.glist = self.geff
            self.geff = np.sqrt(np.sum(np.square(self.glist)))
        else:
            self.glist = np.linspace(self.geff/np.sqrt(self.D-1),self.geff/np.sqrt(self.D-1),self.D-1)
        
        if self.D == 2:
            self.delta = [0]
            self.ep = np.abs(self.delta).max()
        elif isinstance(self.ep, list):
            self.delta = self.ep
            self.ep = np.abs(self.delta).max()
        else:
            self.delta = np.linspace(-self.ep/2,self.ep/2,self.D-1)
        
    
        #LOGIC FOR LISTS INPUT
        
        
        #system operators - cavity - displaced automatically by alpha
        self.a  = qt.tensor(qt.displace(N,self.alpha).dag()*qt.operators.destroy(self.N)*qt.displace(N,self.alpha), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        
        self.aori  = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adagori = self.aori.dag()
        #system operators - atom
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators vec(n,m) = |n><m|
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
        
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,0]*self.vec[0,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a 
        
        self.Pexp = 1j * np.pi * self.n_op_tot
        self.P = self.Pexp.expm()
        
        self.bright = sum([self.glist[n-1]*qt.states.basis(self.D,n) for n in range(1,self.D)]) #vector
        self.ground = self.vec[0,0] #ground*ground.dag
        
        ##MBSM unitary transforms
        self.beta = self.wc + self.wa
        self.Lambda=[self.glist[k-1]/(self.beta+self.ep*((k-1)/(self.D-1)-0.5)) for k in range(1,self.D)]
        self.phi = 1/(2*wc) * np.array([[self.Lambda[k-1]*self.glist[j-1] for j in range(1,self.D)] for k in range(1,self.D)])
        
        
        U1mat = sum([self.Lambda[k-1]*(self.adag*self.vec[k,0] - self.a*self.vec[0,k]) for k in range(1,self.D)])
        U2mat = sum([sum([self.phi[j-1,k-1] * (self.adag**2 * qt.operators.commutator(self.vec[j,0],self.vec[0,k])\
                                                                 -self.a**2 * qt.operators.commutator(self.vec[k,0],self.vec[0,j]))for j in range(1,self.D)]) for k in range(1,self.D)])
        
        self.U1 = qt.Qobj(np.real(sp.linalg.expm(U1mat.full())), dims=[[self.N, self.D], [self.N, self.D]])
        self.U2 = qt.Qobj(np.real(sp.linalg.expm(U2mat.full())), dims=[[self.N, self.D], [self.N, self.D]])
        self.U1_toOrder = qt.Qobj(np.real(expmToN(U1mat.full(),2)), dims=[[self.N, self.D], [self.N, self.D]])
        self.U2_toOrder = qt.Qobj(np.real(expmToN(U2mat.full(),1)), dims=[[self.N, self.D], [self.N, self.D]])
        self.U = self.U2 * self.U1
        self.Udag = self.U.dag()
        self.U1dag = self.U1.dag()
        self.U2dag = self.U2.dag()
        self.U_toOrder = self.U2_toOrder * self.U1_toOrder
        self.U_toOrder_dag = self.U1_toOrder.dag()
        self.U1_toOrder_dag = self.U1_toOrder.dag()
        self.U2_toOrder_dag = self.U2_toOrder.dag()
        
        
    def hamiltonian_nodriving(self):
        if self.rwa==True:
            #constructing hamiltonian in RWA
            self.H_i = sum([self.glist[n-1]*(self.adag*self.vec[0,n] + self.a*self.vec[n,0]) for n  in  range(1,self.D)])
            
        elif self.rwa==False:
            #constructing hamiltonian without RWA
            self.H_i = sum([self.glist[n-1]*(self.adag + self.a)*(self.vec[0,n] + self.vec[n,0]) for n  in  range(1,self.D)])
        
        self.H = self.wc*self.adag*self.a + sum([(self.wa + self.delta[i-1])*self.vec[i,i] for i in range(1,self.D)]) \
            + self.H_i
        return self.H
    
    def hamiltonian_pdsc(self):
        if self.rwa==True:
            #constructing hamiltonian in RWA
            self.H_i = sum([self.glist[n-1]*(self.adag*self.vec[0,n] + self.a*self.vec[n,0]) for n  in  range(1,self.D)])
        
        elif self.rwa==False:
            #constructing hamiltonian without RWA
            self.H_i = sum([self.glist[n-1]*(self.adag + self.a)*(self.vec[0,n] + self.vec[n,0]) for n  in  range(1,self.D)])
        
        self.H = self.wc*self.adag*self.a + self.H_i - 0.5*self.wc*sum([self.vec[n,n] for n  in  range(1,self.D)])
        return self.H
    
    def hamiltonian_withdriving(self):
        self.V = self.omega*(self.a + self.adag) + (sum([self.zeta2*self.vec[0,n] + self.zeta1*self.vec[n,0] for n in range(1,self.D)]))
        self.wl_list = np.linspace(self.start + self.wc, self.end +self.wc, self.accuracy)
        self.Htot = np.empty([self.accuracy],dtype=object)
        for i in range(self.accuracy):
            self.Htot[i] = self.H + self.V - self.wl_list[i]*self.adag*self.a \
                - self.wl_list[i]*sum([self.vec[n,n] for n in range(1,self.D)]) #needs to be original as done after disp transform
        return self.Htot
        
    def hamiltonian(self, accuracy=0, start=0, end=0, suppress=False):
        if self.omega==0 and self.zeta1==0:
            if suppress==False:
                print("hamiltonian_nodriving working...")
            else:
                pass
            return self.hamiltonian_nodriving()
        elif accuracy ==0:
            if suppress==False:
                print("hamiltonian_withdriving working... (single), confusion with take away laser frequency, automatically 0")
            else:
                pass
            self.H = self.hamiltonian_nodriving() + self.omega*(self.a + self.adag) + (sum([self.zeta2*self.vec[0,n] + self.zeta1*self.vec[n,0] for n in range(1,self.D)]))
            return self.H
        else:
            print("hamiltonian_withdriving working... (multiple)")
            if start==0 and end==0:
                start = -np.pi*self.geff + self.wc
                end = np.pi*self.geff + self.wc
                print("automatic driving bounds used")
            else:
                print("manual driving bounds used")
                pass
            self.start = start
            self.end = end
            self.accuracy = accuracy
            self.hamiltonian_nodriving()
            return self.hamiltonian_withdriving()
    
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.vec[0,n] for n in range(1,self.D)]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.vec[n,n] for n in  range(1,self.D)]
        self.coop_pumping = [np.sqrt(self.theta/(self.D-1))*self.vec[n,0] for n in range(1,self.D)]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
        return self.c_ops

    def g2listcalc(self,operator):
        self.g2list = np.empty([len(self.Htot)],dtype=np.float64)
        for i in range(len(self.wl_list)):
            self.g2list[i] = qt.coherence_function_g2(self.Htot[i], None, [0], self.c_ops, operator)[0][0]
            print(i/len(self.wl_list))
        return self.g2list

    def g2listcalcmp(self,operator):
      num_sims = len(self.Htot)
      num_threads = mp.cpu_count() if mp.cpu_count()<num_sims else num_sims
      manager = mp.Manager()
      return_dict = manager.dict()
      jobs = []

      def g2listcalcmp_helper(start,end,procnum) -> None:
        ## TODO: refactor this into multiple functions
        self.g2list_temp = np.empty([end-start],dtype=np.float64)
        for s in range(start,end):
          self.g2list_temp[s-start] = np.real(qt.coherence_function_g2(self.Htot[s], None, [0], self.c_ops, operator)[0][0])

          print(f"Process #{procnum}: {int(((s-start)/(end-start))*100)}% complete")

        print(f"Process #{procnum}: 100% complete")

        return_dict[procnum] = self.g2list_temp

      ## Create processes and dynamically allocate simulations
      for i in range(num_threads):
        start_index = 0 if i==0 else int(i/num_threads * num_sims)
        end_index = num_sims if i+1 == num_threads else int((i+1)/num_threads * num_sims)

        p = Process(target=g2listcalcmp_helper, args=[start_index,end_index,i])
        jobs.append(p)
        p.start()

      for p in jobs: # wait for all jobs to finish before continuing
        p.join()

      ## Stitch values returned from each process back together into a single array
      self.g2list = np.empty([num_sims],dtype=np.float64)
      index = 0
      for i in range(num_threads):
        for j in return_dict[i]:
          self.g2list[index] = j
          index+=1

      return self.g2list

    
    def ss_dm(self, driving=False): #steady state density matrix
        if driving == False:
            self.ss_dm = qt.steadystate(self.H,self.c_ops)
            return self.ss_dm
        elif driving==True:
            self.ss_dm = np.empty([self.accuracy],dtype=object)
            for i in range(self.accuracy):
                self.ss_dm[i] = qt.steadystate(self.Htot[i], self.c_ops)
            return self.ss_dm
        
    def darkstate_proportion(self, driving=False):
        if driving == False:
            self.pdark = 1-(self.ss_dm*(self.ground + qt.tensor(qt.operators.qeye(self.N), self.bright*self.bright.dag()/self.geff**2))).tr()
            return np.real(self.pdark)
        elif driving == True:
            self.pdark = np.empty([self.accuracy],dtype=object)
            for i in range(self.accuracy):
                self.pdark[i] = np.real(1-(self.ss_dm[i]*(self.ground + qt.tensor(qt.operators.qeye(self.N), self.bright*self.bright.dag())/self.geff**2)).tr())
            return self.pdark
    
    def darkstate_proportion_external(self, densitymat):
        self.pdark = 1-(densitymat*(self.ground + qt.tensor(qt.operators.qeye(self.N), self.bright*self.bright.dag()/self.geff**2))).tr()
        return np.real(self.pdark)

    
        
class HighMultilevel:
    
    def __init__(self, N, D, geff, ep, wc, wa, C, crit=0.3, kappa=0, gamma=0, gamma_d=0, theta=0):
        #system variables
        self.N = N #max cavity population
        self.D = D #atomic levels
        self.geff = geff #strength of atom cavity interaction
        self.ep = ep #atomic energy level spacing
        self.wc = wc #cavity frequency
        self.wa = wa #atom frequency
        self.C = C
        self.kappa = kappa #cavity decay
        self.gamma = gamma #radiative devay
        self.gamma_d = gamma_d #dephasing
        self.theta = theta #pumping
        
        self.a = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        #multilevel energies
        
        
        #LOGIC FOR LISTS INPUT
 
        if isinstance(self.geff, np.ndarray):
            self.glist = self.geff
            self.geff = np.sqrt(np.sum(np.square(self.glist)))
        else:
            self.glist = np.linspace(self.geff/np.sqrt(self.D-1),self.geff/np.sqrt(self.D-1),self.D-1)
        
        if self.D == 2:
            self.delta = [0]
            self.ep = np.abs(self.delta).max()
        elif isinstance(self.ep, list):
            self.delta = self.ep
            self.ep = np.abs(self.delta).max()
        else:
            self.delta = np.linspace(-self.ep/2,self.ep/2,self.D-1)
        
        
        #LOGIC FOR LISTS INPUT
        
        
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators vec(n,m) = |n><m|
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
        
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,0]*self.vec[0,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a 
        
        self.Pexp = 1j * np.pi * self.n_op_tot
        self.P = self.Pexp.expm()
        self.crit = crit
        if self.geff<self.crit:
            self.x0=0
        else:
            self.x0 = math.sqrt(2*C)*math.sqrt(self.geff**2/self.wc**2 - self.wa**2/(16*self.geff**2))#geff
            
        self.bright = sum([self.glist[n-1]*qt.states.basis(self.D,n) for n in range(1,self.D)]) #vector
        self.ground = self.vec[0,0] #ground*ground.dag
            
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.vec[0,n] for n in range(1,self.D)]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.vec[n,n] for n in  range(1,self.D)]
        self.coop_pumping = [np.sqrt(self.theta/(self.D-1))*self.vec[n,0] for n in range(1,self.D)]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
            
        return self.c_ops
                
    def hamiltonian(self):
        self.H_i = sum([self.glist[n-1]*(self.adag + self.a)*(self.vec[0,n] + self.vec[n,0]) for n  in  range(1,self.D)])
        self.H = self.wc/math.sqrt(self.C)*self.adag*self.a + sum([math.sqrt(self.C)*(self.wa + self.delta[i-1])*self.vec[i,i] for i in range(1,self.D)]) + self.H_i \
            + self.wc*self.x0/math.sqrt(self.C)*1/math.sqrt(2)*(self.adag+self.a)\
                +math.sqrt(2)*self.x0*sum([self.glist[n-1]*(self.vec[0,n] + self.vec[n,0]) for n  in  range(1,self.D)])
        return self.H
            
class DegenBlochSiegert: 
    
    def __init__(self, N, D, geff, wc, wa):
        self.N=N
        self.D=D
        self.geff=geff
        self.wc=wc
        self.wa=wa
        self.g=self.geff/np.sqrt(self.D-1)
        self.a = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
                
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a
        
        self.Op = sum([self.vec[n,0] for n in range(1,self.D)])
        self.Om = self.Op.dag()
        
        self.Oz = sum([self.vec[n,n] for n in range(1,self.D)]) - self.vec[0,0]
        self.I=qt.tensor(qt.operators.qeye(self.N),qt.operators.qeye(self.D))
        
    def hamiltonian(self):
        self.H=self.wc*self.adag*self.a + 0.5*self.wa*(self.Oz+self.I) + self.g*(self.a*self.Op + self.adag*self.Om) + self.g**2/(self.wa+self.wc)*(self.adag*self.a*(self.Op*self.Om-self.Om*self.Op)-(self.D-1)/2*(self.I-self.Oz))
        return self.H
    
class GeneralBlochSiegert:
    
    def __init__(self, N, D, geff, ep, wc, wa):
        self.N=N
        self.D=D
        self.geff=geff
        self.ep=ep
        self.wc=wc
        self.wa=wa
        self.beta=wc+wa
        self.glist = np.linspace(self.geff/np.sqrt(self.D-1),self.geff/np.sqrt(self.D-1),self.D-1)
        self.a = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators
        self.Lambda=[self.glist[k-1]/(self.beta+self.ep*((k-1)/(self.D-1)-0.5)) for k in range(1,self.D)]
        self.phi = 1/(2*wc) * np.array([[self.Lambda[k-1]*self.glist[j-1] for j in range(1,self.D)] for k in range(1,self.D)])
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
            
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a
        
        self.Op = sum([self.vec[n,0] for n in range(1,self.D)])
        self.Om = self.Op.dag()
        
        self.Oz = sum([self.vec[n,n] for n in range(1,self.D)]) - self.vec[0,0]
        self.I=qt.tensor(qt.operators.qeye(self.N),qt.operators.qeye(self.D))    
        
        self.U1 = qt.Qobj(np.real(sp.linalg.expm(sum([self.Lambda[k-1]*(self.adag*self.vec[k,0] - self.a*self.vec[0,k]) for k in range(1,self.D)]))), dims=[[self.N, self.D], [self.N, self.D]])
        self.U2 = qt.Qobj(np.real(sp.linalg.expm(sum([sum([self.phi[j-1,k-1] * (self.adag**2 * qt.operators.commutator(self.vec[j,0],self.vec[0,k])\
                                                                -self.a**2 * qt.operators.commutator(self.vec[k,0],self.vec[0,j]))for j in range(1,self.D)]) for k in range(1,self.D)]))), dims=[[self.N, self.D], [self.N, self.D]])
        
        self.U1_toOrder = qt.Qobj(np.real(expmToN(sum([self.Lambda[k-1]*(self.adag*self.vec[k,0] - self.a*self.vec[0,k]) for k in range(1,self.D)]),2)), dims=[[self.N, self.D], [self.N, self.D]])
        self.U2_toOrder = qt.Qobj(np.real(expmToN(sum([sum([self.phi[j-1,k-1] * (self.adag**2 * qt.operators.commutator(self.vec[j,0],self.vec[0,k])\
                                                                -self.a**2 * qt.operators.commutator(self.vec[k,0],self.vec[0,j]))for j in range(1,self.D)]) for k in range(1,self.D)]),1)), dims=[[self.N, self.D], [self.N, self.D]])
        
        self.U = self.U2 * self.U1
        self.Udag = self.U.dag()
        self.U1dag = self.U1.dag()
        self.U2dag = self.U2.dag()
        
        self.U_toOrder = self.U2_toOrder * self.U1_toOrder
        self.U_toOrder_dag = self.U1_toOrder.dag()
        self.U1_toOrder_dag = self.U1_toOrder.dag()
        self.U2_toOrder_dag = self.U2_toOrder.dag()
        
        self.Pexp = 1j * np.pi * self.n_op_tot
        self.P = self.Pexp.expm()
    
    def hamiltonian(self):
        if self.D>2:
            self.H_0 = self.wc*self.adag*self.a + 0.5*self.wa*(self.Oz+self.I)
            self.H_ep = self.ep/(self.D-2) * sum([(k-1)*self.vec[k,k] for k in range(1,self.D)]) -self.ep/4 * (self.I+ self.Oz)
            self.H_r = self.a*sum([self.glist[k-1]*self.vec[k,0] for k in range(1,self.D)]) + self.adag*sum([self.glist[k-1]*self.vec[0,k] for k in range(1,self.D)])
            self.H_n = self.adag*self.a*sum([sum([(qt.operators.commutator(self.vec[k,0],self.vec[0,j])*\
                            (self.glist[j-1]*self.Lambda[k-1]+self.glist[k-1]*self.Lambda[j-1]\
                            -(self.beta+self.ep*(j+k-2)/(2*(self.D-2)) -self.ep/2)*self.Lambda[j-1]*self.Lambda[k-1])) for j in range(1,self.D)]) for k in range(1,self.D)])
            self.H_i = -0.5*(self.I-self.Oz)*sum([2*self.glist[j-1]*self.Lambda[j-1]\
                            -(self.beta+self.ep*(j-1)/(self.D-2)-self.ep/2)*self.Lambda[j-1]**2 for j in range(1,self.D)])
            self.H = self.H_0 + self.H_ep + self.H_r + self.H_n + self.H_i
            return self.H
        elif self.D==2:
            tempsys = DegenBlochSiegert(self.N, self.D, self.geff, self.wc, self.wa) #since ep=0 may aswell o.o haha sneaky
            self.H = tempsys.hamiltonian()
            return self.H
    
class Dicke:
    
    def __init__(self, N, M, geff, wc, wa, kappa = 0, gamma=0, gamma_d=0, theta=0, omega = 0, tc = True):
        self.N = N #cavity levels 
        self.M = M #number of atoms
        self.geff = geff #coupling
        self.wc = wc #cavity frequency
        self.wa = wa #atomic frequency
        self.kappa = kappa
        self.gamma = gamma #atom decay
        self.gamma_d = gamma_d #atom dephasing
        self.theta = theta #atom inconherent pumping
        self.tc = tc
        self.omega = omega
        
        self.g = self.geff/np.sqrt(self.M)
        
        self.j = M/2
        self.n = 2*self.j+1
        self.a  = qt.tensor(qt.operators.destroy(N),qt.operators.qeye(int(self.n)))
        self.adag = self.a.dag()
        
        self.Jp = qt.tensor(qt.operators.qeye(N),qt.operators.jmat(self.j, '+'))
        self.Jm = qt.tensor(qt.operators.qeye(N),qt.operators.jmat(self.j, '-'))
        self.Jz = qt.tensor(qt.operators.qeye(N),qt.operators.jmat(self.j, 'z'))
        
        self.N = self.adag*self.a + self.Jz + self.j
        self.Pexp = 1j * np.pi * self.N
        self.P = self.Pexp.expm()
        
    def hamiltonian(self):
        self.H = self.wc*self.adag*self.a + self.wa*self.Jz/2 + self.geff/np.sqrt(self.M)*(self.Jp * self.a + self.Jm * self.adag)
        if self.tc == False:
            self.H = self.H + self.geff/np.sqrt(self.M)*(self.Jp*self.adag + self.Jm*self.a)
        else:
            pass
        return self.H
    
    def hamiltonian_withdriving(self, start, end, accuracy):
        self.V = self.omega*(self.a + self.adag)
        self.wl_list = np.linspace(start + self.wc, end +self.wc, accuracy)
        self.Htot = np.empty([accuracy],dtype=object)
        for i in range(accuracy):
            self.Htot[i] = self.H + self.V - self.wl_list[i]*self.adag*self.a \
                - self.wl_list[i]*self.Jz/2 
        return self.Htot
    
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.Jm]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.Jz]
        self.coop_pumping = [np.sqrt(self.theta)*self.Jp]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
        return self.c_ops
    
    def g2listcalc(self, operator):
        self.g2list = np.empty([len(self.Htot)],dtype=np.float64)
        for i in range(len(self.wl_list)):
            self.g2list[i] = qt.coherence_function_g2(self.Htot[i], None, [0], self.c_ops, operator)[0][0]
            print(i/len(self.wl_list))
        return self.g2list
    
class JCCoherent:
    def __init__(self, N, g, wc, wa, wl, omega = 0, Lambda = 0, kappa = 0, gamma=0, gamma_d=0, D=2):
        self.N = N #cavity levels 
        self.D = D #atomic levels
        self.g = g #coupling
        self.wl = wl #Driving Laser freq
        self.wc = wc #cavity frequency
        self.wa = wa #atom frequency
        
        self.kappa = kappa #cavity decay
        self.gamma = gamma #atom decay
        self.gamma_d = gamma_d #atom dephasing
        
        self.omega = omega
        self.Lambda = Lambda
        self.a  = qt.tensor(qt.operators.destroy(N),qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators vec(n,m) = |n><m|
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
        
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,0]*self.vec[0,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a 
        
        self.n_op_atom = sum([self.vec[n,0]*self.vec[0,n] for n in range(1,self.D)])
        
        self.alpha = self.Lambda/self.g
        self.Delta = self.wa-self.wl
        self.delta = self.wc-self.wl
        
        self.Omegatilde = self.alpha*self.delta - 1j*self.kappa/2 * self.alpha -self.omega
        self.Lambdatilde = self.g*self.alpha-self.Lambda
        
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.vec[0,n] for n in range(1,self.D)]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.vec[n,n] for n in  range(1,self.D)]
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing 
        return self.c_ops
    
    def hamiltonian(self):
        self.Hjc = self.delta*self.adag*self.a + self.Delta*self.vec[1,0]*self.vec[0,1] + self.g*(self.vec[1,0] * self.a + self.vec[0,1] * self.adag)
        self.Hd = self.Omegatilde*self.adag + np.conjugate(self.Omegatilde)*self.a + self.Lambdatilde*self.vec[1,0] + np.conjugate(self.Lambdatilde)*self.vec[0,1]
        self.H = self.Hjc + self.Hd
        return self.H
    
    def ss_dm(self): #steady state density matrix
        self.ss_dm = qt.steadystate(self.H,self.c_ops)
        return self.ss_dm
    
    def x_n_atom(self):
        return qt.expect(self.n_op_atom, self.ss_dm)
    
    def x_fidelity(self):
        return qt.expect(self.vec[0,0], self.ss_dm)
    
    def x_coherence(self):
        return np.absolute(qt.expect(self.vec[0,1],self.ss_dm))**2