import numpy as np
import matplotlib.pyplot as plt
from math import factorial
pi = np.pi

class modJoint:
    
    def __init__(self, M=20,gammaCB=0.4,kappaCB= 2.5,noiseCB=0.0,gammaSB=2,lambdaSB=10,p_same=0.9,stim_trans=False,
                 stim_prior=False,VERBOSE = False,uniform_tuning=False,uniform_trans=False):
        
        # generative model (CB)
        self.M = M
        self.psi = np.linspace(-pi,pi,self.M) # neuron selectivity
        self.gammaCB = gammaCB 
        self.kappaCB = kappaCB
        self.noiseCB = noiseCB
        
        # prior distribution
        self.n_view = 180
        self.theta = np.linspace(-pi,pi-(2*pi/self.n_view),self.n_view)
        self.ori = np.arange(self.n_view)
        self.p_theta = 2 - np.abs(np.sin(self.theta)) # prior distribution
        self.p_theta/=np.sum(self.p_theta) # PDF
        D = np.cumsum(self.p_theta) # CDF
        D = D*2*pi-pi
        self.D = D
        self.spks_sim = np.arange(30)
        self.ori = np.arange(self.n_view)
        
        # SD
        self.n_step = self.n_view
        self.s_0 = np.linspace(0,180,self.n_step)
#         self.kappaSB = kappaSB # concentration of sensory error, is this in paper?
        self.gammaSB = gammaSB
        self.lambdaSB = lambdaSB
        self.sensoryNoise=None
        self.p_same = p_same
        
        # stim_flags
        self.stim_transition = stim_trans
        self.stim_prior = stim_prior
        
        # model flags
        self.uniform_tuning = uniform_tuning  #CB
        self.uniform_trans = uniform_trans    #SB only affects decoder, NOT stim seq. generation
        self.VERBOSE = VERBOSE
        
        self.PAPERS =['https://www.jneurosci.org/content/jneuro/38/32/7132.full.pdf',
                          'https://www.biorxiv.org/content/10.1101/671958v1']
        
    def gauss(self,x,mu=0,sig=1): return 1/(np.sqrt(2*pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))
    def inv_cdf(self,cdf,probe): return np.argmin(np.abs(np.expand_dims(cdf,1) - probe),0) # draws from 'y'-axis
    def angle(self,s_0,s_1):
        d = s_0-s_1
        d[np.abs(d)>90]-= 180*np.sign(d[np.abs(d)>90])
        return d

    def fs(self,stim): # expected firing rate for neurons given ori
        if self.uniform_tuning: # control model with uniform tuning functions
            Du = self.theta[stim]
        else:
            Du = self.D[stim]
        out = np.ones((len(stim),self.M))
        for i in range(len(stim)):
            out[i,:]=  np.exp(self.kappaCB*np.cos(Du[i]-self.psi)-1)
        out*=self.gammaCB/self.M
        out+=self.noiseCB # additive noise to be accounted for
        return out

    def pos_exp(self): # poisson expected n spikes given rate
        resp_exp=[]
        for ni in self.spks_sim:
            resp_exp.append(np.array(self.fs(self.ori)**ni/factorial(ni) * np.exp(-self.fs(self.ori))))
        return np.array(resp_exp)

    def cf(self,s_1): ## eq 6 full transition model (with uniform part)
        return self.p_same*self.c(s_1) + (1-self.p_same)*(1/self.n_step)

    def c(self,s_1): ## eq 7  distribution of transition model
        full = np.exp(-1/(2*self.lambdaSB**2)*np.abs(self.angle(self.s_0,s_1))**self.gammaSB)
        return full/np.sum(full)

    def pm(self,s_1): ## eq. 10 prob m given s
        full = np.exp(-1/(2*self.sensoryNoise**2)*np.abs(self.angle(self.s_0,s_1))**2)
        return full/np.sum(full)
    
    def gen_stim(self): 
        if not(self.stim_transition or self.stim_prior): # "Task" stimuli
            self.stim = np.random.randint(0,180,self.N)
        elif (self.stim_transition) and not (self.stim_prior): # just SB
            self.stim = np.zeros(self.N,dtype=int)
            self.stim[0] = np.random.randint(0,self.n_view) # seed first trial
            for i in range(self.N-1):
                trans_prob = self.cf(self.stim[i])
                self.stim[i+1] = self.inv_cdf(np.cumsum(trans_prob),np.random.rand())

        elif (self.stim_prior) and not (self.stim_transition): #just CB
            self.stim = self.inv_cdf(self.D,np.random.rand(self.N)*2*pi-pi)
            
        elif (self.stim_prior) and (self.stim_transition): #full "Naturalisitic stimuli"
            self.stim = np.zeros(self.N,dtype=int)
            self.stim[0] = self.inv_cdf(self.D,np.random.rand()*2*pi-pi) # seed first trial
            for i in range(self.N-1):
                trans_prob = self.cf(self.stim[i])
                joint_prob = trans_prob*self.p_theta
                joint_prob/=np.sum(joint_prob) # normalize
                self.stim[i+1] = self.inv_cdf(np.cumsum(joint_prob),np.random.rand())
            
    def run_model(self,N=5000):
        self.N = N
        self.gen_stim()

        spks=np.random.poisson(self.fs(self.stim)) + np.random.poisson(self.noiseCB,(len(self.stim),self.M))# generated spikes
        n_spks = np.sum(spks,1)
        spks[n_spks==0]=np.random.poisson(self.fs(self.stim[n_spks==0])) # generated spikes
        n_spks = np.sum(spks,1)
        spks[n_spks==0]=np.random.poisson(self.fs(self.stim[n_spks==0])) # generated spikes
        n_spks = np.sum(spks,1)

        exp_out = self.pos_exp() # prior expectation for # of spikes

        # get log-likelihood for each stimuli! 
        ll = np.zeros((self.N,180))
        for s in range(self.N):
            for i in range(self.M):
                this_ll = exp_out[spks[s][i],:,i]
                ll[s,:] += np.log(this_ll.astype(float))
        ll = np.exp(ll)
        dec_stim = np.argmax(ll,1)
        E_naive = self.angle(dec_stim,self.stim)
        self.sensoryNoise = np.std(E_naive)
        if self.VERBOSE: print(self.sensoryNoise)

        # remove trials with no spikes
        self.stim = self.stim[n_spks>0]
        dec_stim = dec_stim[n_spks>0]
        if self.VERBOSE:
            print('Throwing away %d trials with zero [0] spikes' %sum(n_spks==0))
            print('Med of %d Spikes' %(np.median(n_spks)))
        sensory = ll[n_spks>0,:]

        self.N = np.shape(sensory)[0]
        self.stimHat = np.zeros(self.N)
        for i in range(self.N):
            if self.uniform_trans: # SB control, uniform prior
                if i==0: # never need to recreate
                    prior = np.ones(self.n_step)/self.n_step
            else:
                if i ==0:
                    prior = self.cf(self.stim[-1]) # need to start with something
                else:
                    prior = self.cf(self.stimHat[i-1])

            sensory_estimate = sensory[i,:]
            ps_m = prior * sensory_estimate
            ps_m /= np.sum(ps_m)
            decoded_ori = np.argmax(ps_m) # could use circ_mean
            self.stimHat[i] = self.s_0[decoded_ori]

        self.E = self.angle(self.stimHat,self.stim)

    # visualize results
    def quick_view_sb(self,lwin=100,vis=0):
        E = self.angle(self.stimHat,self.stim)
        self.d = self.angle(self.stim[1:],self.stim[:-1])
        Eu = self.E[1:]

        order = np.argsort(self.d)
        conv_win = self.gauss(np.linspace(-2,2,lwin))
        conv_win/=np.sum(conv_win)
        SBc_E = np.convolve(Eu[order],conv_win,mode='valid')
        SBc_aE = np.convolve(np.abs(Eu[order]),conv_win,mode='valid')
        SBc_d = np.convolve(self.d[order],conv_win,mode='valid')

        if vis:       
            plt.subplot(121)
            plt.plot(SBc_d,SBc_E)
            plt.title('Serial Bias')
            plt.ylabel('Bias (deg)')
            plt.xlabel('$Correct_0 - Correct_{-1}$')
            plt.subplot(122)
            plt.plot(SBc_d,SBc_aE)
            plt.ylabel('|Error| (deg)')
            plt.xlabel('$Correct_0 - Correct_{-1}$')
            plt.tight_layout()
            plt.show()
        else:
            self.SBc_d,self.SBc_E,self.SBc_aE = SBc_d,SBc_E,SBc_aE
            
    def quick_view_cb(self,lwin=100,vis=0): # ,stim,dec_stim=None
        conv_win = self.gauss(np.linspace(-2,2,lwin))
        conv_win/=np.sum(conv_win)
        order = np.argsort(self.stim)
        CBc_aE = np.convolve(np.abs(self.E[order]),conv_win,mode='valid')
        CBc_E = np.convolve(self.E[order],conv_win,mode='valid')
        CBc_stim = np.convolve(self.stim[order],conv_win,mode='valid')
        
        if vis:
            plt.subplot(121)
            plt.plot(vis_s,vis_E)
            plt.title('Cardinal Bias')
            plt.ylabel('Bias (deg)')
            plt.xlabel('Orientation Stim (deg)')
            plt.subplot(122)
            plt.plot(vis_s,vis_aE)
            plt.ylabel('|Error| (deg)')
            plt.xlabel('Orientation Stim (deg)')
            plt.tight_layout()
            plt.show()
        else:
            self.CBc_stim,self.CBc_E,self.CBc_aE = CBc_stim, CBc_E, CBc_aE

    def quick_view_results(self,lwin=1000,sav_root=0): 
        self.quick_view_sb(lwin,vis=0)
        plt.subplot(221)
        plt.plot(self.SBc_d,self.SBc_E)
        plt.title('Serial Bias (%d)' %(self.stim_transition*1))
        plt.ylabel('Bias (deg)')
        plt.xlabel('$Correct_0 - Correct_{-1}$')
        plt.subplot(222)
        plt.plot(self.SBc_d,self.SBc_aE)
        plt.title('$\gamma:%.1f$ $\lambda:%.1f$' %(self.gammaSB,self.lambdaSB))
        plt.ylabel('|Error| (deg)')
        plt.xlabel('$Correct_0 - Correct_{-1}$')

        self.quick_view_cb(lwin,vis=0)
        plt.subplot(223)
        plt.plot(self.CBc_stim,self.CBc_E)
        plt.title('Cardinal Bias (%d)' %(self.stim_prior*1))
        plt.ylabel('Bias (deg)')
        plt.xlabel('Orientation Stim (deg)')
        plt.subplot(224)
        plt.plot(self.CBc_stim,self.CBc_aE)
        plt.ylabel('|Error| (deg)')
        plt.xlabel('Orientation Stim (deg)')
        plt.title('$\gamma:%.1f$ $\kappa:%.1f$ noise:%d' %(self.gammaCB,self.kappaCB,self.noiseCB*100))
        plt.tight_layout()
        if sav_root:
            sav_str = sav_root + 'modelJoint_CB_g%d_k%d_n%d_SB_g%d_l%d.png' %(self.gammaCB*10,self.kappaCB*10,self.noiseCB*100,self.gammaSB,self.lambdaSB)
            plt.savefig(sav_str)
        plt.show()
