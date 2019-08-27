import numpy as np
import matplotlib.pyplot as plt
from math import factorial
pi = np.pi

class modJoint:
    
    def __init__(self, M=20,gamma=0.4,kappaCB= 2.5,noiseFR=0.0,kappaSB=5,gammaSB=2,sensoryNoise=10,lambdaSB=10,p_same=0.9):
        
        # generative model (CB)
        self.M = M
        self.psi = np.linspace(-pi,pi,self.M) # neuron selectivity
        self.gamma = gamma 
        self.kappaCB = kappaCB
        self.noiseFR = noiseFR
        
        # prior distribution
        self.n_view = 180
        theta = np.linspace(-pi,pi-(2*pi/self.n_view),self.n_view)
        self.ori = np.arange(self.n_view)
        p_theta = 2 - np.abs(np.sin(theta)) # prior distribution
        D = np.cumsum(p_theta) # CDF
        D/=D[-1]
        D = D*2*pi-pi
        self.D = D
        self.spks_sim = np.arange(30)
        self.ori = np.arange(self.n_view)
        
        # SD
        self.n_step = self.n_view
        self.s_0 = np.linspace(0,180,self.n_step)
        self.kappaSB = kappaSB # concentration of sensory error
        self.gammaSB = gammaSB
        self.lambdaSB = lambdaSB
        self.sensoryNoise=sensoryNoise
        self.p_same = p_same
        
    def gauss(self,x,mu=0,sig=1): return 1/(np.sqrt(2*pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))
    def angle(self,s_0,s_1):
        d = s_0-s_1
        d[np.abs(d)>90]-= 180*np.sign(d[np.abs(d)>90])
        return d

    def fs(self,stim): # expected firing rate for neurons given ori
        Du = self.D[stim]
        out = np.ones((len(stim),self.M))
        for i in range(len(stim)):
            out[i,:]=  np.exp(self.kappaCB*np.cos(Du[i]-self.psi)-1)
        out*=self.gamma/self.M
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

    def run_model(self,N=5000):

        stim = np.random.randint(0,180,N)
        spks=np.random.poisson(self.fs(stim)) + np.random.poisson(self.noiseFR,(len(stim),self.M))# generated spikes
        n_spks = np.sum(spks,1)
        spks[n_spks==0]=np.random.poisson(self.fs(stim[n_spks==0])) # generated spikes
        n_spks = np.sum(spks,1)
        spks[n_spks==0]=np.random.poisson(self.fs(stim[n_spks==0])) # generated spikes
        n_spks = np.sum(spks,1)

        exp_out = self.pos_exp() # prior expectation for # of spikes

        # get log-likelihood for each stimuli! 
        ll = np.zeros((N,180))
        for s in range(N):
            for i in range(self.M):
                this_ll = exp_out[spks[s][i],:,i]
                ll[s,:] += np.log(this_ll.astype(float))
        ll = np.exp(ll)
        dec_stim = np.argmax(ll,1)
        E_naive = self.angle(dec_stim,stim)
        self.sensoryNoise = np.std(E_naive)
        print(self.sensoryNoise)

        # remove trials with no spikes
        stim = stim[n_spks>0]
        dec_stim = dec_stim[n_spks>0]
        print('Throwing away %d trials with zero [0] spikes' %sum(n_spks==0))
        sensory = ll[n_spks>0,:]

        N = np.shape(sensory)[0]
        inferred_ori = np.zeros(N)
        for i in range(N):
            if i ==0:
                prior = self.cf(stim[-1]) # need to start with something
            else:
                prior = self.cf(inferred_ori[i-1])

            sensory_estimate = sensory[i,:]
            ps_m = prior * sensory_estimate
            ps_m /= np.sum(ps_m)
            decoded_ori = np.argmax(ps_m) # could use circ_mean
            inferred_ori[i] = self.s_0[decoded_ori]

        E = self.angle(inferred_ori,stim)
        return stim, inferred_ori,E


    # visualize results
    def quick_view_sb(self,seq,inferred_ori,l_conv=100,vis=0):
        E = self.angle(inferred_ori,seq)
        d = self.angle(seq[1:],seq[:-1])
        Eu = E[1:]
        
        order = np.argsort(d)
        conv_win = self.gauss(np.linspace(-2,2,l_conv))
        conv_win/=np.sum(conv_win)
        E_out = np.convolve(Eu[order],conv_win,mode='valid')
        aE_out = np.convolve(np.abs(Eu[order]),conv_win,mode='valid')
        d_out = np.convolve(d[order],conv_win,mode='valid')

        if vis:       
            plt.subplot(121)
            plt.plot(d_out,E_out)
            plt.title('Serial Bias')
            plt.ylabel('Bias (deg)')
            plt.xlabel('$Correct_0 - Correct_{-1}$')
            plt.subplot(122)
            plt.plot(d_out,aE_out)
            plt.ylabel('|Error| (deg)')
            plt.xlabel('$Correct_0 - Correct_{-1}$')
            plt.tight_layout()
            plt.show()
        else:
            
            return d_out,E_out,aE_out

    def quick_view_cb(self,stim,dec_stim=None,E=None,l_conv=100,vis=0):
        if type(stim) is tuple:
            stim,dec_stim,E = stim
        E = self.angle(dec_stim,stim)
        
        conv_win = self.gauss(np.linspace(-2,2,l_conv))
        conv_win/=np.sum(conv_win)
        order = np.argsort(stim)
        c_absE = np.convolve(np.abs(E[order]),conv_win,mode='valid')
        c_E = np.convolve(E[order],conv_win,mode='valid')

        pos_stim = np.convolve(stim[order],conv_win,mode='valid')
        return pos_stim,c_E,c_absE

    def quick_view_results(self,stim,inferred_ori,lwin=1000,sav_name=0):
        d_out,E_out,aE_out = self.quick_view_sb(stim,inferred_ori,l_conv=lwin)
        plt.subplot(221)
        plt.plot(d_out,E_out)
        plt.title('Serial Bias')
        plt.ylabel('Bias (deg)')
        plt.xlabel('$Correct_0 - Correct_{-1}$')
        plt.subplot(222)
        plt.plot(d_out,aE_out)
        plt.ylabel('|Error| (deg)')
        plt.xlabel('$Correct_0 - Correct_{-1}$')

        vis_s,vis_E,vis_aE = self.quick_view_cb(stim,inferred_ori,l_conv=lwin)
        plt.subplot(223)
        plt.plot(vis_s,vis_E)
        plt.title('Cardinal Bias')
        plt.ylabel('Bias (deg)')
        plt.xlabel('Orientation Stim (deg)')
        plt.subplot(224)
        plt.plot(vis_s,vis_aE)
        plt.ylabel('|Error| (deg)')
        plt.xlabel('Orientation Stim (deg)')
        plt.tight_layout()
        if sav_name:
            plt.savefig(sav_name)
        plt.show()
