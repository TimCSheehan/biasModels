import numpy as np
from math import factorial
pi = np.pi

M=20 # neurons 
psi = np.linspace(-pi,pi,M) # neuron selectivity
gamma=.4
kappa= 2.5

def gauss(x,mu=0,sig=1): return 1/(np.sqrt(2*pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))

def angle(s_0,s_1):
    d = s_0-s_1
    d[np.abs(d)>90]-= 180*np.sign(d[np.abs(d)>90])
    return d

n_view = 180
theta = np.linspace(-pi,pi-(2*pi/n_view),n_view)
ori = np.arange(n_view)
p_theta = 2 - np.abs(np.sin(theta)) # prior distribution

D = np.cumsum(p_theta) # CDF
D/=D[-1]
D = D*2*pi-pi

n_spks_sim = np.arange(20) 
def fs(stim): 
    out = np.ones((len(stim),M))
    Du = D[stim]
    for i in range(len(stim)):
        foo = np.exp(kappa*np.cos(Du[i]-psi)-1)
        out[i,:]=  foo
    out*=gamma/M
    return out

# poisson expected n spikes given rate
def pos_exp(theta):
    resp_exp=[]
    for ni in n_spks_sim:
        resp_exp.append(np.array(fs(theta)**ni/factorial(ni) * np.exp(-fs(theta))))
    return np.array(resp_exp)

def run_model(N=5000,getLL=0):
    
    stim = np.random.randint(0,180,N)
    
    spks=np.random.poisson(fs(stim)) # generated spikes
    n_spks = np.sum(spks,1)
    spks[n_spks==0]=np.random.poisson(fs(stim[n_spks==0])) # generated spikes
    n_spks = np.sum(spks,1)
    spks[n_spks==0]=np.random.poisson(fs(stim[n_spks==0])) # generated spikes
    n_spks = np.sum(spks,1)
    
    exp_out = pos_exp(ori) # prior expectation for # of spikes
       
    # get log-likelihood for each stimuli! 
    ll = np.zeros((N,180))
    for s in range(N):
        for i in range(M):
            this_ll = exp_out[spks[s][i],:,i]
            ll[s,:] += np.log(this_ll)
    ll = np.exp(ll)
    dec_stim = np.argmax(ll,1)
    
    stim = stim[n_spks>0]
    dec_stim = dec_stim[n_spks>0]
    print('Throwing away %d trials with no spikes' %sum(n_spks==0))
    E = angle(dec_stim,stim)
    if getLL:
        return ll
    return stim, dec_stim,E
    
def quick_vis(stim,dec_stim=None,E=None,l_conv=100):
    if type(stim) is tuple:
        stim,dec_stim,E = stim
    E = angle(dec_stim,stim)
    
    conv_win = gauss(np.linspace(-2,2,l_conv))
    conv_win/=np.sum(conv_win)
    order = np.argsort(stim)
    c_absE = np.convolve(np.abs(E[order]),conv_win,mode='valid')
    c_E = np.convolve(E[order],conv_win,mode='valid')
    
    pos_stim = np.convolve(stim[order],conv_win,mode='valid')
    return pos_stim,c_E,c_absE

