import numpy as np
import matplotlib.pyplot as plt
from math import factorial
pi = np.pi

## CB parameters
M=20 # neurons 
psi = np.linspace(-pi,pi,M) # neuron selectivity
gamma=.4
kappaCB= 2.5

n_view = 180
theta = np.linspace(-pi,pi-(2*pi/n_view),n_view)
ori = np.arange(n_view)
p_theta = 2 - np.abs(np.sin(theta)) # prior distribution

D = np.cumsum(p_theta) # CDF
D/=D[-1]
D = D*2*pi-pi

def gauss(x,mu=0,sig=1): return 1/(np.sqrt(2*pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))
def angle(s_0,s_1):
    d = s_0-s_1
    d[np.abs(d)>90]-= 180*np.sign(d[np.abs(d)>90])
    return d

n_spks_sim = np.arange(20) 
def fs(stim): 
    out = np.ones((len(stim),M))
    Du = D[stim]
    for i in range(len(stim)):
        foo = np.exp(kappaCB*np.cos(Du[i]-psi)-1)
        out[i,:]=  foo
    out*=gamma/M
    return out

# poisson expected n spikes given rate
def pos_exp(theta):
    resp_exp=[]
    for ni in n_spks_sim:
        resp_exp.append(np.array(fs(theta)**ni/factorial(ni) * np.exp(-fs(theta))))
    return np.array(resp_exp)


## SB parameters
n_step = 180 # 
s_0 = np.linspace(0,180,n_step)
kappaSB = 5 # concentration of sensory error
sensoryNoise=10

def c(s_0,s_1=90,lam=10,gamma=2): ## eq 7
    full = np.exp(-1/(2*lam**2)*np.abs(angle(s_0,s_1))**gamma)
    return full/np.sum(full)

def cf(s_0,s_1=90,lam=10,gamma=2,p_same=0.9): ## eq 6
    return p_same*c(s_0,s_1,lam,gamma) + (1-p_same)*(1/n_step)

def pm(s_0,s_1,sensoryNoise): ## eq. 10
    full = np.exp(-1/(2*sensoryNoise**2)*np.abs(angle(s_0,s_1))**2)
    return full/np.sum(full)

def run_model(N=5000):
    
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
    
    ## LL is sensory...
    stim = stim[n_spks>0]
    dec_stim = dec_stim[n_spks>0]
    print('Throwing away %d trials with no spikes' %sum(n_spks==0))
    sensory = ll[n_spks>0,:]
    
    N = np.shape(sensory)[0]
    inferred_ori = np.zeros(N)
    for i in range(N):
        if i ==0:
            prior = cf(s_0,stim[-1]) # need to start with something
        else:
            prior = cf(s_0,inferred_ori[i-1])

        sensory_estimate = sensory[i,:]
        ps_m = prior * sensory_estimate
        ps_m /= np.sum(ps_m)
        decoded_ori = np.argmax(ps_m) # could use circ_mean
        inferred_ori[i] = s_0[decoded_ori]

    E = angle(inferred_ori,stim)
    return stim, inferred_ori,E


def quick_view_sb(seq,inferred_ori,l_conv=100):
    E = angle(inferred_ori,seq)
    d = seq[1:] - seq[:-1]
    d[np.abs(d)>90] -=180*np.sign(d[np.abs(d)>90])
    Eu = E[1:]
    order = np.argsort(d)
    
    conv_win = gauss(np.linspace(-2,2,l_conv))
    conv_win/=np.sum(conv_win)
    
    E_out = np.convolve(Eu[order],conv_win,mode='valid')
    aE_out = np.convolve(np.abs(Eu[order]),conv_win,mode='valid')
    d_out = np.convolve(d[order],conv_win,mode='valid')

    return d_out,E_out,aE_out

def quick_view_cb(stim,dec_stim=None,E=None,l_conv=100):
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

def quick_view_results(stim,inferred_ori,lwin=1000):
    d_out,E_out,aE_out = quick_view_sb(stim,inferred_ori,l_conv=lwin)
    plt.subplot(221)
    plt.plot(d_out,E_out)
    plt.title('Serial Bias')
    plt.ylabel('Bias (deg)')
    plt.xlabel('$Correct_0 - Correct_{-1}$')
    plt.subplot(222)
    plt.plot(d_out,aE_out)
    plt.ylabel('|Error| (deg)')
    plt.xlabel('$Correct_0 - Correct_{-1}$')
    # plt.tight_layout()
    # plt.show()

    vis_s,vis_E,vis_aE = quick_view_cb(stim,inferred_ori,l_conv=lwin)
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
    plt.show()
