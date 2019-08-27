import numpy as np

n_step = 180 # 
s_0 = np.linspace(0,180,n_step)
N = 5000 # n trials
pi = np.pi
kappa = 5 # concentration of sensory error
sensoryNoise=10

def angle(s_0,s_1):
    d = s_0-s_1
    d[np.abs(d)>90]-= 180*np.sign(d[np.abs(d)>90])
    return d

def c(s_0,s_1=90,lam=10,gamma=2): ## eq 7
    full = np.exp(-1/(2*lam**2)*np.abs(angle(s_0,s_1))**gamma)
    return full/np.sum(full)

def cf(s_0,s_1=90,lam=10,gamma=2,p_same=0.9): ## eq 6
    return p_same*c(s_0,s_1,lam,gamma) + (1-p_same)*(1/n_step)

def pm(s_0,s_1,sensoryNoise): ## eq. 10
    full = np.exp(-1/(2*sensoryNoise**2)*np.abs(angle(s_0,s_1))**2)
    return full/np.sum(full)
def gauss(x,mu=0,sig=1): return 1/np.sqrt(2*pi*sig**2)*np.exp(-((x-mu)**2)/(2*sig**2)) # for convolution


def run_model():
    seq = np.random.randint(0,180,N)
    sensory = np.random.vonmises((seq-90)/90*pi,kappa)*90/pi+90
    inferred_ori = np.zeros(N)
    
    for i in range(N):
        if i ==0:
            prior = cf(s_0,sensory[-1]) # need to start with something
        else:
            prior = cf(s_0,inferred_ori[i-1])
        sensory_estimate = pm(s_0,sensory[i],sensoryNoise)
        ps_m = prior * sensory_estimate
        ps_m /= np.sum(ps_m)
        decoded_ori = np.argmax(ps_m) # could use circ_mean
        inferred_ori[i] = s_0[decoded_ori]
    return seq,inferred_ori

def quick_view(seq,inferred_ori,l_conv=100):
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