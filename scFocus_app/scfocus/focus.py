from .environment import Env, ReplayBuffer, train_off_policy
from .model import SAC
import numpy as np
import torch
import tqdm
import time
from scipy.stats import multivariate_normal
from sklearn.preprocessing import minmax_scale
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class focus:
    def __init__(self, f, hidden_dim=128, n=8, max_steps=5, pct_samples=.125, n_states=2, err_scale=1, bins=15, capacity=1e4, actor_lr=1e-4, critic_lr=1e-3, alpha_lr=1e-4, target_entropy=-1, tau=5e-3, gamma=.99, num_episodes=1e3, batch_size=64, res=.05, device=device):
        """
        Initialize the focus.  
  
        Parameters  
        ----------
        f: array like
            Latent space of the orignal data
       
        """
        self.state_d        = (2+bins)*n_states*n
        self.hidden_dim     = hidden_dim
        self.action_d       = 2*n_states*n
        self.action_space   = (f[:,:n_states].min().item(), f[:,:n_states].max().item())
        self.actor_lr       = actor_lr
        self.critic_lr      = critic_lr
        self.alpha_lr       = alpha_lr
        self.target_entropy = target_entropy
        self.tau            = tau
        self.gamma          = gamma
        self.device         = device
        self.capacity       = capacity
        self.ensemble       = []
        self.env            = Env(n, f, max_steps, pct_samples, n_states, err_scale, bins)
        self.memory         = []
        self.max_steps      = max_steps
        self.num_episodes   = num_episodes
        self.minimal_size   = num_episodes/10*max_steps
        self.batch_size     = batch_size
        self.res            = res
        self.fp             = []
        self.r              = []
        self.e              = []

   
    
    def meta_focusing(self, n):
        
        start = time.time()
        for i in range(n):
            self.meta_fit()
            self.focus_fit(10)
        end = time.time()
        print(f'Meta focusing time used: {(end - start):.2f} seconds')
        return self
    
    def meta_fit(self):
        
        start = time.time()
        self.ensemble.append(SAC(self.state_d,
                                 self.hidden_dim,
                                 self.action_d,
                                 self.action_space,
                                 self.actor_lr,
                                 self.critic_lr,
                                 self.alpha_lr,
                                 self.target_entropy,
                                 self.tau,
                                 self.gamma,
                                 self.device)
                          )
        self.memory.append(ReplayBuffer(self.capacity))
        r, e = train_off_policy(self.env, self.ensemble[-1], self.memory[-1], self.num_episodes, self.minimal_size, self.batch_size)
        self.r.append(np.vstack(r).ravel())
        self.e.append(np.vstack(e).ravel())
        end   = time.time()
        print(f'Meta fitting time used: {(end - start):.2f} seconds')
        return self
        
    def focus_fit(self, episodes):
        
        start = time.time()
        episode_weight = []
        with tqdm.tqdm(total=int(episodes), desc='Focus fitting...') as pbar:
            self.weights = None
            for i_episode in range(int(episodes)):
                ls_weights = []
                state = self.env.reset()
                for i in range(self.max_steps):
                    with torch.no_grad():
                        action = self.ensemble[-1].take_action(state)
                    action = action.ravel()
                    mus = action[:int(action.shape[-1]/2)]
                    logstds = action[int(action.shape[-1]/2):]
                    L = self.env.n_states
                    bra_weights = []
                    for j in range(self.env.n):
                        mu = mus[L*j:L*(j+1)]
                        logstd = logstds[L*j:L*(j+1)]
                        std = np.log1p(np.exp(logstd))
                        mn = multivariate_normal(mu, np.diag(self.env.sigma / (1 + np.exp(-std))))
                        weights = minmax_scale(mn.logpdf(self.env.f[:,:self.env.n_states]))
                        bra_weights.append(weights)
                    ls_weights.append(bra_weights)
                    next_state, _, _ = self.env.step(action)
                    state = next_state
                weights = np.array(ls_weights)
                if self.weights is not None:
                    err = np.linalg.norm(weights - self.weights)
                    if err < 3 and i_episode > 2:
                        break
                self.weights = np.array(ls_weights)
                episode_weight.append(ls_weights)
                pbar.update(1)
        fp = np.array(episode_weight)      
        self.fp.append(fp.T.mean(axis=-1).mean(axis=-1))
        end = time.time()
        print(f'Focus fitting time used: {(end - start):.2f} seconds')
        return self   


    def merge_fp2(self):
        
        self.merge_fp()
        self.fp = [np.hstack(self.mfp)]
        self.merge_fp()
        return self
    
    def merge_fp(self):
        
        self.mfp = []
        for fp in self.fp:
            n = int(fp.shape[0] * self.res)
            ord = np.argsort(fp, axis=0)[-n:,:]
            groups = []
            for i in range(fp.shape[1]):
                if any([i in g for g in groups]):
                    continue
                g_ = [i]
                if i != fp.shape[1] - 1:
                    for j in range(i+1, fp.shape[1]):
                        if len(set(ord[:,i]).intersection(set(ord[:,j]))) > .25*n:
                            g_.append(j)
                    groups.append(g_)
                else:
                    groups.append(g_)
            mfp = []
            for g in groups:
                if len(g) > 1:
                    mfp.append(fp[:,g].mean(axis=1)[:,np.newaxis])
                else:
                    mfp.append(fp[:,g])
            mfp = np.hstack(mfp)
            self.mfp.append(mfp)
        return self
        
    def focus_diff(self):
         
        self.entropy = (self.mfp*-np.log(self.mfp)).sum(axis=1)
        self.pseudotime = 1 - minmax_scale(self.entropy)
        return self
        
    
        