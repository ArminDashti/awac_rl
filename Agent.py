import torch
import torch.nn as nn
import numpy as np
from torch import distributions

from armin_utils import utils

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
        
        
class SquashedNormal(distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, cfg, loc, scale):
        self.cfg = cfg
        self.loc = loc
        self.scale = scale

        self.base_dist = distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    

class BetaDist(distributions.transformed_distribution.TransformedDistribution):
    class _BetaDistTransform(distributions.transforms.Transform):
        domain = distributions.constraints.real
        codomain = distributions.constraints.interval(-1.0, 1.0)

        def __init__(self, cache_size=1):
            super().__init__(cache_size=cache_size)

        def __eq__(self, other):
            return isinstance(other, _BetaDistTransform)

        def _inverse(self, y):
            return (y.clamp(-0.99, 0.99) + 1.0) / 2.0

        def _call(self, x):
            return (2.0 * x) - 1.0

        def log_abs_det_jacobian(self, x, y):
            # return log det jacobian |dy/dx| given input and output
            return torch.Tensor([math.log(2.0)]).to(x.device)

    def __init__(self, alpha, beta):
        self.base_dist = pyd.beta.Beta(alpha, beta)
        transforms = [self._BetaDistTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
    
class StochasticActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        state_space_size = self.cfg['state_space_size']
        act_space_size = self.cfg['act_space_size']
        self.log_std_low = self.cfg['log_std_low']
        self.log_std_high = self.cfg['log_std_high']
        hidden_size = self.cfg['hidden_size']
        self.dist_impl = self.cfg['dist_impl']
        
        self.fc1 = nn.Linear(state_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2 * act_space_size)
        self.apply(weight_init)
    
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        
        mu, log_std = out.chunk(2, dim=1)
        
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            
            log_std = self.log_std_low + 0.5 * (self.log_std_high - self.log_std_low) * (log_std + 1)
            std = log_std.exp()
            
            dist = SquashedNormal(cfg, mu, std)
            
        if self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=1)
            dist = BetaDist(alpha, beta)
            
        return dist
    
    

class BigCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        state_space_size = self.cfg['state_space_size']
        act_space_size = self.cfg['act_space_size']
        hidden_size = self.cfg['hidden_size']
        
        self.fc1 = nn.Linear(state_space_size + act_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.apply(weight_init)


    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
    
    
class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actor = StochasticActor(cfg).to(self.cfg['device'])
        self.critic1 = BigCritic(cfg).to(self.cfg['device'])
        self.critic2 = BigCritic(cfg).to(self.cfg['device'])
        
    
    def _to_train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        

    def _to_eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()


    def _process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(utils.device)


    def _process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)
    
    
    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self._process_act(act)
        return act

    
    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self._process_state(state)
            
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
            
        self.actor.train()
        if from_cpu:
            act = self._process_act(act)
        return act


