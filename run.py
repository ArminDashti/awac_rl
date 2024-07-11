# https://github.com/jakegrigsby/deep_control/
# https://arxiv.org/pdf/2006.15134

from awac_rl import AWAC
from armin_utils import utils
from armin_RL.envs import gym
from armin_RL imprt utils as rl_utils
from awac_rl.agent import Agent
device = 'cpu'

cfg = utils.yaml_to_dict("C:/Users/armin/Documents/GitHub/awac_rl/cfg.yaml")
agent = = Agent(cfg)
target_agent = copy.deepcopy(agent)

rl_utils.hard_update(target_agent.critic1, agent.critic1)
rl_utils.hard_update(target_agent.critic2, agent.critic2)

critic_optimizer = torch.optim.Adam(
    chain(agent.critic1.parameters(), agent.critic2.parameters()),
    lr=0.001,
    weight_decay=0.9,
    betas=(0.9, 0.999))
        
actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=0.001, weight_decay=0.9, betas=(0.9, 0.999))
     
adv_estimator = AdvEstimator(cfg, agent.actor, 
                             [agent.critic1, agent.critic2], 
                             method=cfg['adv_method'], 
                             n=cfg['adv_method_n'])
        
adv_filter = AdvEstimatorFilter(adv_estimator, crr_function, beta=cfg['beta'])



total_steps = cfg['num_steps_offline'] + cfg['num_steps_online']
done = True

for step in range(total_steps):
    if step > cfg['num_steps_offline']:
        for _ in range(cfg['transitions_per_online_step']):
            if done:
                state = train_env.reset()
                steps_this_ep = 0
                done = False
            action = agent.sample_action(state)
            next_state, reward, done, info = train_env.step(action)
            if infinite_bootstrap:
                if steps_this_ep + 1 == max_episode_steps:
                    done = False
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True
                
    for _ in rangegradient_updates_per_step):
        if actor_per:
            actor_batch, *_ = buffer.sample(batch_size)
            critic_batch, priority_idxs = buffer.sample_uniform(batch_size)
        else:
            batch = buffer.sample(batch_size)
            actor_batch = batch
            critic_batch = batch
        
        agent.train()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = critic_batch
        state_batch = state_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        done_batch = done_batch.to(device)
        
        with torch.no_grad():
            action_dist_s1 = agent.actor(next_state_batch)
            action_s1 = action_dist_s1.rsample()
            logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
            
            
            target_action_value_s1 = torch.min(
                target_agent.critic1(next_state_batch, action_s1),
                target_agent.critic2(next_state_batch, action_s1))
            
            td_target = reward_batch + gamma * (1.0 - done_batch) * target_action_value_s1
            
        # update critics
        agent_critic1_pred = agent.critic1(state_batch, action_batch)
        agent_critic2_pred = agent.critic2(state_batch, action_batch)
        td_error1 = td_target - agent_critic1_pred
        td_error2 = td_target - agent_critic2_pred
        critic_loss = 0.5 * (td_error1 ** 2 + td_error2 ** 2)
        critic_loss = critic_loss.mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        
        if critic_clip:
            torch.nn.utils.clip_grad_norm_(chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip)
        critic_optimizer.step()
        
        
        if actor_per:
            with torch.no_grad():
                adv = adv_filter.adv_estimator(state_batch, action_batch)
            new_priorities = (F.relu(adv) + 1e-5).cpu().detach().squeeze(1).numpy()
            buffer.update_priorities(priority_idxs, new_priorities)
            
        if update_policy:
            state_batch, *_ = actor_batch
            state_batch = state_batch.to(device)

            dist = agent.actor(state_batch)
            actions = dist.sample()
            logp_a = dist.log_prob(actions).sum(-1, keepdim=True)
                
            with torch.no_grad():
                filtered_adv = adv_filter(state_batch, actions)
                    
            actor_loss = -(logp_a * filtered_adv).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
                
            if actor_clip:
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), actor_clip)
                    
            actor_optimizer.step()
            
            
    if step % target_delay == 0:
        rl_utils.soft_update(self.target_agent.critic1, self.agent.critic1, cfg['tau'])
        rl_utils.soft_update(self.target_agent.critic2, self.agent.critic2, cfg['tau'])
