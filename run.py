import torch
import torch.nn as nn
import numpy as np


class AWAC:
    def __init__(
            agent,
            buffer,
            train_env,
            test_env,
            num_steps_offline=25_000,
            num_steps_online=500_000,
            gradient_updates_per_step=1,
            transitions_per_online_step=1,
            max_episode_steps=100_000,
            actor_per=True,
            batch_size=1024,
            tau=0.005,
            beta=1.0,
            crr_function="binary",
            adv_method="mean",
            adv_method_n=4,
            actor_lr=1e-4,
            critic_lr=1e-4,
            gamma=0.99,
            eval_interval=5000,
            eval_episodes=10,
            warmup_steps=1000,
            actor_clip=None,
            critic_clip=None,
            actor_l2=0.0,
            critic_l2=0.0,
            target_delay=2,
            actor_delay=1,
            save_interval=100_000,
            name="awac_run",
            render=False,
            verbosity=0,
            infinite_bootstrap=True,
            **kwargs
            ):


        self.agent = agent.to(device).train()
        self.target_agent = copy.deepcopy(agent).to(device).train()
        
        self.hard_update(target_agent.critic1, agent.critic1)
        self.hard_update(target_agent.critic2, agent.critic2)

        
        self.critic_optimizer = torch.optim.Adam(
            chain(agent.critic1.parameters(), agent.critic2.parameters()),
            lr=critic_lr,
            weight_decay=critic_l2,
            betas=(0.9, 0.999),
            )
        
        self.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr, weight_decay=actor_l2, betas=(0.9, 0.999))
        self.adv_estimator = AdvantageEstimator(agent.actor, [agent.critic1, agent.critic2], method=adv_method, n=adv_method_n)
        self.adv_filter = AdvEstimatorFilter(adv_estimator, crr_function, beta=beta)


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
        
    
    def learn(self,
        buffer=buffer,
        target_agent=target_agent,
        agent=agent,
        adv_filter=adv_filter,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        batch_size=batch_size,
        gamma=gamma,
        critic_clip=critic_clip,
        actor_clip=actor_clip,
        update_policy=step % actor_delay == 0,
        actor_per=actor_per,):
        
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
                target_agent.critic2(next_state_batch, action_s1),
            )
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
            torch.nn.utils.clip_grad_norm_(
                chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip
            )
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
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
            actor_optimizer.step()
    
    
    def evaluate(self):
        pass
    
    
    
    def run(self):

        total_steps = num_steps_offline + num_steps_online
        steps_iter = range(total_steps)
        if verbosity:
            steps_iter = tqdm.tqdm(steps_iter)

        done = True
        for step in steps_iter:
            if step > num_steps_offline:
                for _ in range(transitions_per_online_step):
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

        for _ in range(gradient_updates_per_step):
            self.learn(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                adv_filter=adv_filter,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
                update_policy=step % actor_delay == 0,
                actor_per=actor_per)

            # move target model towards training model
            if step % target_delay == 0:
                self.soft_update(target_agent.critic1, agent.critic1, tau)
                self.soft_update(target_agent.critic2, agent.critic2, tau)

        if (step % eval_interval == 0) or (step == total_steps - 1):
            mean_return = run.evaluate_agent(agent, test_env, eval_episodes, max_episode_steps, render)


    return agent
