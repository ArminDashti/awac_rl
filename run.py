from AWAC import AWAC

num_steps_online = 50000
num_steps_offline = 500000
max_episode_steps = 100000
transitions_per_online_step = 1
batch_size = 1024
tau = 0.005
actor_lr = 1e-4
gamma = 0.99
buffer_size = 1000000
eval_interval = 5000
eval_episodes = 10
actor_clip = None
critic_clip = None
name = 'awac_run'
actor_l2 = 0
critic_l2 = 0
target_delay = 2
gradient_updates_per_step = 1
log_std_low = -10
log_std_high = 2 
beta = 1
crr_function = 'binary'
adv_method = 'mean'
adv_method_n = 4


algo = AWAC(params)
algo.run()