# https://github.com/jakegrigsby/deep_control/
# https://arxiv.org/pdf/2006.15134

from awac_rl import AWAC
from armin_utils import utils
from armin_RL.envs import gym


cfg = utils.yaml_to_dict("C:/Users/armin/Documents/GitHub/awac_rl/cfg.yaml")


total_steps = cfg['num_steps_offline'] + cfg['num_steps_online']
done = True

for step in range(total_steps):
    if step > cfg['num_steps_offline']:
        for _ in range(cfg['transitions_per_online_step']):

