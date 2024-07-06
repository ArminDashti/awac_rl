#https://github.com/jakegrigsby/deep_control/
# https://arxiv.org/pdf/2006.15134

from awac_rl import AWAC
from armin_utils import utils
from armin_RL.envs import gym


cfg = utils.yaml_to_dict("C:/Users/armin/Documents/GitHub/awac_rl/cfg.yaml")
env = gym.Make('MountainCarContinuous-v0')

awac = AWAC.AWAC(env=env, cfg=cfg)
awac.run()


