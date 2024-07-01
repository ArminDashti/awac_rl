from awac_rl import AWAC
from armin_utils import utils


cfg = utils.yaml_to_dict("C:/Users/armin/Documents/GitHub/awac_rl/cfg.yaml")
awac = AWAC.AWAC(cfg)
awac.run()


