# Import the envs module so that envs register themselves
import policy_based.envs.gym_minigrid.envs

# Import wrappers so it's accessible when installing with pip
import policy_based.envs.gym_minigrid.wrappers

from policy_based.rlf import register_env_interface
from policy_based.envs.gym_minigrid.minigrid_interface import MiniGridInterface, DualLavaInterface


register_env_interface('^MiniGrid-Dual(.*?)$', DualLavaInterface)
register_env_interface('^MiniGrid(?!Dual)(.*?)$', MiniGridInterface)
