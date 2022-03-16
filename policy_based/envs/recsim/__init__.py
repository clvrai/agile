from gym.envs.registration import register
# from value_based.envs import RecSim
from policy_based.rlf import register_env_interface
from policy_based.envs.recsim.recsim_interface import RecSimInterface

register_env_interface('^RecSim-v0$', RecSimInterface)
'''
register(
    id = 'RecSimEnv-v0',
    entry_point = 'value_based.envs.recsim.environments.interest_evolution:RecoEnv'
)
'''
