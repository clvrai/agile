
from gym.envs.registration import register
from policy_based.rlf import register_env_interface

from policy_based.envs.interfaces.create_env_interface import CreateGameInterface, CreatePlayInterface

register_env_interface('^Create((?!Play).)*$', CreateGameInterface)
register_env_interface('^Create(.*?)Play(.*)?$', CreatePlayInterface)
register_env_interface('^StateCreate(.*?)Play(.*)?$', CreatePlayInterface)

register(
    id='CreateGamePlay-v0',
    entry_point='policy_based.envs.interfaces.create_play:CreatePlay',
)

register(
    id='StateCreateGamePlay-v0',
    entry_point='policy_based.envs.interfaces.create_play:StateCreatePlay',
)
