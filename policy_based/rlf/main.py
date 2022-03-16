from policy_based.rlf.rl.rl import load_from_checkpoint, train, init_torch, create_algo, create_rollout_buffer
import torch
from policy_based.rlf.rl.logger import Logger
from policy_based.rlf.rl.checkpointer import Checkpointer
from policy_based.rlf.rl.envs import make_vec_envs
from policy_based.rlf.rl.evaluation import full_eval
from value_based.commons.launcher import launch_embedding, launch_env


class RunSettings(object):
    def __init__(self):
        pass

    def get_policy_type(self):
        raise NotImplemented()

    def get_train_env_trans(self, args, task_id=None):
        pass

    def get_test_env_trans(self, args, task_id=None):
        pass

    def get_args(self):
        raise ValueError('not implemented')

    def get_test_args(self):
        raise ValueError('not implemented')

    def get_set_args(self):
        raise ValueError('not implemented')


def get_num_updates(args):
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.lr_env_steps is None:
        args.lr_env_steps = args.num_env_steps
    lr_updates = int(
        args.lr_env_steps) // args.num_steps // args.num_processes
    return num_updates, lr_updates


def run_policy(run_settings):
    '''
        First function to be called from main.py while training or evaluating policy
    '''
    args = run_settings.get_args()
    test_args = run_settings.get_test_args()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    test_args.device = args.device

    log = Logger(run_settings.get_set_args())
    log.set_prefix(args)
    log.set_prefix(test_args)

    checkpointer = Checkpointer(args)

    init_torch(args)
    args.env_trans_fn = run_settings.get_train_env_trans()
    args.test_env_trans_fn = run_settings.get_test_env_trans()

    test_args.env_trans_fn = run_settings.get_train_env_trans()
    test_args.test_env_trans_fn = run_settings.get_test_env_trans()
    args.trajectory_len = None
    test_args.trajectory_len = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.device,
                         False, args.env_trans_fn, args)

    # Decide between DistanceMethod (output action embeddings) and
    # MainMethod (input action embeddings) style of architectures
    policy_class = run_settings.get_policy_type()

    policy = policy_class(args, envs.observation_space, envs.action_space)

    action_space = envs.action_space if not policy.is_slate else policy.action_space

    if checkpointer.should_load():
        load_from_checkpoint(policy, envs, checkpointer)

    updater = create_algo(policy, args) # Simply PPO

    rollouts = create_rollout_buffer(policy, envs,
                                     action_space,
                                     args)

    if args.eval_only:
        full_eval(envs, policy, log, checkpointer, args)
        return

    log.watch_model(policy)

    start_update = 0
    if args.resume:
        updater.load_resume(checkpointer)
        policy.load_resume(checkpointer)
        start_update = checkpointer.get_key('step')

    num_updates, lr_updates = get_num_updates(args)

    train(envs, rollouts, policy, updater, log, start_update,
          num_updates, lr_updates, args, test_args, checkpointer)
