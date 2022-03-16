
def str2bool(v):
    return v.lower() == 'true'


def add_args(parser):
    parser.add_argument('--use_double', type=str2bool, default=False)
    parser.add_argument('--use_dist_double', type=str2bool, default=True)
    parser.add_argument('--use_mean_entropy', type=str2bool, default=None)

    parser.add_argument('--sync_host', type=str, default='')
    parser.add_argument('--sync_port', type=str, default='22')

    ########################################################
    # Distribution args
    ########################################################
    parser.add_argument('--use_beta', type=str2bool, default=True)
    parser.add_argument('--use_gaussian_distance', type=str2bool, default=True)
    parser.add_argument('--softplus', type=str2bool, default=True)
    parser.add_argument('--fixed_variance', type=str2bool, default=False)

    ########################################################
    ## PPO / A2C specific args
    ########################################################

    ## **
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    ## **
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--rl_use_radam', type=str2bool, default=False)
    parser.add_argument(
        '--lr_env_steps', type=int, default=None, help='only used for lr schedule')

    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    ## **
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    ## **
    parser.add_argument(
        '--use_gae',
        type=str2bool,
        default=True,
        help='use generalized advantage estimation')
    ## **
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    ## **
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=None,
        help='entropy term coefficient (old default: 0.01)')
    ## **
    parser.add_argument(
        '--value_loss_coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    ## **
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    ## **
    parser.add_argument(
        '--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument(
        '--cuda_deterministic',
        type=str2bool,
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    ## **
    parser.add_argument(
        '--num_processes',
        type=int,
        default=None,
        help='how many training CPU processes to use (default: 32)')
    parser.add_argument(
        '--eval_num_processes',
        type=int,
        default=None,
        help='how many training CPU processes to use (default: None)')
    ## **
    parser.add_argument(
        '--num_steps',
        type=int,
        default=None,
        help='number of forward steps in A2C/PPO (old default: 128)')
    ## **
    parser.add_argument(
        '--ppo_epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    ## **
    parser.add_argument(
        '--num_mini_batch',
        type=int,
        default=4,
        help='number of batches for ppo (default: 4)')
    ## **
    parser.add_argument(
        '--clip_param',
        type=float,
        default=0.1,
        help='ppo clip parameter (old default: 0.2)')
    parser.add_argument(
        '--prefix',
        default='',
        help='prefix of log dir')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 1)')
    parser.add_argument(
        '--save_interval',
        type=int,
        default=50,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--backup',
        type=str,
        default=None,
        help='whether to backup or not. Specify your username (default: None)')
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    ## **
    parser.add_argument(
        '--num_env_steps',
        type=int,
        default=None,
        help='number of environment steps to train (default: 1e8)')
    parser.add_argument(
        '--env_name',
        default='PongNoFrameskip_v4',
        help='environment to train on (default: PongNoFrameskip_v4)')
    parser.add_argument(
        '--log_dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save_dir',
        default='./policy_based/data/trained_models/',
        help='directory to save agent trained models (default: ./policy_based/data/trained_models/)')

    parser.add_argument(
        '--load_file',
        default='',
        help='.pt weights file')
    parser.add_argument(
        '--resume',
        default=False,
        type=str2bool,
        help='Resume training')
    parser.add_argument(
        '--no_cuda',
        default=False,
        type=str2bool,
        help='disables CUDA training')
    ## **
    parser.add_argument(
        '--use_proper_time_limits',
        default=False,
        type=str2bool,
        help='compute returns taking into account time limits')
    ## **
    parser.add_argument(
        '--recurrent_policy',
        type=str2bool,
        default=False,
        help='use a recurrent policy')
    ## **
    parser.add_argument(
        '--use_linear_lr_decay',
        type=str2bool,
        default=True,
        help='use a linear schedule on the learning rate')

