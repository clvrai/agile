import argparse
from policy_based.rlf.args import add_args, str2bool
from value_based.commons.args import recsim_parser
from value_based.commons.args import add_recsim_args

import torch

def set_if_none(args, var, value):
    vars(args)[var] = value if vars(args)[var] is None else vars(args)[var]

def set_none(args, var, value):
    vars(args)[var] = None


def grid_args(args):
    if args.env_name.startswith('MiniGrid-Dual'):
        assert not args.render_info_grid
        args.gt_embs = True
        args.action_set_size = args.grid_available_upto + 2 - 4 * args.grid_remove_diagonal
        args.test_action_set_size = args.grid_available_upto + 2 - 4 * args.grid_remove_diagonal
        set_if_none(args, 'env_total_train_actions', 13)
        set_if_none(args, 'num_processes', 64)
        set_if_none(args, 'num_steps', 512)
        set_if_none(args, 'eval_num_processes', 4)
        set_if_none(args, 'grid_scaled_subgoal', True)
        set_if_none(args, 'grid_state_visitation', True)



    if args.load_embeddings_file is None and args.save_embeddings_file is None:
        args.load_embeddings_file = 'gw_st'

    set_if_none(args, 'exp_type', 'rnd')

    set_if_none(args, 'onehot_state', True)
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'action_set_size', 50)
    set_if_none(args, 'test_action_set_size', 50)
    set_if_none(args, 'o_dim', 16)
    set_if_none(args, 'z_dim', 16)
    set_if_none(args, 'num_env_steps', 10000000)
    set_if_none(args, 'lr_env_steps', 40000000)

    if args.distance_based:
        set_if_none(args, 'num_steps', 256)
        set_if_none(args, 'use_mean_entropy', False)
    else:
        set_if_none(args, 'num_steps', 512)

    set_if_none(args, 'entropy_coef', 5e-2)
    set_if_none(args, 'num_eval', 20)
    set_if_none(args, 'eval_interval', 25)

    set_if_none(args, 'load_all_data', True)

    args.up_to_option = '5'
    args.grid_flatten = True
    args.no_frame_stack = True



def grid_play_args(args):

    set_if_none(args, 'exp_type', 'rnd')

    args.up_to_option = '5'
    args.trajectory_len = 6
    if args.train_embeddings and args.n_trajectories == 1024:
        args.n_trajectories = 672

    set_if_none(args, 'emb_epochs', 10000)
    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'o_dim', 16)
    set_if_none(args, 'z_dim', 16)
    set_if_none(args, 'onehot_state', True)
    set_if_none(args, 'emb_save_interval', 100)

    args.num_eval = 20
    args.action_random_sample = False
    set_if_none(args, 'load_all_data', True)
    if args.prefix == '':
        args.prefix = 'gw_train_embs'
    args.play_grid_size = 80
    args.grid_flatten = False
    args.no_frame_stack = True


def create_args(args):
    set_if_none(args, 'num_processes', 32)

    set_if_none(args, 'exp_type', 'NewMain')
    set_if_none(args, 'split_type', 'gran_1')
    set_if_none(args, 'action_set_size', 25)
    set_if_none(args, 'test_action_set_size', 25)
    args.num_candidates = args.action_set_size
    set_if_none(args, 'entropy_coef', 5e-3)
    set_if_none(args, 'num_env_steps', 100000000)
    set_if_none(args, 'lr_env_steps', 100000000)

    set_if_none(args, 'create_num_variable_activators', 2)

    if args.distance_based:
        set_if_none(args, 'num_steps', 256)
        set_if_none(args, 'use_mean_entropy', True)
    else:
        set_if_none(args, 'num_steps', 384)
    set_if_none(args, 'num_eval', 20)

    set_if_none(args, 'o_dim', 128)
    set_if_none(args, 'z_dim', 128)
    if args.load_embeddings_file is None and args.save_embeddings_file is None:
        args.load_embeddings_file = 'create_g1_len7'

    if args.play_env_name is not None:
        if args.play_env_name.startswith('State'):
            args.load_all_data = True
        else:
            set_if_none(args, 'image_resolution', 48)
            args.load_all_data = False
            args.hidden_dim_traj = 128
            args.encoder_dim = 128
            args.hidden_dim_option = 128
    else:
        set_if_none(args, 'image_resolution', 64)

def create_play_args(args):
    set_if_none(args, 'exp_type', 'NewMain')
    if args.split_type is None:
        args.split_type = 'full_clean'
    if args.env_name.startswith('State'):
        set_if_none(args, 'emb_epochs', 10000)
        if args.train_embeddings and args.n_trajectories == 1024:
            args.n_trajectories = 464
        args.emb_batch_size = 128
        args.load_all_data = True
        args.prefix = 'StateCreate'
    else:
        set_if_none(args, 'emb_epochs', 5000)
        if args.train_embeddings and args.n_trajectories == 1024:
            args.n_trajectories = 45
        set_if_none(args, 'image_resolution', 48)

        args.emb_batch_size = 32
        args.load_all_data = False
        args.hidden_dim_traj = 128
        args.encoder_dim = 128
        args.hidden_dim_option = 128
        args.prefix = 'VideoCreate'

    set_if_none(args, 'num_processes', 32)
    set_if_none(args, 'o_dim', 128)
    set_if_none(args, 'z_dim', 128)
    args.action_random_sample = False

def recsim_args(args):
    args.gt_embs = True
    add_recsim_args(args)
    set_if_none(args, 'action_set_size', 25)
    set_if_none(args, 'test_action_set_size', 25)
    args.num_candidates = args.action_set_size
    args.num_processes = args.batch_step_size
    set_if_none(args, 'eval_num_processes', args.batch_step_size)
    args.batch_step_size = 1
    set_if_none(args, 'num_env_steps', 40000000)
    set_if_none(args, 'lr_env_steps', 40000000)
    args.num_steps *= args.slate_size
    args.recsim_if_mdp = True

    set_if_none(args, 'num_eval', 20)
    set_if_none(args, 'entropy_coef', 1e-2)
    set_if_none(args, 'num_steps', 256)
    set_if_none(args, 'eval_interval', 20)

    args.no_frame_stack = True


def env_specific_args(args):
    if args.env_name.startswith('MiniGrid-Empty'):
        grid_play_args(args)
    elif args.env_name.startswith('MiniGrid'):
        grid_args(args)
    elif args.env_name.startswith('CreateLevel'):
        create_args(args)
    elif 'Create' in args.env_name:
        create_play_args(args)
    elif args.env_name.startswith('RecSim'):
        recsim_args(args)

def method_specific_args(args):
    args.gnn_ppo = False
    if args.method_name.lower() == 'relational':
        args.action_set_summary = True
        args.action_feature_categorical = True
        args.mask_categorical = False
        args.if_visualise_attention = True
        args.separate_critic_action_set_summary = True
        args.separate_critic_summary_nodes = True
        args.input_mask_categorical = False
        args.action_summarizer = 'gnn'
    elif args.method_name.lower() == 'summary':
        args.action_set_summary = True
        args.action_feature_categorical = True
        set_if_none(args, 'action_summarizer', 'gnn')
        args.mask_categorical = False
        args.if_visualise_attention = True
        args.separate_critic_action_set_summary = False
        args.input_mask_categorical = False
    elif args.method_name.lower() == 'baseline':
        args.action_set_summary = False
        args.action_feature_categorical = True
        args.mask_categorical = False
        args.if_visualise_attention = False
        args.separate_critic_action_set_summary = False
        args.input_mask_categorical = False
        args.action_summarizer = 'None'
    elif args.method_name.lower() == 'input_mask':
        args.action_set_summary = True
        args.action_feature_categorical = False
        args.mask_categorical = True
        args.action_summarizer = 'None'
        args.if_visualise_attention = False
        args.separate_critic_action_set_summary = False
        args.input_mask_categorical = True
    elif args.method_name.lower() == 'mask':
        args.action_set_summary = False
        args.action_feature_categorical = False
        args.input_mask_categorical = False
        args.mask_categorical = True
        args.action_summarizer = 'None'
        args.if_visualise_attention = False
        args.separate_critic_action_set_summary = False
    else:
        args.gnn_ppo = True
        raise NotImplementedError



def general_args(args):
    set_if_none(args, 'emb_save_interval', 10)
    set_if_none(args, 'num_eval', 5)
    set_if_none(args, 'eval_interval', 50)
    set_if_none(args, 'load_all_data', False)
    set_if_none(args, 'image_resolution', 64)
    set_if_none(args, 'separate_skip', False)
    if args.distance_based:
        set_if_none(args, 'use_mean_entropy', True)
    else:
        set_if_none(args, 'use_mean_entropy', False)


def revert_general_args(args):
    set_none(args, 'emb_save_interval', 10)
    set_none(args, 'num_eval', 5)
    set_none(args, 'eval_interval', 50)
    set_none(args, 'load_all_data', False)
    set_none(args, 'image_resolution', 64)
    set_none(args, 'separate_skip', False)
    set_none(args, 'use_mean_entropy', True)


def fixed_action_settings(args):
    if args.fixed_action_set:
        args.half_tool_ratio = None
        args.action_set_size = None



def get_args(arg_str=None, include_method_specific=True):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--fine_tune', type=str2bool, default=False)
    parser.add_argument('--conditioned_aux', type=str2bool, default=False)
    parser.add_argument('--conditioned_non_linear', type=str2bool, default=False)
    parser.add_argument('--load_best_name', type=str, default=None)

    parser.add_argument('--usage_policy', type=str2bool, default=False)
    parser.add_argument('--usage_loss_coef', type=float, default=1.0)
    parser.add_argument('--normalize_embs', type=str2bool, default=False)
    parser.add_argument('--only_vis_embs', type=str2bool, default=False)
    parser.add_argument('--random_policy', type=str2bool, default=False)
    parser.add_argument('--separate_skip', type=str2bool, default=None)
    parser.add_argument('--pac_bayes', type=str2bool, default=False)
    parser.add_argument('--pac_bayes_delta', type=float, default=0.1)
    parser.add_argument('--complexity_scale', type=float, default=10.0)

    parser.add_argument('--create_len_filter', type=float, default=None)


    ########################################################
    # Action sampling related
    parser.add_argument('--gt_clusters', type=str2bool, default=False,
            help='Only used for Full_clean split')
    parser.add_argument('--strict_gt_clusters', type=str2bool, default=False)
    parser.add_argument('--n_clusters', type=int, default=10)

    parser.add_argument('--analysis_angle', type=int, default=None)
    parser.add_argument('--analysis_emb', type=float, default=None)
    parser.add_argument('--train_mix_ratio', type=float, default=None)
    ########################################################

    ########################################################
    # Entropy Reward related
    parser.add_argument('--reward_entropy_coef', type=float, default=0.0)
    ########################################################

    parser.add_argument('--gran_factor', type=float, default=1.0)

    parser.add_argument('--distance_effect', type=str2bool, default=False)
    parser.add_argument('--distance_sample', type=str2bool, default=False)
    parser.add_argument('--only_pos', type=str2bool, default=False)


    ########################################################
    # Action set generation args
    parser.add_argument('--action_seg_loc', type=str, default='./policy_based/data/action_segs')
    parser.add_argument('--action_random_sample', type=str2bool,
            default=True, help='Randomly sample actions or not.')

    parser.add_argument('--action_set_size', type=int, default=None)
    parser.add_argument('--test_action_set_size', type=int, default=None)
    parser.add_argument('--play_data_folder', type=str, default='./policy_based/method/embedder/data')
    parser.add_argument('--emb_model_folder', type=str, default='./policy_based/data/embedder/trained_models/')
    parser.add_argument('--create_play_len', type=int, default=7)
    parser.add_argument('--create_play_run_steps', type=int, default=3)
    parser.add_argument('--create_play_colored', type=str2bool, default=False)
    parser.add_argument('--create_play_fixed_tool', type=str2bool, default=False)
    parser.add_argument('--play_env_name', type=str, default=None)
    parser.add_argument('--image_resolution', type=int, default=None,
        help='If lower image resolution is to be used in play data')
    parser.add_argument('--image_mask', type=str2bool, default=True)

    parser.add_argument('--input_channels', type=int, default=1,
        help='No. of input channels for HTVAE')

    parser.add_argument('--train_split', type=str2bool, default=None)
    parser.add_argument('--training_action_set_size', type=int, default=None)
    parser.add_argument('--test_split', type=str2bool, default=False)
    parser.add_argument('--eval_split', type=str2bool, default=False)
    parser.add_argument('--eval_split_ratio', type=float, default=0.5,
            help='Fraction of action set that is eval')

    parser.add_argument('--both_train_test', type=str2bool, default=False)
    parser.add_argument('--fixed_action_set', type=str2bool, default=False)

    parser.add_argument('--load_fixed_action_set', type=str2bool, default=False,
        help='For nearest neighbor lookup of discrete policy at evaluation')

    parser.add_argument('--num_z', type=int, default=1)

    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--decay_clipping', type=str2bool, default=False)


    ########################################################
    # Method specific args
    ########################################################
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--action_proj_dim', type=int, default=1)
    parser.add_argument('--load_only_actor', type=str2bool, default=True)
    parser.add_argument('--sample_k', type=str2bool, default=False)
    parser.add_argument('--do_gumbel_softmax', type=str2bool, default=False)
    parser.add_argument('--discrete_fixed_variance', type=str2bool, default=False)
    parser.add_argument('--use_batch_norm', type=str2bool, default=False)

    parser.add_argument('--gt_embs', type=str2bool, default=False)


    parser.add_argument(
        '--cont_entropy_coef',
        type=float,
        default=1e-1,
        help='scaling continuous entropy coefficient term further (default: 0.1)')

    # Discrete Beta settings
    parser.add_argument('--discrete_beta', type=str2bool, default=False)
    parser.add_argument('--max_std_width', type=float, default=3.0)
    parser.add_argument('--constrained_effects', type=str2bool, default=True)
    parser.add_argument('--bound_effect', type=str2bool, default=False)

    parser.add_argument('--emb_margin', type=float, default=1.1)

    parser.add_argument('--nearest_neighbor', type=str2bool, default=False)
    parser.add_argument('--combined_dist', type=str2bool, default=False)
    parser.add_argument('--combined_add', type=str2bool, default=False)

    parser.add_argument('--no_frame_stack', type=str2bool, default=False)

    parser.add_argument('--dist_hidden_dim', type=int, default=64)
    parser.add_argument('--dist_linear_action', type=str2bool, default=True)
    parser.add_argument('--dist_non_linear_final', type=str2bool, default=True)

    parser.add_argument('--exp_logprobs', type=str2bool, default=False)
    parser.add_argument('--kl_pen', type=float, default=None)
    parser.add_argument('--cat_kl_loss', type=float, default=None)

    parser.add_argument('--reparam', type=str2bool, default=True)
    parser.add_argument('--no_var', type=str2bool, default=False)
    parser.add_argument('--z_mag_pen', type=float, default=None)

    # Distance Model specific
    parser.add_argument('--distance_based', type=str2bool, default=False)
    parser.add_argument('--cosine_distance', type=str2bool, default=False)

    # Gridworld specific
    parser.add_argument('--up_to_option', type=str, default=None)
    parser.add_argument('--no_diag', type=str2bool, default=True)
    parser.add_argument('--option_penalty', type=float, default=0.0)
    parser.add_argument('--grid_flatten', type=str2bool, default=True)
    parser.add_argument('--grid_playing', type=str2bool, default=False)
    parser.add_argument('--play_grid_size', type=int, default=80)
    parser.add_argument('--onehot_state', type=str2bool, default=None)

    parser.add_argument('--not_upto', type=str2bool, default=True)
    parser.add_argument('--orig_crossing_env', type=str2bool, default=False)
    parser.add_argument('--max_grid_steps', type=int, default=50)
    parser.add_argument('--grid_subgoal', type=str2bool, default=True)
    parser.add_argument('--grid_fixed_rivers', type=str2bool, default=False)
    parser.add_argument('--grid_safe_wall', type=str2bool, default=True)

    # Varying Grid
    parser.add_argument('--grid_dig_env', type=str2bool, default=True)
    parser.add_argument('--grid_scaled_subgoal', type=str2bool, default=True)
    parser.add_argument('--grid_render_last_twice', type=str2bool, default=True)
    parser.add_argument('--grid_state_visitation', type=str2bool, default=True)
    parser.add_argument('--grid_dig_reward', type=str2bool, default=True)
    parser.add_argument('--grid_neg_lava_reward', type=float, default=0.)
    parser.add_argument('--grid_lava_dead_count', type=int, default=1)
    parser.add_argument('--grid_append_dig_availability', type=str2bool, default=False)
    parser.add_argument('--grid_available_upto', type=int, default=9)
    parser.add_argument('--grid_remove_diagonal', type=str2bool, default=True)
    parser.add_argument('--grid_agent_pos', type=str2bool, default=True)


    # Video specific
    parser.add_argument('--vid_dir', type=str, default='./policy_based/data/vids')
    parser.add_argument('--attention_dir', type=str, default='./policy_based/data/attention')
    parser.add_argument('--obs_dir', type=str, default='./policy_based/data/obs')
    parser.add_argument('--should_render_obs', type=str2bool, default=False)
    parser.add_argument('--result_dir', type=str, default='./policy_based/data/results')
    parser.add_argument('--vid_fps', type=float, default=5.0)
    parser.add_argument('--eval_only', type=str2bool, default=False)
    parser.add_argument('--evaluation_mode', type=str2bool, default=False)

    parser.add_argument('--high_render_dim', type=int, default=256, help='Dimension to render evaluation videos at')
    parser.add_argument('--high_render_freq', type=int, default=50)
    parser.add_argument('--no_test_eval', type=str2bool, default=False)
    parser.add_argument('--num_render', type=int, default=None)
    parser.add_argument('--num_eval', type=int, default=None)
    parser.add_argument('--render_info_grid', type=str2bool, default=False)
    parser.add_argument('--deterministic_policy', type=str2bool, default=False)

    parser.add_argument('--debug_render', type=str2bool, default=False)
    parser.add_argument('--render_gifs', type=str2bool, default=False)
    parser.add_argument('--verbose_eval', type=str2bool, default=True)


    # CREATE specific
    parser.add_argument('--half_tools', type=str2bool, default=True)
    parser.add_argument('--half_tool_ratio', type=float, default=0.5)
    parser.add_argument('--marker_reward', type=str, default='reg',
                    help='Type of reward given for the marker ball [reg, dir]')
    parser.add_argument('--create_target_reward', type=float, default=1.0)
    parser.add_argument('--create_sec_goal_reward', type=float, default=2.0)

    parser.add_argument('--run_interval', type=int, default=10)
    parser.add_argument('--render_high_res', type=str2bool, default=False)
    parser.add_argument('--render_ball_traces', type=str2bool, default=False)
    parser.add_argument('--render_text', type=str2bool, default=False)
    parser.add_argument('--render_changed_colors', type=str2bool, default=False)

    # Mega render args
    parser.add_argument('--render_mega_res', type=str2bool, default=False)
    parser.add_argument('--render_mega_static_res', type=str2bool, default=False)
    parser.add_argument('--mega_res_interval', type=int, default=4)
    parser.add_argument('--anti_alias_blur', type=float, default=0.0)

    parser.add_argument('--render_result_figures', type=str2bool, default=False)
    parser.add_argument('--render_borders', type=str2bool, default=False)

    parser.add_argument('--success_failures', type=str2bool, default=False)
    parser.add_argument('--success_only', type=str2bool, default=False)

    parser.add_argument('--exp_type', type=str, default=None,
        help='Type of experiment')
    parser.add_argument('--split_type', type=str, default=None,
        help='Type of Splitting for New tools for create game')
    parser.add_argument('--deterministic_split', type=str2bool, default=False)

    # Create environment specific
    parser.add_argument('--create_max_num_steps', type=int, default=30,
        help='Max number of steps to take in create game (Earlier default 25)')
    parser.add_argument('--create_permanent_goal', type=str2bool, default=True)
    parser.add_argument('--large_steps', type=int, default=40,
        help='Large steps (simulation gap) for create game (Earlier default 40)')
    parser.add_argument('--skip_actions', type=int, default=1,
        help='No. of actions to skip over for create game')
    parser.add_argument('--play_large_steps', type=int, default=30,
        help='Large steps (simulation gap) for create game play env')
    parser.add_argument('--no_overlap_env', type=str2bool, default=False)
    parser.add_argument('--threshold_overlap', type=str2bool, default=True)

    # Varying Action Space Specific
    parser.add_argument('--create_activator_tools', type=str2bool, default=True,
        help='Whether to use activator tools functionality')
    parser.add_argument('--create_num_activators', type=int, default=5)
    parser.add_argument('--create_num_variable_activators', type=int, default=None)
    parser.add_argument('--create_fixed_activators', type=str2bool, default=False,
        help='Whether to use all activator tools as available')
    parser.add_argument('--create_keep_nop', type=str2bool, default=False)
    parser.add_argument('--gnn_ppo', type=str2bool, default=False,
        help='Whether to use GCDQN style architecture in PPO action selection')
    parser.add_argument('--gcdqn_gat_two_hops', type=str2bool, default=False,
                    help="Whether to have 2 steps of message passing in GAT3")
    parser.add_argument('--gat_scale_attention', type=float, default=1.)
    parser.add_argument('--graph_norm_type', type=str, default="None", help="gn/bn/None")
    parser.add_argument('--gnn_node_nonlinearity', type=str2bool, default=True,
        help='Here for backward compatibility. Its function is always On')
    parser.add_argument('--gnn_residual_connection', type=str2bool, default=True,
        help='Residual Connection with GAT')
    parser.add_argument('--ablation_without_gnn', type=str2bool, default=False,
        help='GraphCategorical module, but without any Graph computation')
    parser.add_argument('--gnn_gat_model', type=str, default="gat2", help="gat2/gat2_geo/gat4_geo")
    parser.add_argument('--gat_num_attention_heads', type=int, default=1)
    parser.add_argument('--if_visualise_attention', type=str2bool, default=False)
    parser.add_argument('--use_state_mlp', type=str2bool, default=True)
    parser.add_argument('--gnn_add_state_act', type=str2bool, default=False)
    parser.add_argument('--gnn_num_message_passing', type=int, default=1)
    parser.add_argument('--gnn_layer_norm', type=str2bool, default=False)
    parser.add_argument('--state_act_layer_norm', type=str2bool, default=False,
        help='Layer Norm after concatenation of state and action to a node (does not apply to add)')
    parser.add_argument('--gnn_alpha_teleport', type=float, default=1.0,
            help="GAT2: Scaling of residual connection (default=1.0)")
    parser.add_argument('--gnn_use_main_mlp', type=str2bool, default=True)
    parser.add_argument('--gnn_use_action_summary', type=str2bool, default=False)
    parser.add_argument('--summarizer_use_only_action', type=str2bool, default=False)
    parser.add_argument('--pre_summarize_linear', type=str2bool, default=False)
    parser.add_argument('--mask_categorical', type=str2bool, default=False,
        help='Standard RL 1/2 Baseline')
    parser.add_argument('--input_mask_categorical', type=str2bool, default=False,
        help='Standard RL 2 Baseline')
    parser.add_argument('--input_mask_mlp', type=str2bool, default=False)
    parser.add_argument('--env_total_train_actions', type=int, default=None)
    parser.add_argument('--gnn_use_orig_state_act', type=str2bool, default=False)
    parser.add_argument('--action_set_summary', type=str2bool, default=False)
    parser.add_argument('--action_feature_categorical', type=str2bool, default=False,
        help='Takes action feature directly into the categorical distribution computation')
    parser.add_argument('--action_summarizer', type=str, default=None, help='gnn/lstm/deep_set')
    parser.add_argument('--separate_critic_action_set_summary', type=str2bool, default=False)
    parser.add_argument('--method_name', type=str, default=None, help='relational/summary/baseline/input_mask/mask')
    parser.add_argument('--separate_critic_summary_nodes', type=str2bool, default=False)
    parser.add_argument('--concat_relational_action_features', type=str2bool, default=False)
    parser.add_argument('--separate_critic_state', type=str2bool, default=False)
    parser.add_argument('--lstm_summarizer_num_layers', type=int, default=2)

    ########################################################
    # Embedding specific
    ########################################################

    parser.add_argument('--use_random_embeddings', type=str2bool, default=False)
    parser.add_argument('--verify_embs', type=str2bool, default=False)
    parser.add_argument('--n_distributions', type=int, default=1)


    parser.add_argument('--use_action_trajectory', type=str2bool, default=False)
    parser.add_argument('--emb_batch_size', type=int, default=128)
    parser.add_argument('--trajectory_len', type=int, default=None)
    parser.add_argument('--n_trajectories', type=int, default=1024)
    parser.add_argument('--o_dim', type=int, default=None,
                    help='dimension of action (a or o) embeddings (Earlier default: 3)')
    parser.add_argument('--z_dim', type=int, default=None,
                    help='dimension of (trajectory) z variables (default: 5)')
    parser.add_argument('--print_vars', type=str2bool, default=False,
                    help='whether to print all learnable parameters for sanity check '
                         '(default: False)')
    parser.add_argument('--emb_epochs', type=int, default=None)
    parser.add_argument('--emb_viz_interval', type=int, default=5,
                    help='number of epochs between visualizing action space '
                         '(default: -1 (only visualize last epoch))')
    parser.add_argument('--emb_save_interval', type=int, default=None,
                        help='number of epochs between saving model '
                             '(default: -1 (save on last epoch))')
    parser.add_argument('--n_hidden_traj', type=int, default=3,
                    help='number of hidden layers in modules outside option network '
                         '(default: 3)')
    parser.add_argument('--hidden_dim_traj', type=int, default=64,
                    help='dimension of hidden layers in modules outside option network '
                         '(default: 128)')
    parser.add_argument('--encoder_dim', type=int, default=64,
                    help='size of LSTM encoder output (default: 128)')
    parser.add_argument('--emb_learning_rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
    parser.add_argument('--emb_use_batch_norm', type=str2bool, default=True)
    parser.add_argument('--emb_use_radam', type=str2bool, default=True)
    parser.add_argument('--emb_schedule_lr', type=str2bool, default=False)
    parser.add_argument('--deeper_encoder', type=str2bool, default=False,
                    help='whether or not to use deep convolutional encoder and decoder')
    parser.add_argument('--save_git_diff', type=str2bool, default=False)
    parser.add_argument('--save_dataset', type=str2bool, default=False)
    parser.add_argument('--load_dataset', type=str2bool, default=True)
    parser.add_argument('--load_all_data', type=str2bool, default=None)
    parser.add_argument('--save_emb_model_file', type=str, default=None)
    parser.add_argument('--load_emb_model_file', type=str, default=None)

    parser.add_argument('--train_embeddings', type=str2bool, default=False)
    parser.add_argument('--test_embeddings', type=str2bool, default=False)
    parser.add_argument('--resume_emb_training', type=str2bool, default=False)

    parser.add_argument('--emb_method', type=str, default='htvae')
    parser.add_argument('--shared_var', type=str2bool, default=True)

    parser.add_argument('--load_emb_logvar', type=str2bool, default=True)

    parser.add_argument(
        '--emb_log_interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')

    parser.add_argument('--start_option', type=int, default=None)

    parser.add_argument('--save_embeddings_file', type=str, default=None)
    parser.add_argument('--load_embeddings_file', type=str, default=None)
    parser.add_argument('--load_embeddings_dir', type=str, default='policy_based/method/embedder/saved_embeddings')
    parser.add_argument('--skip_tsne', type=str2bool, default=False)
    parser.add_argument('--num_distributions', type=int, default=400)
    parser.add_argument('--plot_samples', type=int, default=300)

    # htvae specific
    parser.add_argument('--n_hidden_option', type=int, default=3,
                    help='number of hidden layers in option network modules '
                     '(default: 3)')
    parser.add_argument('--hidden_dim_option', type=int, default=128,
                    help='dimension of hidden layers in option network (default: 128)')
    parser.add_argument('--n_stochastic', type=int, default=1,
                        help='number of z variables in hierarchy (default: 1)')
    parser.add_argument('--htvae_clip_gradients', type=str2bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')

    parser.add_argument('--emb_non_linear_lstm', type=str2bool, default=True)
    parser.add_argument('--emb_mlp_decoder', type=str2bool, default=False)
    parser.add_argument('--effect_only_decoder', type=str2bool, default=False)
    parser.add_argument('--concat_oz', type=str2bool, default=False)
    parser.add_argument('--no_initial_state', type=str2bool, default=False)

    #### Action Input to Policy specific
    parser.add_argument('--use_option_embs', type=str2bool, default=True)

    parser.add_argument('--action_base_output_size', type=int, default=64,
                        help='Dimensionality of action base output (Earlier 32 default)')
    parser.add_argument('--action_base_hidden_size', type=int, default=128,
                        help='Dimensionality of action base hidden layers (Earlier 32 default)')
    parser.add_argument('--state_encoder_hidden_size', type=int, default=64,
                        help='Dimensionality of state encoder hidden layers')

    add_args(parser)
    recsim_parser(ps=parser)

    if arg_str is not None:
        args = parser.parse_args(arg_str.split(' '))
    else:
        args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.eval_only and args.num_eval == 10:
        args.num_eval = 100

    if args.save_dataset:
        args.load_dataset = False

    args.train_split = (not args.test_split and not args.eval_split)

    env_specific_args(args)
    if include_method_specific:
        method_specific_args(args)

    general_args(args)

    fixed_action_settings(args)
    args.training_action_set_size = args.action_set_size

    if args.fine_tune:
        args.test_action_set_size = args.action_set_size
        args.nearest_neighbor = True
        args.fixed_action_set = True
        args.action_random_sample = False
        args.test_split = True
        args.train_split = False
        args.both_train_test = False

    if args.eval_interval == -1:
        args.eval_interval = None

    return args
