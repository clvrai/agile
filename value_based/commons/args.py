""" Ideally this file contains and manages all the constants and args used in this project
    I'd like to avoid a messy project...
"""
import argparse

""" Important Constants """
# Constants defining the basic behaviour
HISTORY_SIZE = 3
DIM_ITEM_EMBED = 32
DATA_SPLIT = {"train": 0.6, "val": 0.05, "test": 0.4}
# ENV_RANDOM_SEED = 19920925  # NOTE: it seems this makes the results quite bumpy....
ENV_RANDOM_SEED = 10
SKIP_TOKEN = -1
EMPTY_USER_ID = -1
EMPTY_ITEM_ID = -1
# SLATE_MASK_TOKEN = - np.infty
SLATE_MASK_TOKEN = - 1000000000.0

"""=== ML-100k ==="""
# https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
# ML100K_NUM_ITEMS = 1682
# ML100K_NUM_USERS = 943
# ML100K_NUM_RATINGS = 5
# ML100K_ITEM_FEATURES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
#                         'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
#                         'Western']
# ML100K_USER_FEATURES = ['age', 'F', 'M', 'administrator', 'artist', 'doctor', 'educator', 'engineer', 'entertainment',
#                         'executive', 'healthcare', 'homemaker', 'lawyer', 'librarian', 'marketing', 'none', 'other',
#                         'programmer', 'retired', 'salesman', 'scientist', 'student', 'technician', 'writer']
# ML100K_DIM_ITEM = len(ML100K_ITEM_FEATURES)  # 18
# ML100K_DIM_USER = len(ML100K_USER_FEATURES)  # 24
# USER_HISTORY_COL_NAME = "t-"
# USER_HISTORY_COLS = ["t-{}".format(t + 1) for t in range(HISTORY_SIZE)]

"""=== RecSim ==="""
# To define the variance of GMM in distribution.py
TRAIN_ITEM_COV_COEFF = TEST_ITEM_COV_COEFF = USER_COV_COEFF = 0.1

MIN_QUALITY_SCORE = -100  # The min quality score.
MAX_QUALITY_SCORE = 100  # The max quality score.
SIGMA_QUALITY_SCORE = 0.05

# Define the config of a distribution to sample the length of content
MAX_VIDEO_LENGTH = 10.0  # The maximum length of videos.
MIN_VIDEO_LENGTH = 2.0  # The minimum length of videos.
MEAN_VIDEO_LENGTH = 4.0  # The mean length of videos.
SIGMA_VIDEO_LENGTH = 0.05

# Bonus of items that satisfy the hard constraint of CPR
CPR_BONUS = +3
CPR_BONUS2 = +2

"""=== Sample ==="""
NUM_FAKE_ITEMS = 81
DIM_FAKE_ITEM = 4126
DIM_ITEM_DEEP, DIM_ITEM_WIDE = 4126 - 4096, 4096
NUM_FAKE_USERS = 407614
DIM_FAKE_USER = 137
FAKE_CATEGORY_NAMES = ["genre", "size", "season"]
USER_HISTORY_COL_NAME = "hist_seq"


def str2bool(v):
    """ Used to convert the command line arg of bool into boolean var """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def recsim_parser(ps=None):
    ps.add_argument('--slate_size', type=int, default=6, help='length of a slate')
    ps.add_argument('--batch_step_size', type=int, default=16, help="num of users in a step")
    ps.add_argument('--max_episode_steps', type=int, default=15)
    # num_allItems is the num of all items including new items to inference
    ps.add_argument('--num_allItems', type=int, default=500, help='number of all items')
    ps.add_argument('--num_allUsers', type=int, default=500, help='number of all users')
    # num_candidates is the num of candidate items that will be dealt by an agent
    ps.add_argument('--num_candidates', type=int, default=20, help='number of candidates')
    ps.add_argument('--num_candidates_sp', type=int, default=1, help='number of special candidates')
    ps.add_argument('--num_candidates_correct_normal', type=int, default=2, help='number of special candidates')
    ps.add_argument('--num_intermediate_candidates', type=int, default=15, help='number of intermediate candidates')
    ps.add_argument('--no_click_mass', type=float, default=4.0, help="base score for skip probability")

    # RecSim
    ps.add_argument('--recsim_reward_characteristic', type=str, default="None",
                    help="intrasession_diversity / shannon_entropy / specificity")
    ps.add_argument('--recsim_type_reward', type=str, default="click", help="metrics as a reward")
    ps.add_argument('--recsim_skew_ratio', type=float, default=0.4, help="skewedness used in item resampling")
    ps.add_argument('--recsim_num_categories', type=int, default=10, help="num of main categories")
    ps.add_argument('--recsim_num_subcategories', type=int, default=4, help="num of sub categories")
    ps.add_argument('--recsim_time_budget', type=float, default=200.0, help="MDP or not")
    ps.add_argument('--recsim_skip_penalty', type=float, default=50.0, help="MDP or not")
    ps.add_argument('--recsim_if_mdp', type=str2bool, default=True, help="MDP or not")
    # The following ranges apply only when sampling method is "normal". Default is "random"
    ps.add_argument('--recsim_itemDist_train_low', type=float, default=-0.6, help="min of dist of train item feat")
    ps.add_argument('--recsim_itemDist_train_high', type=float, default=0.6, help="max of dist of train item feat")
    # Uniformly sample from two ranges
    ps.add_argument('--recsim_itemDist_test_low', type=float, default=-1.0, help="min of dist of test item feat")
    ps.add_argument('--recsim_itemDist_test_high', type=float, default=1.0, help="max of dist of test item feat")
    ps.add_argument('--recsim_type_base_reward', type=str, default="click", help="click/watchtime")
    ps.add_argument('--type_UserChoiceModel', type=str, default="SlateDependentChoiceModel",
                    help="MultinomialLogitChoiceModel/MultinomialProportionalChoiceModel/ExponentialCascadeChoiceModel/"
                         "ProportionalCascadeChoiceModel/SlateDependentChoiceModel "
                         "For our case - keep it fixed to SlateDependentChoiceModel as others don't depend on slate")
    ps.add_argument('--recsim_base_SlateChoiceModel', type=str, default="MultinomialProportionalChoiceModel",
                    help="Underlying choice model in SlateDependent one "
                         "MultinomialLogitChoiceModel/MultinomialProportionalChoiceModel/ExponentialCascadeChoiceModel/"
                         "ProportionalCascadeChoiceModel")
    ps.add_argument('--recsim_choice_model_slate_alpha', type=float, default=0.0,
                    help="to weigh scores b/w slate and items in SlateDependentUserChoiceModel")
    ps.add_argument('--recsim_choice_model_user_alpha', type=float, default=0.3,
                    help="weighting of actual user score")
    ps.add_argument('--type_slateAggFn', type=str, default="mean",
                    help="mean; how to get slate embedding in user choice model")
    ps.add_argument('--recsim_type_itemFeatures', type=str, default="continuous",
                    help="discrete/continuous; type of doc_obs")
    ps.add_argument('--recsim_ratio_for_train_items', type=float, default=0.5, help="Ratio of train/test item split")
    ps.add_argument('--recsim_if_simple_obs_format', type=str2bool, default=True, help="to change the format of obs")
    ps.add_argument('--recsim_if_new_action_env', type=str2bool, default=True,
                    help="if we split the candidate items into train/test so that there'd be some unseen items that an"
                         "agent needs to deal with during test")
    ps.add_argument('--recsim_itemFeat_samplingMethod', type=str, default="rejection_sampling",
                    help="rejection_sampling / normal")
    ps.add_argument('--recsim_itemFeat_samplingDist', type=str, default="GMM", help="uniform / GMM")
    ps.add_argument('--recsim_rejectionSampling_distance', type=float, default=0.0,
                    help="distance of test items from the training samples in the item feature space")
    ps.add_argument('--recsim_resampling_method', type=str, default="skewed", help="random / category_based / skewed")
    ps.add_argument('--recsim_choice_model_metric_alpha', type=float, default=0.0, help="")
    ps.add_argument('--recsim_choice_model_how_to_apply_metric', type=str, default="add", help="add / multiply")
    ps.add_argument('--recsim_if_noClickMass_include', type=str2bool, default=True, help="")
    ps.add_argument('--recsim_if_use_subcategory', type=str2bool, default=True, help="Decide if we use the CPR metric")
    ps.add_argument('--recsim_user_model_metric_if_vector', type=str2bool, default=False,
                    help="if we should vectorise the metric used in the user choice model")
    ps.add_argument('--recsim_choice_model_cpr_alpha', type=float, default=5.0, help="weighing the degree of CPR score")
    ps.add_argument('--recsim_no_click_mass_alpha', type=float, default=0.1, help="weighing no-click-mass")
    ps.add_argument('--recsim_no_click_mass_method', type=str, default="None", help="min_max / slate_size / None")
    ps.add_argument('--recsim_cpr_op', type=str, default="multiply", help="add / multiply")
    ps.add_argument('--recsim_user_model_type_constraint', type=str, default="None",
                    help="None/user/candidate: types of the hard constraint")
    ps.add_argument('--recsim_choice_model_type_constraint', type=str, default="None", help="soft / hard")
    ps.add_argument('--recsim_type_specificity', type=str, default="entropy", help="entropy / cosine")
    ps.add_argument('--recsim_type_cpr_diversity_subcategory', type=str, default="entropy", help="entropy / cosine")
    ps.add_argument('--recsim_visualise_console', type=str2bool, default=False, help="Whether to visualise on console")
    ps.add_argument('--recsim_if_add_metric_to_click', type=str2bool, default=False,
                    help="whether to add the metric reward to click")
    ps.add_argument('--recsim_if_special_items', type=str2bool, default=False,
                    help="whether to use the special items for pre-defined pairings")
    ps.add_argument('--recsim_if_special_items_flg', type=str2bool, default=False,
                    help="whether to use the special items for pre-defined pairings")
    ps.add_argument('--recsim_if_special_items_flg_one_hot', type=str2bool, default=False,
                    help="whether to use the special items for pre-defined pairings")
    ps.add_argument('--recsim_special_item_sampling_method', type=str, default="category", help="random / category")
    ps.add_argument('--recsim_special_bonus_coeff', type=float, default=0.2, help="scale the bonus of special items")
    ps.add_argument('--recsim_if_no_fix_for_done', type=str2bool, default=True, help="scale the bonus of special items")
    ps.add_argument('--recsim_resample_num_categories', type=int, default=2, help="scale the bonus of special items")
    ps.add_argument('--recsim_num_noisy_items', type=int, default=2, help="scale the bonus of special items")
    ps.add_argument('--recsim_slate_reward', type=str2bool, default=True, help="scale the bonus of special items")
    ps.add_argument('--recsim_slate_reward_type', type=str, default="all", help="last / position")

    # === Embedding
    ps.add_argument('--item_embedding_type', type=str, default="random", help="random / one_hot / pretrained")
    ps.add_argument('--item_embedding_path', type=str, default="", help="path to a pretrained item-embedding")
    ps.add_argument('--user_embedding_type', type=str, default="random", help="random / one_hot / pretrained")
    ps.add_argument('--user_embedding_path', type=str, default="", help="path to a pretrained user-embedding")

    ps.add_argument('--if_debug', type=str2bool, default=False, help="if print out many things on Console")


def cdqn_get_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()
    ps.add_argument('--logging', type=str2bool, default=False, help="whether to print any logs")
    ps.add_argument('--prefix', type=str, default='', help="Assign a prefix to label experiments on W&B")

    """ === CDQN === """
    # === Training Procedure
    ps.add_argument('--num_epochs', type=int, default=100, help="num of epochs")
    ps.add_argument('--decay_ratio', type=float, default=0.05, help="ratio of total epochs to decay over")
    ps.add_argument('--num_fillIns', type=int, default=10, help="num of fillIns from logged data to replay buffer")
    ps.add_argument('--num_updates', type=int, default=1, help="num of updating Q-networks after one epoch")
    ps.add_argument('--random_seed', type=int, default=2021, help="random seed")
    ps.add_argument('--result_dir', type=str, default='./result/', help='result folder')
    ps.add_argument('--save_dir', type=str, default='./result/', help='weight saving folder')
    ps.add_argument('--buffer_size', type=int, default=1000, help='replay buffer size')
    ps.add_argument('--minimum_fill_replay_buffer', type=int, default=0,
                    help='replay buffer minimum fill size before training starts')
    ps.add_argument('--device', type=str, default='cpu', help="cpu or cuda")
    ps.add_argument('--if_qualitative_result', type=str2bool, default=False, help="cpu or cuda")

    # Scheduler
    ps.add_argument('--epsilon_start', type=float, default=1.0, help="init value of epsilon decay")
    ps.add_argument('--epsilon_end', type=float, default=0.1, help="final value of epsilon decay")

    # Update
    ps.add_argument('--sync_freq', type=int, default=500, help='frequency of syncing the target/main nets in epoch')
    ps.add_argument('--soft_update_tau', type=float, default=0.0, help='tau for soft update')
    ps.add_argument('--batch_size', type=int, default=32, help='mini-batch size')

    # Evaluation
    ps.add_argument('--eval_epsilon', type=float, default=0.0, help="epsilon during evaluation")
    ps.add_argument('--eval_freq', type=int, default=10, help="frequency of evaluation phase")
    ps.add_argument('--num_eval_episodes', type=int, default=1, help="num of episodes in one training")
    ps.add_argument('--if_visualise_debug', type=str2bool, default=False,
                    help="visualise the detailed info, eg, Q-val")
    ps.add_argument('--if_visualise_agent', type=str2bool, default=False, help="visualise the agent's decision making")

    # === Architecture
    ps.add_argument('--agent_type', type=str, default='gcdqn', help='cdqn/dqn/random')
    ps.add_argument('--agent_weight_path', type=str, default='None', help='cdqn/dqn/random')
    ps.add_argument('--if_e_node', type=str2bool, default=False, help='')
    ps.add_argument('--agent_standardRL_type', type=str, default='None', help='1/2/None')
    ps.add_argument('--if_item_retrieval', type=str2bool, default=False, help="If we use Re-Ranking framework")

    # GNN: Product Graph
    ps.add_argument('--action_summarizer', type=str, default="None", help="None/gnn/lstm/deep_set")
    ps.add_argument('--graph_num_hops', type=int, default=1, help="num of hops in GNN message passing")
    ps.add_argument('--graph_type', type=str, default="None",
                    help="gcn/gat/gap/None. Also many gat types in value_based/gnn/GAT/__init__.py")
    ps.add_argument('--graph_dim_hidden', type=int, default=64, help="dim of intermediate node features")
    ps.add_argument('--graph_gat_num_heads', type=int, default=1, help="num of heads for GAT")
    ps.add_argument('--graph_aggregation_type', type=str, default="mean", help="sum/mean/None")
    ps.add_argument('--graph_norm_type', type=str, default="None", help="gn/bn/None")
    ps.add_argument('--graph_gat_arch', type=str, default="",
                    help="Applies architecture specifications from string to GAT_Final. See value_based/gnn/GAT/models.py")

    # GCDQN
    ps.add_argument('--node_type_dim', type=int, default=0,
                    help="dimensions of the current or candidate vector to be appended to each node")
    ps.add_argument('--use_agg_node', type=str2bool, default=False,
                    help="Whether to use an aggregate node in gcdqn and gdqn")
    ps.add_argument('--gcdqn_use_mask', type=str2bool, default=True,
                    help="Whether to use masking operation in DRAG")
    ps.add_argument('--gcdqn_skip_connection', type=str2bool, default=True,
                    help="Whether to use state and item embedding as skip connection in GCDQN q-networks")
    ps.add_argument('--gcdqn_simple_skip_connection', type=str2bool, default=False,
                    help="Whether to use state and item embedding as skip connection in GCDQN q-networks")
    ps.add_argument('--gcdqn_dim_out', type=int, default=64,
                    help="Output dimensions of gcdqn, in the case of skip connections; else defaults to 1")
    ps.add_argument('--boltzmann', type=str2bool, default=False,
                    help="Whether to use Boltzmann or only eps-greedy exploration in gcdqn")
    ps.add_argument('--gcdqn_no_graph', type=str2bool, default=True,
                    help="Only have a shared MLP Q-net for {state, item}")
    ps.add_argument('--gcdqn_use_slate_soFar', type=str2bool, default=True,
                    help="Whether to use slate encoding as input to GCDQN")
    ps.add_argument('--gcdqn_empty_graph', type=str2bool, default=False,
                    help="Whether to use slate encoding as input to GCDQN")
    ps.add_argument('--gcdqn_singleGAT', type=str2bool, default=False,
                    help="Whether to use slate encoding as input to GCDQN")
    ps.add_argument('--gcdqn_no_summary', type=str2bool, default=False,
                    help="Whether to use slate encoding as input to GCDQN")
    ps.add_argument('--gcdqn_no_slate_for_node', type=str2bool, default=False,
                    help="Whether to use the slate-embed for nodes in Action Graph")
    ps.add_argument('--gcdqn_use_pre_summarise_mlp', type=str2bool, default=False,
                    help="Whether to use the pre-summarise-mlp")
    ps.add_argument('--gcdqn_use_post_summarise_mlp', type=str2bool, default=True,
                    help="Whether to use the post-summarise-mlp")
    ps.add_argument('--if_pre_summarize_linear', type=str2bool, default=False,
                    help="Whether to use the linear pre-summarise-mlp")
    ps.add_argument('--if_post_summarize_linear', type=str2bool, default=False,
                    help="Whether to use the linear post-summarise-mlp")
    ps.add_argument('--gcdqn_attention_bonus', type=str2bool, default=False,
                    help="Whether to use the attention-bonus")
    ps.add_argument('--gcdqn_twin_GAT', type=str2bool, default=False, help="Whether to use the attention-bonus")
    ps.add_argument('--gcdqn_random_target', type=str2bool, default=False, help="Whether to use the attention-bonus")
    ps.add_argument('--gcdqn_stack_next_q', type=str2bool, default=True, help="Whether to use the attention-bonus")
    # ps.add_argument('--gnn_ppo', type=str2bool, default=False,
    #                 help='Whether to use GCDQN style architecture in PPO action selection')
    ps.add_argument('--gat_dropout', type=float, default=0.0,
                    help='gat_scale_attention for GAT models')
    ps.add_argument('--gat_scale_attention', type=float, default=1.0,
                    help='gat_scale_attention for GAT models')
    ps.add_argument('--gnn_residual_connection', type=str2bool, default=True,
                    help="GATSimple - Residual Connection")
    ps.add_argument('--gnn_alpha_teleport', type=float, default=1.0,
                    help="Alpha of Teleport Term in Personalised PageRank")
    ps.add_argument('--state_act_layer_norm', type=str2bool, default=False,
                    help="GCDQN - Take Layer Norm after state-act-type concatenation")  # only for GCDQN2
    ps.add_argument('--gcdqn_bug_fixed_target_gat', type=str2bool, default=False,
                    help="GCDQN - Take Layer Norm after state-act-type concatenation")  # only for GCDQN2
    ps.add_argument('--gcdqn_twin_GAT_soft_update', type=float, default=0.0,
                    help="GCDQN - Take Layer Norm after state-act-type concatenation")  # only for GCDQN2
    # NOTE: Don't use this. Use graph_norm_type above! because this is not used in GAT2
    ps.add_argument('--gnn_layer_norm', type=str2bool, default=False, help="Layer Norm inside GAT Simple")
    ps.add_argument('--summarizer_use_only_action', type=str2bool, default=False, help="how to construct node-feat"
                                                                                       "in the action graph")
    ps.add_argument('--lstm_summarizer_num_layers', type=int, default=2)

    # Q-nets
    ## How to instantiate the optimiser
    ps.add_argument('--if_use_main_target_for_others', type=str2bool, default=True, help="")
    ps.add_argument('--if_position_wise_target', type=str2bool, default=False, help="how to set the bellman target")
    ps.add_argument('--if_single_optimiser', type=str2bool, default=False, help="if we need the multiple optimisers")
    ps.add_argument('--if_one_shot_instantiation', type=str2bool, default=False, help="if we instantiate optim at once")
    ## architecture of Q-nets
    ps.add_argument('--if_sequentialQNet', type=str2bool, default=True,
                    help='compute Q-values at once or one by one. Depends on the method being used. Check value_based/policy/agent.py')
    ps.add_argument('--q_net_dim_hidden', type=str, default="256_64", help='hidden dims and join num of neurons by _')
    ps.add_argument('--q_net_if_share_weight', type=str2bool, default=True,
                    help='if we use same Q-net for each pos. CDQN specific')
    ps.add_argument('--q_net_type_act_fn', type=int, default=1,
                    help='defaults to ELU, unused. see value_based/commons/pt_activation_fns.py')
    ps.add_argument('--q_net_lr', type=float, default=0.0001, help="learning rate for Q-nets")
    ps.add_argument('--if_use_lr_scheduler', type=str2bool, default=False, help="learning rate for Q-nets")
    ps.add_argument('--lr_scheduler_alpha', type=float, default=0.95, help="learning rate for Q-nets")
    ps.add_argument('--gamma', type=float, default=0.99, help="gamma for bellman target")
    ps.add_argument('--use_intra_slate_gamma', type=str2bool, default=False, help="Use gamma=1 for intra slate and this gamme for inter-slate transitions")
    ps.add_argument('--grad_clip', type=float, default=1.0, help="clipping the gradient for Q-nets")

    # === Encoders
    # Obs encoder
    ps.add_argument('--obs_encoder_type', type=str, default="None", help="sequential-rnn/lstm/deep_set")
    ps.add_argument('--obs_encoder_dim_hidden', type=int, default=64)

    # Slate encoder
    ps.add_argument('--slate_encoder_type', type=str, default="sequential-rnn", help="basic/sequential(LSTM)")
    ps.add_argument('--slate_encoder_dim_hidden', type=int, default=64)

    # Act encoder
    ps.add_argument('--act_encoder_type', type=str, default="None", help="basic/dense(MLP)")
    ps.add_argument('--act_encoder_dim_out', type=int, default=64)

    # === Environment: Batch-Env so that there are multiple users in a step
    ps.add_argument('--env_name', type=str, default="recsim", help='recsim/movielens/ml-100k')

    recsim_parser(ps)

    # common args for DatasetEnv
    ps.add_argument('--data_if_use_log', type=str2bool, default=False)
    ps.add_argument('--data_category_cols', type=str, default="None/None")
    ps.add_argument('--data_dir', type=str, default="./data/new_movielens/ml-100k/", help="dir for logged data")
    ps.add_argument('--data_click_bonus', type=float, default=1.0, help="dir for logged data")

    # Reward Model
    ps.add_argument('--rm_if_use_Rmodel', type=str2bool, default="False")
    ps.add_argument('--rm_dim_hidden', type=str, default="64_32")
    ps.add_argument('--rm_lr', type=float, default=1e-3)
    ps.add_argument('--rm_norm_type', type=str, default="None")
    ps.add_argument('--rm_dim_out', type=int, default=1)
    ps.add_argument('--rm_weight_path', type=str, default="")
    ps.add_argument('--rm_if_train_simulator', type=str2bool, default=True)
    ps.add_argument('--rm_reward_model_type', type=int, default=1)
    ps.add_argument('--rm_weight_save_dir', type=str, default="")
    ps.add_argument('--rm_deep_embedding_path', type=str, default="")
    ps.add_argument('--rm_wide_embedding_path', type=str, default="")
    ps.add_argument('--rm_offline_or_online', type=str, default="offline")
    return ps


def preprocess_get_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()
    ps.add_argument('--pp_num_epochs', type=int, default=100, help='number of epochs')
    ps.add_argument('--pp_batch_size', type=int, default=64, help='number of epochs')
    ps.add_argument('--pp_num_threads', type=int, default=10, help='num of threads')
    ps.add_argument('--pp_how_to_split', type=str, default="item")
    ps.add_argument('--pp_data_dir', type=str, default="../../../data/new_movielens/ml-100k")
    ps.add_argument('--pp_debug', type=str2bool, default=False)
    return ps


def gnn_get_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()
    ps.add_argument('--gnn_random_seed', type=int, default=2021)
    ps.add_argument('--gnn_data_type', type=str, default="ml-100k")
    ps.add_argument('--gnn_data_dir', type=str, default="./data/new_movielens/ml-100k")
    ps.add_argument('--gnn_model_dir', type=str, default="./trained_weight/gnn")
    ps.add_argument('--gnn_num_epochs', type=int, default=100)
    ps.add_argument('--gnn_val_freq', type=int, default=5)
    ps.add_argument('--gnn_lr', type=float, default=0.001)
    ps.add_argument('--gnn_beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    ps.add_argument('--gnn_beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    ps.add_argument('--gnn_dropout', type=float, default=0.7)
    ps.add_argument('--gnn_grad_clip', type=float, default=1.0)
    ps.add_argument('--gnn_hidden', type=str, default="64_32_16_8")
    ps.add_argument('--gnn_nb', type=int, default=2)
    ps.add_argument('--gnn_device', type=str, default="cuda")

    return ps


def SL_get_args(ps: argparse.ArgumentParser = None):
    if ps is None:
        ps = argparse.ArgumentParser()

    ps.add_argument('--sl_model_type', type=str, default="gru4rec")
    ps.add_argument('--sl_data_dir', type=str, default="./data/new_movielens/ml-100k")
    ps.add_argument('--sl_save_dir', type=str, default="./trained_weight/reward_model")
    ps.add_argument('--sl_num_epochs', type=int, default=100)
    ps.add_argument('--sl_num_trainEpochs', type=int, default=100)
    ps.add_argument('--sl_batch_size', type=int, default=32)
    ps.add_argument('--sl_test_freq', type=int, default=10)
    ps.add_argument('--sl_slate_size', type=int, default=10)
    ps.add_argument('--sl_num_layers_SeqModel', type=int, default=3)
    ps.add_argument('--sl_if_debug', type=str2bool, default=True)
    ps.add_argument('--sl_device', type=str, default="cpu")

    # important params
    ps.add_argument('--sl_if_use_RIE', type=str2bool, default=True)
    ps.add_argument('--sl_random_seed', type=int, default=2021)
    return ps


def add_recsim_args(args: argparse.Namespace):
    args.dim_item = args.recsim_num_categories + int(args.recsim_if_special_items_flg) + \
                    int(args.recsim_if_special_items_flg_one_hot)
    args.dim_user = 0  # In recsim, a user is represented by items so that we don't have specific user-attr
    args.item_feat = None
    args.user_feat = None


def add_args(args: argparse.Namespace):
    """ Set the env specific params """

    args.decay_steps = int(args.num_epochs * args.decay_ratio)  # for epsilon-decay and empirically decided
    if args.recsim_if_new_action_env:
        args.num_trainItems = int(args.num_allItems * args.recsim_ratio_for_train_items)
        args.num_testItems = int(args.num_allItems * (1 - args.recsim_ratio_for_train_items))
        if args.recsim_if_special_items:
            # One special item per category or sub-category
            args.num_allItems_sp = args.num_trainItems_sp = args.num_testItems_sp = args.recsim_num_categories
        else:
            args.num_allItems_sp = args.num_trainItems_sp = args.num_testItems_sp = 0
    else:
        args.num_trainItems = args.num_testItems = args.num_allItems
        if args.recsim_if_special_items:
            # One special item per category or sub-category
            args.num_allItems_sp = args.num_trainItems_sp = args.num_testItems_sp = args.recsim_num_categories
        else:
            args.num_allItems_sp = args.num_trainItems_sp = args.num_testItems_sp = 0

    args.dim_item_hidden = DIM_ITEM_EMBED

    if args.env_name.lower() == "recsim":
        add_recsim_args(args)

        # Obs encoder
        args.obs_encoder_dim_in = args.dim_item - int(args.recsim_if_special_items_flg) - \
                                  int(args.recsim_if_special_items_flg_one_hot)
        args.obs_encoder_mlp_dim_hidden = args.q_net_dim_hidden
        args.obs_encoder_dim_out = args.dim_item - int(args.recsim_if_special_items_flg) - \
                                   int(args.recsim_if_special_items_flg_one_hot)

        # Slate encoder
        args.slate_encoder_mlp_dim_hidden = args.q_net_dim_hidden
        args.slate_encoder_dim_in = args.dim_item_hidden
        args.slate_encoder_dim_out = args.dim_item_hidden

        # Product Graph's input; node_feat = [item_embed, state_embed]
        args.graph_dim_in = args.dim_item + args.obs_encoder_dim_out
        args.graph_dim_out = args.dim_item + args.obs_encoder_dim_out

        # Act encoder
        args.act_encoder_dim_in = args.graph_dim_out if args.graph_type != "gap" else args.graph_dim_in
        args.act_encoder_dim_out = args.dim_item

        # Define the state/obs space
        if args.recsim_if_mdp:
            args.dim_state = args.dim_item
        else:
            args.dim_state = args.obs_encoder_dim_out

    elif args.env_name.lower() == "sample":
        args.dim_deep, args.dim_wide = DIM_ITEM_DEEP, DIM_ITEM_WIDE
        # args.dim_item = DIM_FAKE_ITEM
        args.dim_item = DIM_ITEM_EMBED
        args.dim_user = DIM_FAKE_USER
        args.item_feat = []
        args.user_feat = []

        # Obs encoder
        args.obs_encoder_dim_in = args.dim_user + args.dim_item
        args.obs_encoder_mlp_dim_hidden = args.q_net_dim_hidden
        args.obs_encoder_dim_out = DIM_ITEM_EMBED

        # Slate encoder
        args.slate_encoder_dim_in = args.dim_item
        args.slate_encoder_mlp_dim_hidden = args.q_net_dim_hidden
        args.slate_encoder_dim_out = DIM_ITEM_EMBED

        # Product Graph's input; node_feat = [item_embed, state_embed]
        args.graph_dim_in = args.dim_item + args.obs_encoder_dim_out
        args.graph_dim_out = args.dim_item + args.obs_encoder_dim_out

        # Act encoder
        args.act_encoder_dim_in = args.graph_dim_out if args.graph_type != "gap" else args.graph_dim_in
        args.act_encoder_dim_out = DIM_ITEM_EMBED

        # Define the state/obs space
        args.dim_state = args.obs_encoder_dim_out

    elif args.env_name.lower() == "ml-100k":
        assert False

    # === Specific args for each agent ===
    if args.agent_type == "cdqn":
        # args.q_net_if_share_weight = True  # to compare against AGILE
        # if standardRL CDQN
        if args.agent_standardRL_type == "1":
            args.dim_in = args.obs_encoder_dim_out + args.slate_encoder_dim_out
            args.if_sequentialQNet = False
        elif args.agent_standardRL_type == "2":
            args.dim_in = args.obs_encoder_dim_out + args.slate_encoder_dim_out + args.dim_item_hidden
            args.if_sequentialQNet = False
        else:  # others: vanilla cdqn / rcdqn
            if args.graph_type == "None":
                # === CDQN ===
                # dim_in = dim_state + dim_slate_embed + dim_item(for sequentially computing Q-values)
                args.dim_in = args.obs_encoder_dim_out + args.slate_encoder_dim_out + args.dim_item_hidden
            else:
                # === RCDQN: e-global ===
                # dim_in = state_embed + slate_embed + original_node_feat + summary_vec
                args.dim_in = args.obs_encoder_dim_out + \
                              args.slate_encoder_dim_out + \
                              args.graph_dim_hidden + \
                              args.graph_dim_hidden
        if args.boltzmann:
            args.epsilon_start = 0
            args.epsilon_end = 0
    elif args.agent_type == "dqn":
        # if standardRL DQN
        if args.agent_standardRL_type == "1":
            args.dim_in = args.obs_encoder_dim_out
            args.if_sequentialQNet = False
        elif args.agent_standardRL_type == "2":
            args.dim_in = args.obs_encoder_dim_out + args.num_trainItems
            args.if_sequentialQNet = False
        else:
            # whether vanilla DQN or RDQN
            if args.act_encoder_type != "None":
                args.dim_in = args.obs_encoder_dim_out + args.act_encoder_dim_out + args.dim_item_hidden
            else:
                args.dim_in = args.obs_encoder_dim_out + args.dim_item_hidden
    elif args.agent_type == "lird":
        args.dim_in = args.obs_encoder_dim_out
        args.actor_dim_out = args.dim_item
        args.critic_dim_state = args.obs_encoder_dim_out
        args.critic_dim_item = args.dim_item
    elif args.agent_type == "bfdqn":
        args.dim_in = args.obs_encoder_dim_out
        args.recsim_if_new_action_env = False
        args.num_candidates = args.num_trainItems = args.num_testItems = args.num_allItems
    elif args.agent_type.startswith("gdqn"):
        args.dim_in = args.obs_encoder_dim_out + args.dim_item + 2
        if args.boltzmann:
            args.epsilon_start = 0
            args.epsilon_end = 0
    elif args.agent_type.startswith("gcdqn"):
        if args.action_summarizer in ["lstm", "deep_set"]:
            args.gcdqn_twin_GAT = False
        # if standardRL CDQN
        if args.agent_standardRL_type == "1":
            args.skip_connection_dim_in = args.obs_encoder_dim_out + args.slate_encoder_dim_out
            args.if_sequentialQNet = False
            args.gcdqn_twin_GAT = False
        elif args.agent_standardRL_type == "2":
            args.skip_connection_dim_in = args.obs_encoder_dim_out + args.slate_encoder_dim_out + args.dim_item_hidden
            args.if_sequentialQNet = False
            args.gcdqn_twin_GAT = False
        else:
            if args.gcdqn_no_graph:  # CDQN-SHARE
                args.skip_connection_dim_in = args.obs_encoder_dim_out + args.dim_item_hidden + \
                                              args.slate_encoder_dim_out
                args.gcdqn_use_pre_summarise_mlp = args.gcdqn_use_post_summarise_mlp = False
                args.gcdqn_twin_GAT = False
            else:  # AGILE / Summary-variants
                # dim_item_hidden is from action_linear, which is an action in the architecture
                args.main_Q_net_dim_in = args.obs_encoder_dim_out + args.dim_item_hidden + args.node_type_dim
                if args.gcdqn_skip_connection:
                    # skip_connection_dim_in: MLP after GAT to provide Q-vals
                    args.skip_connection_dim_in = args.obs_encoder_dim_out
                    if args.gcdqn_use_slate_soFar: args.skip_connection_dim_in += args.slate_encoder_dim_out
                    if args.action_summarizer not in ["", "None"]:  # AGILE
                        # dim_in: pre_summarise_mlp, main_Q_net_dim_in: GAT
                        if args.summarizer_use_only_action:
                            args.main_Q_net_dim_in = args.dim_item_hidden  # processed action from action_linear
                            # if args.gnn_residual_connection:
                            #     args.gcdqn_dim_out = args.main_Q_net_dim_in
                            if not args.gcdqn_no_summary:  # it only influences the skip-connection MLP
                                # args.skip_connection_dim_in += args.dim_item_hidden  # summary_vec
                                args.skip_connection_dim_in += args.gcdqn_dim_out  # summary_vec
                            args.skip_connection_dim_in += args.dim_item_hidden  # action_feat
                            # if not args.gcdqn_no_slate_for_node: args.skip_connection_dim_in += args.graph_dim_hidden
                            # summarizer_use_only_action, gcdqn_no_summary
                        else:
                            args.dim_in = args.obs_encoder_dim_out + args.node_type_dim
                            if args.gcdqn_use_slate_soFar: args.dim_in += args.slate_encoder_dim_out
                            if not args.gcdqn_no_slate_for_node: args.dim_in += args.slate_encoder_dim_out
                            args.main_Q_net_dim_in = args.graph_dim_hidden \
                                if args.gcdqn_use_pre_summarise_mlp else args.dim_in
                            if args.if_e_node:
                                args.skip_connection_dim_in += args.graph_dim_hidden \
                                    if args.gcdqn_use_pre_summarise_mlp else args.dim_in  # action_feat
                            else:
                                args.skip_connection_dim_in += args.dim_item_hidden  # action_feat
                            if not args.gcdqn_no_summary:
                                args.skip_connection_dim_in += args.graph_dim_hidden \
                                    if args.gcdqn_use_post_summarise_mlp else args.dim_in  # summary_vec
                        if args.gnn_residual_connection:
                            args.gcdqn_dim_out = args.main_Q_net_dim_in
                    else:
                        if args.gnn_residual_connection:
                            args.gcdqn_dim_out = args.main_Q_net_dim_in + args.slate_encoder_dim_out
                        args.skip_connection_dim_in = (args.main_Q_net_dim_in - args.node_type_dim) + args.gcdqn_dim_out
            if args.gcdqn_simple_skip_connection:
                if args.action_summarizer not in ["", "None"]:  # AGILE
                    args.skip_connection_dim_in = args.obs_encoder_dim_out + (args.graph_dim_hidden * 2)
                else:
                    args.skip_connection_dim_in = args.gcdqn_dim_out

    elif args.agent_type.startswith("agile"):
        # dim_in = state_embed + node_feat + summary_vec
        args.dim_in = args.obs_encoder_dim_out + args.graph_dim_hidden + args.graph_dim_hidden
        if args.boltzmann:
            args.epsilon_start = 0
            args.epsilon_end = 0

        if not args.gcdqn_skip_connection:
            args.gcdqn_dim_out = 1

        if args.gcdqn_no_graph:
            args.gcdqn_dim_out = 0

        if args.gcdqn_use_slate_soFar:
            args.dim_in += args.slate_encoder_dim_out

        if args.gcdqn_empty_graph:
            assert not args.gcdqn_no_graph
    # debug purpose
    args.if_see_action = False
    return args


def check_args(args: argparse.Namespace):
    """Sanity check of args; you can set any rule here to constraint args"""
    if args.recsim_if_new_action_env:
        assert args.num_trainItems > args.num_candidates
        assert args.num_testItems > args.num_candidates


def get_all_args():
    ps = argparse.ArgumentParser()
    ps = cdqn_get_args(ps=ps)
    ps = preprocess_get_args(ps=ps)
    ps = gnn_get_args(ps=ps)
    args = ps.parse_args()
    args = add_args(args=args)
    check_args(args=args)
    return args
