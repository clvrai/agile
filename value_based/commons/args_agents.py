import itertools

_dict = {
    "Random": {
        "agent_type": ["random"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["None"],
        "q_net_if_share_weight": [False],
        "if_item_retrieval": [False]
    },
    "DQN": {
        "agent_type": ["dqn"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["None"],
        "q_net_if_share_weight": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["None"],
        "act_encoder_type": ["None"],
        "if_item_retrieval": [False],
        "agile_skip_connection": [False],
        "agile_simple_skip_connection": [False],
        "agile_no_graph": [True],
    },
    "STANDARD-CDQN": {
        "agent_type": ["cdqn"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["None"],
        "q_net_if_share_weight": [False, True],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "act_encoder_type": ["None"],
        "if_item_retrieval": [False],
        "if_sequentialQNet": [False],
        "agent_standardRL_type": ["1", "2"],
        "agile_no_graph": [False],
    },
    "CDQN": {
        "agent_type": ["cdqn"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["None"],
        "q_net_if_share_weight": [False, True],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "act_encoder_type": ["None"],
        "if_item_retrieval": [False],
        "agile_no_graph": [False],
    },
    "CDQN-GCDQN": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["None"],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "act_encoder_type": ["None"],
        "if_item_retrieval": [False],
        "gcdqn_no_graph": [True],
        "gcdqn_use_slate_soFar": [True],
        "agent_standardRL_type": ["None"],
    },
    "SRL-GCDQN": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["None"],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "act_encoder_type": ["None"],
        "if_item_retrieval": [False],
        "gcdqn_no_graph": [True],
        "gcdqn_use_slate_soFar": [True],
        # "agent_standardRL_type": ["1", "2"],
        "agent_standardRL_type": ["2"],
        "if_sequentialQNet": [False],
    },
    "RCDQN": {
        "agent_type": ["gcdqn"],
        # "graph_aggregation_type": ["mean", "sum"],
        "graph_aggregation_type": ["mean"],
        # "graph_type": ["gcn", "gat_geo4", "gap"],
        "graph_type": ["gat2"],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "if_item_retrieval": [False],
        "node_type_dim": [0],
        "action_summarizer": ["gnn"],
        "graph_norm_type": ["ln"],
        "if_e_node": [False],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
    },
    "DRAG": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["None"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True, False],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "gnn_residual_connection": [True, False],
    },
    "AGILE": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn", "sequential-lstm", "sequential-deep_set"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [True],
        "if_pre_summarize_linear": [False],
        "if_position_wise_target": [True],
        "gcdqn_stack_next_q": [True],
    },
    "AGILE-TWIN-GAT": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn", "sequential-lstm", "sequential-deep_set"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        # "gcdqn_use_pre_summarise_mlp": [True],
        "gcdqn_use_pre_summarise_mlp": [False],
        "gcdqn_twin_GAT": [True],
    },
    "AGILE-no-pre-summariser": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [False],
    },
    "AGILE-no-post-summariser": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [True],
        "gcdqn_use_post_summarise_mlp": [False],
    },
    "AGILE-no-slate-for-node": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [True],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [True],
    },
    "AGILE-no-summary": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [True],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [True],
    },
    "AGILE-state-action": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [True],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [False],
    },
    "AGILE-state-action-no-summary": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [True],
        "gcdqn_no_slate_for_node": [True],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [False],
    },
    "AGILE-action": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [True],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [False],
    },
    "AGILE-action-no-summary": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["gat2", "gcn"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [True],
        "action_summarizer": ["gnn"],
        "gcdqn_no_summary": [True],
        "gcdqn_no_slate_for_node": [False],
        "if_e_node": [True],
        "gcdqn_use_pre_summarise_mlp": [False],
    },
    "Summary-state-action": {
        "agent_type": ["gcdqn"],
        "graph_aggregation_type": ["mean"],
        "graph_type": ["None"],
        "gcdqn_use_mask": [True],
        "gcdqn_skip_connection": [True],
        "gcdqn_simple_skip_connection": [False],
        "obs_encoder_type": ["sequential-rnn"],
        "slate_encoder_type": ["sequential-rnn"],
        "graph_norm_type": ["ln"],
        "graph_gat_num_heads": [1],
        "node_type_dim": [0],
        "gcdqn_use_slate_soFar": [True],
        "summarizer_use_only_action": [False],
        "action_summarizer": ["lstm", "deep_set"],
        "gcdqn_no_summary": [False],
        "gcdqn_no_slate_for_node": [True],
        "if_e_node": [False],
        "gcdqn_use_pre_summarise_mlp": [False],
        "gcdqn_use_post_summarise_mlp": [False],
    },
}

params_dict = {}
for k, v in _dict.items():
    keys, values = zip(*v.items())
    _list_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
    params_dict[k] = _list_dict


# print(params_dict)
# asdf


def check_param(_params_dict):
    """ Check if the params for an agent is right """
    return True