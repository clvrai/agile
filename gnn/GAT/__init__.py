from gnn.GAT.models import GAT2, GAT_Final, GATSimple_RecSim, GATSimple
from gnn.GAT.models_geometric import GAT as GAT_g, GAT2 as GAT2_g, GAT3 as GAT3_g, GAT4 as GAT4_g, GAT5 as GAT5_g, \
    GATSimple_RecSim as GATSimple_g


def launcher(model_name: str = "gat4"):
    if model_name.lower() == "gat2":
        return GAT2
    elif model_name.lower() == "gat_final":
        return GAT_Final
    elif model_name.lower() == "gat_simple":
        return GATSimple_RecSim
    elif model_name.lower() == "gat_geo1":
        return GAT_g
    elif model_name.lower() == "gat_geo2":
        return GAT2_g
    elif model_name.lower() == "gat_geo3":
        return GAT3_g
    elif model_name.lower() == "gat_geo4":
        return GAT4_g
    elif model_name.lower() == "gat_geo5":
        return GAT5_g
    elif model_name.lower() == "gat_geo_simple":
        return GATSimple_g
    else:
        raise ValueError


def test():
    import torch
    print("=== test ===")
    batch_size = 9
    dim_in, dim_hidden = 32, 16
    dim_out, num_heads = 1, 1
    args = {
        "env_name": "recsim",
        "gcdqn_gat_two_hops": False,
        "gnn_ppo": False,
        "graph_norm_type": "bn",
        "num_candidates": 10,
        "device": "cpu",
        "gat_scale_attention": 1.0,
        "gnn_residual_connection": True,
        "gnn_alpha_teleport": 0.9,
        "graph_gat_arch": "mha_gatVIZ_norm_mha_mlp",
    }

    _input = torch.randn(batch_size, args["num_candidates"], dim_in)
    Adj = torch.ones(args["num_candidates"], args["num_candidates"]) - torch.eye(args["num_candidates"])
    Adj = Adj[None, ...].repeat(batch_size, 1, 1)

    for model in [GAT2, GAT_Final, GATSimple, GAT_g, GAT2_g, GAT3_g, GAT4_g, GAT5_g, GATSimple_g]:
        gat = model(dim_in=dim_in, dim_hidden=dim_in, dim_out=dim_out, num_heads=num_heads, args=args)
        for _ in range(5):
            out = gat(_input, Adj)
        a = gat.get_attention(first=True)
        assert out.shape == (batch_size, args["num_candidates"], dim_out)


if __name__ == '__main__':
    test()
