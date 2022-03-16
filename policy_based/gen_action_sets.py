from policy_based.arguments import get_args
import random
import os.path as osp
import os
import numpy as np
from policy_based.envs.create_game.tool_gen import ToolGenerator

# Ex usage python scripts/gen_action_sets.py --action-seg-loc envs/action_segs_new --env-name MiniGrid

args = get_args()

if args.env_name.startswith('Create'):
    tool_gen = ToolGenerator(args.gran_factor)

    train_tools, test_tools = tool_gen.get_train_test_split(args)

    # Randomize here
    np.random.shuffle(train_tools)
    np.random.shuffle(test_tools)

    add_str = ('_' + args.split_type) if (args.split_type is not None and 'New' in args.exp_type) else ''

    new_dir = osp.join(args.action_seg_loc, 'create_' + args.exp_type + add_str)
    if not osp.exists(new_dir):
        os.makedirs(new_dir)
    train_filename = osp.join(new_dir, 'set_train.npy')
    with open(train_filename, 'wb') as f:
        np.save(f, train_tools)

    test_filename = osp.join(new_dir, 'set_test.npy')
    with open(test_filename, 'wb') as f:
        np.save(f, test_tools)

else:
    print('Unspecified Environment!')
