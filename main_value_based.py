import os
import time
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from value_based.commons.plot_agent import visualise_agent
from value_based.commons.seeds import set_randomSeed
from value_based.commons.launcher import launch_env, launch_agent, launch_encoders, launch_embedding
from value_based.commons.replay_buffer import ReplayBuffer
from value_based.commons.scheduler import AnnealingSchedule
from value_based.commons.utils import logging, create_directory_name, create_log_file, mean_dict


def run(args, log_buffer=None):
    """ Based on Algo.2 in CascadingDQN, X.Chen et al., 2020 """
    print(vars(args))

    # from pudb import set_trace; set_trace()
    dict_embedding = launch_embedding(args=args)
    env = launch_env(dict_embedding=dict_embedding, args=args)
    dict_embedding["item"] = env.append_special_normal_flag_to_embedding(item_embedding=dict_embedding["item"])
    encoders_dict = launch_encoders(item_embedding=dict_embedding["item"], args=args)
    if args.if_use_main_target_for_others:
        _encoders_dict = launch_encoders(item_embedding=dict_embedding["item"], args=args)
        encoders_dict = {
            "obs_encoder": [encoders_dict["obs_encoder"], _encoders_dict["obs_encoder"]],
            "slate_encoder": [encoders_dict["slate_encoder"], _encoders_dict["slate_encoder"]],
            "act_encoder": [encoders_dict["act_encoder"], _encoders_dict["act_encoder"]],
        }
    else:
        encoders_dict = {
            "obs_encoder": [encoders_dict["obs_encoder"]],
            "slate_encoder": [encoders_dict["slate_encoder"]],
            "act_encoder": [encoders_dict["act_encoder"]],
        }
    agent = launch_agent(dict_embedding=dict_embedding, encoders_dict=encoders_dict, args=args)
    agent.set_baseItems_dict(baseItems_dict=env.baseItems_dict)
    if args.if_qualitative_result:
        agent.load(save_dir=os.path.join(args.result_dir, "weights"))
    scheduler = AnnealingSchedule(start=args.epsilon_start, end=args.epsilon_end, decay_steps=args.decay_steps)
    _if_mdp = True if args.recsim_if_mdp and args.env_name == "recsim" else False
    replay_buffer = ReplayBuffer(size=args.buffer_size, if_mdp=_if_mdp)

    """ ==== Fill replay buffer with random interactions ==== """
    if args.agent_type != "random":
        t1 = time.time()
        min_replay_buffer_size = min(args.buffer_size // 10, args.minimum_fill_replay_buffer)
        while len(replay_buffer) < min_replay_buffer_size:
            _, _, env, agent = fill_replay_buffer(agent=agent,
                                                  env=env,
                                                  epsilon=1.0,
                                                  global_ts=None,
                                                  replay_buffer=replay_buffer,
                                                  args=args)
        logging('Filled the replay buffer with {} steps in {} seconds'.format(min_replay_buffer_size, time.time() - t1))

    # === for each episode
    global_ts = 0  # time-step
    for epoch in range(args.num_epochs):
        # Evaluation phase; Run Evaluation once at the beginning of the training!
        if (epoch + 1) % args.eval_freq == 0:
            args.ts = global_ts  # for visualisation of attention map!
            for train_test_flg in ["train", "test"]:
                if (train_test_flg == "test") and (args.agent_standardRL_type != "None"):
                    continue
                args.train_test_flg = train_test_flg
                eval_metrics, env, agent, attention_map_image_list = _evaluate(agent=agent, env=env, args=args)

                if log_buffer is not None:  # Test purpose
                    log_buffer.store_metrics(ep_metrics=eval_metrics, if_eval=True)
                else:
                    print('Prefix: ', args.prefix)

        if args.agent_type != "random":
            """ === Train the policy === """
            # debug purpose
            if args.if_debug:
                if (epoch + 1) % args.eval_freq == 0:
                    args.if_see_action = True
                else:
                    args.if_see_action = False

            global_ts, train_metrics, env, agent = _train(agent=agent,
                                                          env=env,
                                                          epsilon=scheduler.get_value(ts=epoch),
                                                          global_ts=global_ts,
                                                          replay_buffer=replay_buffer,
                                                          args=args)

            """ === Visualise the results of one training === """
            logging(f"[Train] epoch: {epoch} global_ts: {global_ts} epsilon: {scheduler.get_value(ts=epoch):.3f} "
                    f"reward: {train_metrics['ep_reward']} hit rate: {train_metrics['hit_rate']} "
                    f"specificity: {train_metrics['specificity']:.3f} entropy: {train_metrics['shannon_entropy']:.3f}")
            if log_buffer is not None:  # Test purpose
                log_buffer.store_metrics(ep_metrics=train_metrics, if_eval=False)

            """ === Update the policy with mini-batch === """
            results = list()
            for _i in range(args.num_updates):
                # Control statement of gradient checker
                if _i == 0:
                    agent.set_if_check_grad(flg=False)
                elif (_i + 1) == args.num_updates:
                    if args.logging:
                        agent.set_if_check_grad(flg=True)
                    else:
                        agent.set_if_check_grad(flg=False)

                if args.agent_type == "lird":
                    obses, actions, rewards, next_obses, dones, action_embeds = \
                        replay_buffer.sample(batch_size=args.batch_size)
                    candidate_lists = None
                else:
                    obses, actions, rewards, next_obses, dones, candidate_lists = \
                        replay_buffer.sample(batch_size=args.batch_size)

                result = agent.update(obses,
                                      actions if args.agent_type != "lird" else action_embeds,
                                      rewards,
                                      next_obses,
                                      dones,
                                      candidate_lists)
                results.append(result)

            # Save the agent
            if ((epoch + 1) % (args.eval_freq * 5) == 0) or (epoch == 0):
                agent.save(save_dir=os.path.join(args.result_dir, "weights"), epoch=epoch)

            """ === Visualise the results of updating === """
            results = mean_dict(_list_dict=results)
            logging(f"[Update] epoch: {epoch} global_ts: {global_ts} result: {results['loss']} ")

            # Sync the main and target networks
            if (epoch + 1) % args.sync_freq == 0:
                agent.sync(tau=float(args.soft_update_tau))

        """ === Before the next epoch ==== """
        agent.increment_epoch(_v=1)  # need to keep track of epoch to visualise the weights of Q-nets internally
    return log_buffer


def fill_replay_buffer(agent, env, global_ts, epsilon, replay_buffer, args):
    env.set_if_eval(flg=False)
    obs = env.reset()
    done, ts = False, 0
    ##### agent = None -> set
    while not np.all(done):
        # action = agent.select_action(obs=obs, candidate_list=env.items_dict["train"], epsilon=epsilon)
        action = agent.random_policy.select_action(batch_size=obs.shape[0], candidate_list=env.items_dict["train"])
        if args.agent_type == "lird": action, action_embed = action  # need to unpack the action when using LIRD
        next_obs, reward, done, info = env.step(action=action)

        """ === After One batch time-step === """
        # In the case of RecSim batch_step_size == num_users so that we need to be careful about that!
        env.update_metrics(reward=reward.ravel(), gt_items=info["gt_items"], pred_slates=action)

        # Add experiences to replay memory
        next_obs_index = 0  # only next_obs is of the shape of the num of current active users so we need another index!
        for i in range(obs.shape[0]):
            # done indicates if a user who was active before action turned inactive by executing the action
            if done[i]:
                if args.recsim_if_mdp:
                    _next_obs = np.zeros_like(obs[0])  # Store the dummy obs: when compute next Q-vals it doesn't matter
                else:
                    _next_obs = obs.create_empty_obs()
            else:
                _next_obs = next_obs[next_obs_index]
                next_obs_index += 1
            replay_buffer.add(obs_t=obs[i],
                              action=action[i, :],
                              reward=info["slate_reward"][i] if args.recsim_slate_reward else reward[i],
                              obs_tp1=_next_obs,
                              done=info["slate_done"][i] if args.recsim_slate_reward else done[i],
                              candidate_list=env.items_dict["train"] if args.agent_type != "lird"
                              else action_embed[i, :])

        """ === Before the next time-step === """
        obs = next_obs

    return None, None, env, agent


def _train(agent, env, global_ts, epsilon, replay_buffer, args):
    # from pudb import set_trace; set_trace()
    env.set_if_eval(flg=False)
    obs = env.reset()
    done, ts = False, 0
    agent.train()
    while not np.all(done):
        action = agent.select_action(obs=obs, candidate_list=env.items_dict["train"], epsilon=epsilon)
        if args.agent_type == "lird": action, action_embed = action  # need to unpack the action when using LIRD
        next_obs, reward, done, info = env.step(action=action)

        """ === After One batch time-step === """
        # In the case of RecSim batch_step_size == num_users so that we need to be careful about that!
        ts += obs.shape[0]
        global_ts += obs.shape[0]
        env.update_metrics(reward=reward.ravel(), gt_items=info["gt_items"], pred_slates=action)

        # Add experiences to replay memory
        next_obs_index = 0  # only next_obs is of the shape of the num of current active users so we need another index!
        for i in range(obs.shape[0]):
            # done indicates if a user who was active before action turned inactive by executing the action
            if done[i]:
                if args.recsim_if_mdp:
                    _next_obs = np.zeros_like(obs[0])  # Store the dummy obs: when compute next Q-vals it doesn't matter
                else:
                    _next_obs = obs.create_empty_obs()
            else:
                _next_obs = next_obs[next_obs_index]
                next_obs_index += 1
            replay_buffer.add(obs_t=obs[i],
                              action=action[i, :],
                              reward=info["slate_reward"][i] if args.recsim_slate_reward else reward[i],
                              obs_tp1=_next_obs,
                              done=info["slate_done"][i] if args.recsim_slate_reward else done[i],
                              candidate_list=env.items_dict["train"] if args.agent_type != "lird"
                              else action_embed[i, :])

        if args.if_see_action:  # debug purpose
            print(action)

        """ === Before the next time-step === """
        obs = next_obs

    """ === After all the steps === """
    _metrics = env.get_metrics()
    return global_ts, _metrics, env, agent


def _evaluate(agent, env, args):
    # from pudb import set_trace; set_trace()
    env.set_if_eval(flg=True)
    env.set_if_eval_train_or_test(train_or_test=args.train_test_flg)
    total_metrics = list()
    agent.eval()
    attention_map_image_list = list()
    with torch.no_grad():
        for ep in range(args.num_eval_episodes):
            if ep == 0:
                env.set_if_visualise_console(flg=True)  # print on the console only at the first eval episode
                agent.set_if_visualise(flg=True)
                if args.if_visualise_agent:
                    already_visualised = False
                else:
                    already_visualised = True
            obs = env.reset()
            done, ts = False, 0
            while not np.all(done):
                action = agent.select_action(obs=obs,
                                             candidate_list=env.items_dict[args.train_test_flg],
                                             epsilon=args.eval_epsilon)
                if args.agent_type == "lird": action, action_embed = action  # need to unpack the action when using LIRD
                next_obs, reward, done, info = env.step(action=action)

                if not already_visualised and args.agent_type != "random":
                    if any(reward >= 1.0) or np.random.rand() < 0.1:
                        # Select the time-steps to visualise the attention map
                        data = agent.get_out_info()
                        data["category"] = env.get_mainCategory_of_items(arr_items=env.items_dict[args.train_test_flg])
                        data["special_item_flg_vec"] = env.get_special_item_flg_vec(
                            arr_items=env.items_dict[args.train_test_flg])
                        _attention_map_image_list = visualise_agent(env=env, inputs=data, args=vars(args))
                        if _attention_map_image_list is not None:
                            attention_map_image_list += _attention_map_image_list
                        already_visualised = True

                """ === After One batch time-step === """
                ts += obs.shape[0]
                env.update_metrics(reward=reward.ravel(), gt_items=info["gt_items"], pred_slates=action)

                if args.if_see_action:  # debug purpose
                    print(action)

                """ === Before the next time-step === """
                obs = next_obs
                agent.set_if_visualise(flg=False)

            """ === After One epoch === """
            if ep == 0: env.set_if_visualise_console(flg=False)
            _metrics = env.get_metrics()
            if args.logging:
                logging(f"[Evaluate: {args.train_test_flg}] epoch: {ep} ts: {ts} reward: {_metrics['ep_reward']} "
                        f"hit rate: {_metrics['hit_rate']} specificity: {_metrics['specificity']:.3f} "
                        f"entropy: {_metrics['shannon_entropy']:.3f}")
            total_metrics.append(_metrics)

    total_metrics = mean_dict(_list_dict=total_metrics)
    logging(f"[Evaluate: {args.train_test_flg}] reward: {total_metrics['ep_reward']} "
            f"hit rate: {total_metrics['hit_rate']} specificity: {total_metrics['specificity']:.3f} "
            f"entropy: {total_metrics['shannon_entropy']:.3f}")
    env.set_if_eval(flg=False)
    agent.train()
    return total_metrics, env, agent, attention_map_image_list


def main(args):
    # Prep for an experiment
    args.result_dir = create_directory_name(dir_name=args.result_dir)
    create_log_file(dir_name=args.result_dir, args=args)

    # Set the random seed
    set_randomSeed(seed=args.random_seed)

    # run an experiment
    run(args=args)


if __name__ == '__main__':
    from value_based.commons.args import get_all_args

    main(args=get_all_args())
