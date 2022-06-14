import random

from tqdm import trange
import torch

import Models.RL.Envs.septica as sep
import Models.RL.septica_agent as sa
from Models.RL.rl_agent_utils import LinearScheduleEpsilon, ReplayBuffer

if __name__ == "__main__":
    env = sep.SepticaEnv()
    agent1 = sa.SepticaAgent(env=env, gamma=1, batch_size=64, replay_buff_size=3000, lr=1e-3, pre_train_steps=300,
                                 eps_scheduler=LinearScheduleEpsilon(start_eps=1, final_eps=0.05, pre_train_steps=300,
                                                                 final_eps_step=5*10 ** 4))
    agent2 = sa.SepticaAgent(env=env, gamma=1, batch_size=64, replay_buff_size=3000, lr=1e-3, pre_train_steps=300,
                                 eps_scheduler=LinearScheduleEpsilon(start_eps=1, final_eps=0.05, pre_train_steps=300,
                                                                 final_eps_step=5*10 ** 4))
    agents = [agent1, agent2]
    losses = [[], []]
    ep_rewards = [[], []]
    train_matches = 2 * 10 ** 4
    # train_matches = 1
    for ep in trange(train_matches):
        init_state = env.reset()

        first_agent = random.randint(0, 1)
        hand_index = 1
        current_agent_idx = first_agent
        current_state = init_state
        done = False
        reward = None
        rewards = [0, 0]
        add_to_buff = True
        other_agent = agents[(current_agent_idx + 1) % 2]
        still_is_first = True
        while not done:
            if reward is not None:
                if len(env.played_cards) > 0 or (len(env.played_cards) == 0 and not still_is_first):
                    older_state = None
                    old_state = current_state
                    current_state = next_state_enemy
                else:
                    older_state = old_state
                    older_action = old_action
                    older_extra_info = old_extra_info
                    older_reward = old_reward
                    old_state = current_state
                    current_state = next_state_me
                # for every step but the first one, use the state returned from the previous agent's step as the current state
                old_action = action
                old_extra_info = extra_info
                old_reward = reward
            current_agent = agents[current_agent_idx]  # type: sa.SepticaAgent
            # print(f"current state is {current_state}")
            is_first = current_state[-2]
            current_state = current_agent.process_state(current_state)
            hand_index = int(first_agent == current_agent_idx)  # first player is always player hand in env
            action = current_agent.get_action([current_state], eps=current_agent.eps_scheduler.get_value(current_agent.total_steps))[0]
            action, extra_info = action
            # print(f"action taken is {action, extra_info}")
            next_state_me, next_state_enemy, reward, done = env.step_individual(hand_index, is_first, action, extra_info)
            still_is_first = next_state_me[-2]
            # print(still_is_first)
            # print(len(env.played_cards))
            rewards[current_agent_idx] += reward
            rewards[(current_agent_idx + 1) % 2] -= reward
            if other_agent.total_steps > 0 and older_state is None:
                # add new env to replay buffer of the other dude
                # handle (s_a1, 1) -> (s'_a1, 0), where it's the same agent in the next step, too
                if other_agent != current_agent:
                    next_state = current_agent.process_state(next_state_enemy)
                else:
                    next_state = current_state
                other_agent.replay_buffer.add(state=old_state, action=other_agent.action_to_idx(old_action, old_extra_info), reward=old_reward-reward, next_state=next_state, done=done)
            elif other_agent.total_steps > 0 and older_state is not None:
                next_state = current_agent.process_state(next_state_enemy)
                other_agent.replay_buffer.add(state=older_state,
                                              action=other_agent.action_to_idx(older_action, older_extra_info),
                                              reward=older_reward - old_reward, next_state=next_state, done=done)
            if done:
                # terminal states get added to the replay buffer regardless
                next_state_me = current_agent.process_state(next_state_me)
                current_agent.replay_buffer.add(state=current_state, action=current_agent.action_to_idx(action, extra_info), reward=reward, next_state=next_state_me, done=done)
            current_agent.total_steps += 1
            if current_agent.total_steps > current_agent.pre_train_steps:
                loss = current_agent.train_step()
                losses[current_agent_idx].append(loss)
            other_agent = agents[current_agent_idx]
            if len(env.played_cards) > 0 or (len(env.played_cards) == 0 and not still_is_first):
                # print("when the magic in the summer")
                current_agent_idx = (current_agent_idx + 1) % 2
        ep_rewards[0].append(rewards[0])
        ep_rewards[1].append(rewards[1])

        if ep % 20 == 0 and ep > 0:
            print(f"Avg loss for player 1 is {sum(losses[0])/len(losses[0])}.")
            print(f"Avg loss for player 2 is {sum(losses[1])/len(losses[1])}.")
            print(f"Loss of last 20 eps for player 1 is {sum(losses[0][-20:])/20}.")
            print(f"Loss of last 20 eps for player 2 is {sum(losses[1][-20:])/20}.")
            print(f"Avg reward for player 1 is {sum(ep_rewards[0])/len(ep_rewards[0])}.")
            print(f"Avg reward for player 2 is {sum(ep_rewards[1])/len(ep_rewards[1])}.")
            print(f"Reward of last 20 eps for player 1 is {sum(ep_rewards[0][-20:])/20}.")
            print(f"Reward of last 20 eps for player 2 is {sum(ep_rewards[1][-20:])/20}.")
            print(f"Eps is {(agent1.eps_scheduler.get_value(agent1.total_steps)+agent2.eps_scheduler.get_value(agent2.total_steps))/2}.")
        if ep % 200 == 0 and ep > 0:
            torch.save(agent1.q.state_dict(),
                       f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_q_{ep}_doubleq_agent1.model")
            torch.save(agent1.q_target.state_dict(),
                       f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_qtarget_{ep}_doubleq_agent1.model")
            torch.save(agent2.q.state_dict(),
                       f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_q_{ep}_doubleq_agent2.model")
            torch.save(agent2.q_target.state_dict(),
                       f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_qtarget_{ep}_doubleq_agent2.model")

            agent1.get_statistics(10 ** 2)
