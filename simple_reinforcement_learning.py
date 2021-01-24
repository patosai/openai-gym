#!/usr/bin/env python3

import gym
import torch


def create_env():
    # print(gym.envs.registry.all())
    env = gym.make('CartPole-v0')
    print("observation space: ", env.observation_space)
    print("observation space shape: ", env.observation_space.shape)
    print("observation space high: ", env.observation_space.high)
    print("observation space low: ", env.observation_space.low)
    print("action space: ", env.action_space)
    return env


def create_model(env):
    input_size = env.observation_space.shape[0]
    output_size = None

    if isinstance(env.action_space, gym.spaces.Discrete):
        output_size = env.action_space.n

    assert output_size is not None, "implement something to calculate the output_size"

    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 4),
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, output_size),
        torch.nn.Softmax(),
    )


def observation_to_model_action_and_gradient(env, model, observation):
    model_output = model(observation)
    action = None
    gradient = None

    if isinstance(env.action_space, gym.spaces.Discrete):
        action = torch.multinomial(model_output, 1, replacement=True)
        action_one_hot = torch.nn.functional.one_hot(action, model_output.numel())[0]
        gradient = torch.dot(model_output, action_one_hot.float())

    assert action is not None, "implement something to calculate model action"

    return action, gradient


def main():
    with create_env() as env:
        model = create_model(env)

        episode_rewards = []

        for episode_num in range(20):
            print("episode {}".format(episode_num))

            observation_gradient_pairs = []
            observation = env.reset()

            total_reward = 0
            for t in range(100):
                env.render()

                observation = torch.Tensor(observation)
                action, gradient = observation_to_model_action_and_gradient(env, model, observation)
                observation_gradient_pairs.append([observation, gradient])
                observation, reward, done, info = env.step(action.item())

                total_reward += reward

                if done:
                    print("episode finished after {} timesteps, {} reward".format(t+1, total_reward))
                    break

            # update the model
            episode_rewards.append(total_reward)
            if len(episode_rewards) > 0:
                previous_average_reward = sum(episode_rewards)/len(episode_rewards)
                difference = previous_average_reward - total_reward
                model.zero_grad()
                for observation, gradient in observation_gradient_pairs:
                    # compute gradients; stacked calls to .backward() without a zero_grad will sum the gradients
                    gradient.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        learning_rate = 0.005
                        param -= difference * learning_rate * param.grad


if __name__ == "__main__":
    main()
