#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch


gym_name = 'CartPole-v1'
model_filename = "cartpole_v1.pt"
gym_max_steps = 200 # max steps for cartpole is 200


def create_env():
    # print(gym.envs.registry.all())
    env = gym.make(gym_name)
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
        torch.nn.Linear(input_size, input_size*output_size),
        torch.nn.Linear(input_size*output_size, input_size*output_size),
        torch.nn.Linear(input_size*output_size, input_size*output_size),
        torch.nn.Linear(input_size*output_size, output_size),
        torch.nn.Softmax(dim=0),
    )


def save_model(model, filename=model_filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename=model_filename):
    model.load_state_dict(torch.load(filename))
    model.eval()


def observation_to_model_action_and_gradient(env, model, observation):
    model_output = model(observation)
    action = None
    gradient = None

    if isinstance(env.action_space, gym.spaces.Discrete):
        action = torch.multinomial(model_output, 1, replacement=True)
        num_model_outputs = model_output.numel()
        one_hot = torch.zeros(num_model_outputs)
        one_hot[action] = 1
        # print("action", action,
        #       "model output", model_output,
        #       "weights", weights)
        gradient = torch.dot(model_output, one_hot.float())

    assert action is not None, "implement something to calculate model action"

    return action, gradient


def plot_reward(reward_per_episode, title="", show=True):
    plt.plot(range(len(reward_per_episode)), reward_per_episode)
    plt.xlabel('Episode #')
    plt.ylabel('Reward')
    plt.title(title)
    if show:
        plt.show()


def constrain(x, minimum, maximum):
    if x > maximum:
        return maximum
    elif x < minimum:
        return minimum
    else:
        return x


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def train():
    print("Training")

    episode_rewards = []

    with create_env() as env:
        model = create_model(env)

        for episode_num in range(1500):
            observation_gradient_pairs = []
            observation = env.reset()

            total_reward = 0

            num_steps_taken = gym_max_steps
            for t in range(gym_max_steps):
                env.render()

                observation = torch.Tensor(observation)
                action, gradient = observation_to_model_action_and_gradient(env, model, observation)
                observation_gradient_pairs.append([observation, gradient])
                observation, reward, done, info = env.step(action.item())

                total_reward += reward

                if done:
                    num_steps_taken = t+1
                    break

            print("episode {}, {} timesteps, {} reward".format(episode_num, num_steps_taken, total_reward))
            if num_steps_taken == gym_max_steps:
                save_model(model, model_filename)

            # update the model
            if len(episode_rewards) > 0:
                last_few_episodes = episode_rewards[-5:]
                previous_average_reward = sum(last_few_episodes)/len(last_few_episodes)
                difference = total_reward - previous_average_reward

                model.zero_grad()

                for observation, gradient in observation_gradient_pairs:
                    # compute gradients; stacked calls to .backward() without a zero_grad will sum the gradients
                    gradient.backward()
                with torch.no_grad():
                    learning_rate = 0.001
                    max_learning_rate = 0.01
                    reward_multiplier = learning_rate * difference

                    effective_learning_rate = constrain(reward_multiplier, -1*max_learning_rate, max_learning_rate)
                    for param in model.parameters():
                        param += effective_learning_rate * param.grad
            episode_rewards.append(total_reward)

    plt.subplot(2, 1, 1)
    plot_reward(episode_rewards, "No averaging", show=False)
    plt.subplot(2, 1, 2)
    plot_reward(moving_average(episode_rewards, 10), "10 episode moving average", show=True)


def replay():
    print("Replaying")
    with create_env() as env:
        model = create_model(env)
        load_model(model, model_filename)

        episode_rewards = []

        for episode_num in range(100):
            num_steps_taken = 0
            observation = env.reset()
            total_reward = 0
            for t in range(gym_max_steps):
                env.render()

                num_steps_taken += 1
                observation = torch.Tensor(observation)
                action, gradient = observation_to_model_action_and_gradient(env, model, observation)
                observation, reward, done, info = env.step(action.item())

                total_reward += reward

                if done:
                    break
            episode_rewards.append(total_reward)
            print("episode {}, {} timesteps, {} reward".format(episode_num, num_steps_taken, total_reward))
    plot_reward(episode_rewards)


if __name__ == "__main__":
    train()
