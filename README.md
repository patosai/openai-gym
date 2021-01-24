OpenAI Gym
=========

Here I attempt to tackle the puzzles in OpenAI gym using a variety of machine learning techniques.

### Simple Reinforcement Learning

The basic concept is, for each episode, I keep track of the observations and resulting sampled actions. Depending if the reward is greater than the average of previous episodes, the actions are encouraged or discouraged by updating gradients on a fully connected neural network. This update is done after every episode.

The key idea is that in the short term, while some actions may not be optimal but they're buoyed by better actions at other times, in the long run these non-optimal actions will be averaged out and corrected by increasingly better actions.

