import numpy as np
from drl_agent import DQNAgent
from drl_environment import Environment

def debug(line):
    #print(line)
    with open("results.txt", "a") as file:
        file.write(line + "\n")

def success(sc):
    if sc < 0:
        sc = 0
    with open("success.txt", "a") as file:
        file.write(str(sc) + "\n")

env = Environment()
state_size = env.state_size
action_size = env.action_size
agent = DQNAgent(state_size, action_size)

max_reward = 4
num_steps = 100
num_episodes = 1000
for episode in range(num_episodes):
    total_reward = 0
    for step in range(num_steps):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        while not done:
            action, actionBy = agent.act(state)

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        line = f"Episode: {episode+1}/{num_episodes}  Step: {step+1}/{num_steps}  Total Reward: {total_reward}  " + \
                f"Epsilon: {agent.epsilon:.2} Last Action: {actionBy}"

        # debug
        debug(line)
    success((total_reward/max_reward/num_steps)*100)

    agent.train_long_memory()

debug("Completed!")
