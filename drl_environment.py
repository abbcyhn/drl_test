class Environment:
    def __init__(self):
        self.state_size = 4
        self.action_size = 4
    
    def reset(self):
        return [0, 0, 0, 0]
    
    def step(self, action):
        reward = 0
        done = False
        next_state = [0] * self.state_size

        for index, a in enumerate(action):
            if a == 1:
                done = True
                next_state[index] = 1
                if index == 3:
                    reward = 4
                else:
                    reward = -(index+1)

        return next_state, reward, done
