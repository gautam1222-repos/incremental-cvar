import numpy as np 

class Policy:

    def __init__(self, action_space):
        self.action_space = action_space
        self.params = np.ones(21)
        self.params = self.params.reshape((-1,1))
    
    def phi(self, state, action):
        state_action = np.array([])
        val = 0
        
        for i in range(len(state)):
            for j in range(len(state[i])):
                val += state[i][j]*2**j
            state_action = np.append(state_action, val)
                    
        state_action = np.append(state_action, action)

        state_action = state_action.reshape((-1,1))
        norm = np.linalg.norm(state_action)
        state_action = state_action*(1/(norm+1e-12))
        return state_action

    def softmax(self, state):
        softmax_arr = np.zeros(32)
        for a in self.action_space:
            softmax_arr[a] = self.policy(a, state)
        return softmax_arr


    def policy(self, action, state):
        Nr = np.exp(np.matmul(self.phi(state, action).T, self.params).item())
        Dr = 0
        for a in self.action_space:
            Dr += np.exp(np.matmul(self.phi(state, action).T, self.params).item())
        
        return Nr/Dr

    def log_policy(self, action, state):
        Nr = np.matmul(self.phi(state, action).T, self.params).item()
        Dr = 0
        for a in self.action_space:
            Dr += np.exp(np.matmul(self.phi(state, a).T, self.params).item())
        
        return Nr-np.log2(Dr)/np.log2(np.exp(1))
    
    def diff_log_policy_wrt_params(self, action, state):
        Nr = self.phi(state, action)

        Dr = 0
        dDr = 0
        for a in self.action_space:
            dDr += np.exp(np.matmul(self.phi(state, a).T, self.params).item()) * self.phi(state, a)
            Dr += np.exp(np.matmul(self.phi(state, a).T, self.params).item())
        
        return (Nr+dDr/Dr)