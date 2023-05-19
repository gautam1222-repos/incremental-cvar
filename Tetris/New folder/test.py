import numpy as np
from tetris import Tetris
import policy

def run(policy, timesteps):
    
    env = Tetris()
    log_trans_prob = 0

    state = env.get_initial_state()
    
    for _ in range(timesteps):
        print('-'*20,'>timestep: ',_+1)
        softmax_arr = policy.softmax(state)
        action = 31
        max_prob_trans = np.max(softmax_arr)
        
        for idx in range(len(softmax_arr)):
            if softmax_arr[idx] == max_prob_trans:
                action = idx
                break
        print('piece pos = ', env.current_piece.x, ',', env.current_piece.y)
        print(env.allowed_actions(env.valid_actions(env.grid, env.current_piece), env.current_piece))
        if action in range(20) and action not in env.allowed_actions(env.valid_actions(env.grid, env.current_piece), env.current_piece):
            action = 31
        print('*'*40)
        print('action = ', action)
        print('*'*40)
        next_state = env.step(action)
        log_trans_prob += policy.diff_log_policy_wrt_params(action, state)

        
        if not(env.run):
            break
        
        state = next_state
        
    env.close()
    total_score = env.score
    log_prob_traj = log_trans_prob
    del(env)
    return total_score, log_prob_traj

def test(var_step, gcvar_step, param_step, alpha, params, samples = 512):
    policy.params = params
    rewards = []
    log_prob_traj = []

    for _ in range(samples):
        r, p = run(policy, 400)
        rewards.append(r)
        log_prob_traj.append(p)
    
    sort_rew = rewards.copy()
    sort_rew.sort()

    var = sort_rew[int(samples*alpha)]

    # delta = np.zeros

    # for i in range(samples):
        

