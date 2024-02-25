from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ProjectAgent:
    def __init__(self, num_actions, gamma=0.99, iterations=400):
        self.num_actions = num_actions
        self.gamma = gamma
        self.iterations = iterations
        self.q_functions = None

    def fit(self, S, A, R, S2, D, disable_tqdm=False):
        self.q_functions = rf_fqi(S, A, R, S2, D, self.iterations, self.num_actions, self.gamma, disable_tqdm)

    def act(self, observation, use_random=False):
        if self.q_functions is None:
            raise ValueError("Agent must be trained before acting.")
        Q_values = np.zeros(self.num_actions)
        last_q = self.q_functions
        for a in range(self.num_actions):
            SA = np.hstack((observation, np.array([[a]] * len(observation))))
            Q_values[a] = last_q.predict(SA)
            return np.argmax(Q_values)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_functions, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_functions = pickle.load(f)

def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Q = None 
    SA = np.append(S, A, axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter == 0:
            value = R.copy()
        else:
            Q2 = np.zeros((nb_samples, nb_actions))
            for a2 in range(nb_actions):
                A2 = a2 * np.ones((S.shape[0], 1))
                S2A2 = np.append(S2, A2, axis=1)
                Q2[:, a2] = Q.predict(S2A2) if Q is not None else np.zeros(nb_samples)
            max_Q2 = np.max(Q2, axis=1)
            value = R + gamma * (1 - D) * max_Q2
        Q = RandomForestRegressor()
        Q.fit(SA, value)
    return Q  


def collect_samples(env, horizon, disable_tqdm=False):
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        s, _ = env.reset()
        for _ in range(horizon):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                break
            s = s2
        if done :
            print("done!")
    S = np.array(S)
    A = np.array(A).reshape((-1, 1))
    R = np.array(R)
    S2 = np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D
