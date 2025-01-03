from rtwtypes import *
import numpy as np
import os
import ctypes
import gym
from gym import spaces

class DLLModel:
    def __init__(self, model="dllModel"):
        self.model = model
        self.dll_path = os.path.abspath(f"{model}_win64.dll")
        self.dll = ctypes.windll.LoadLibrary(self.dll_path)

        self.Simulink_Step = 0.001 # Simulink model step size.
        self.Dt = int(1/self.Simulink_Step) # Control frequency per second.
        self.TotalStep = 1146

        self.__initialize = getattr(self.dll, f"{model}_initialize")
        self.__step = getattr(self.dll, f"{model}_step")
        self.__model_terminate = getattr(self.dll, f"{model}_terminate")

        self.variables = {
            'APUPCMD': real_T,
            'Step': real_T,
            'InputSignal': real_T,
        }

        self.variables_store = {
            'APUPCMD': real_T,
        }

        for var_name, var_type in self.variables.items():
            setattr(self, var_name, var_type.in_dll(self.dll, var_name))

    def init_log(self):
        self.data = {var_name: [] for var_name in self.variables}

    def initialize(self):
        self.step_num = 0
        self.done = False
        self.init_log()
        self.__initialize()

    def terminate(self):
        self.__model_terminate()

    def step(self):
        if not self.done:
            self.step_num += 1
            for Dtstep in range(self.Dt):
                self.__step()
                for var_name in self.variables_store:
                    self.data[var_name].append(getattr(self, var_name).value)
            self.done = (self.step_num == self.TotalStep)

    def get_state(self):
        return np.array([
            self.APUPCMD.value,
        ])

class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()
        self.model = DLLModel()

        self._max_episode_steps = self.model.TotalStep

        self.action_space = spaces.Box(low=-2.5, high=2.5, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64
        )

    def reset(self):
        self.model.terminate()
        self.model.initialize()
        return self.model.get_state()

    def step(self, action):
        self.model.InputSignal.value = action
        self.model.step()
        state = self.model.get_state()
        reward = self.calculate_reward(state)
        done = self.model.done
        return state, reward, done, {}

    def calculate_reward(self, state):
        return -10

    def close(self):
        self.model.terminate()


if __name__ == '__main__':
    env = Env()
    act_list = []
    dll_fbk_list = []

    s, done = env.reset(), False

    act_list.append(0)
    dll_fbk_list.append(s)
    t = 0

    while not done:
        a = t
        s, r, done, _ = env.step(a)
        act_list.append(a)
        dll_fbk_list.append(s)
        t += 1
    env.close()

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    x = range(len(act_list))
    y1 = act_list
    y2 = dll_fbk_list
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, linestyle='-', label='Act List Curve')
    plt.plot(x, y2, linestyle='-', label='DLL Feedback Curve')
    plt.title('Comparison of Action and DLL Feedback', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    annotation_text = (
        "DLL is twice as large as Act\n"
        "and is consistent with the Simulink model,\n"
        "indicating that the environment is correct."
    )
    plt.annotate(
        annotation_text,
        xy=(0.2, 0.5), xycoords='axes fraction',
        fontsize=10, color='blue',
        bbox=dict(boxstyle="round", fc="w", ec="blue", lw=1, alpha=0.5)
    )
    plt.legend()
    plt.grid(True)
    plt.show()



