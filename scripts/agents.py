import os
import pickle
from sumo_rl import SumoEnvironment
from abc import ABC, abstractmethod
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

class LearningAgent():

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        self.config = config
        self.env = env
        self.agent = None
        self.name = name

    def _init_agent(self):
        pass

    def get_name(self) -> str:
        return self.name

    def run(self, learn: bool, out_path: str) -> str:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        pass

class FixedCycle(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        super().__init__(config, env, name)

    def _init_agent(self):
        self.agent = None

    def run(self, learn: bool, out_path: str) -> str:
        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            done = False
            self.env.reset()
            while not done:
                done = self._step()
            self.env.save_csv(out_file, curr_run)

        self.env.close()
        return out_path

    def save(self, path: str) -> None:
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        pass

    def _step(self) -> bool:
        for _ in range(self.env.delta_time):
            self.env._sumo_step()
            
        self.env._compute_observations()
        self.env._compute_rewards()
        self.env._compute_info()
        return self.env._compute_dones()['__all__']

class SarsaAgent(LearningAgent):
    
    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        super().__init__(config, env, name)

    def _init_agent(self):
        import gymnasium as gym
        import numpy as np
    
        if hasattr(self.env, 'ts_ids'):
            self.ts_ids = self.env.ts_ids
        elif hasattr(self.env, 'traffic_signals'):
            self.ts_ids = list(self.env.traffic_signals.keys())
        else:
            temp_obs = self.env.reset()
            self.ts_ids = list(temp_obs.keys())
            
        if hasattr(self.env, 'observation_space') and hasattr(self.env.observation_space, 'shape'):
            self.obs_dim = self.env.observation_space.shape[0]
        else:
            first_ts = self.ts_ids[0]
            self.obs_dim = self.env.observation_spaces[first_ts].shape[0]
            
        if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'n'):
            self.n_actions = self.env.action_space.n
        else:
            first_ts = self.ts_ids[0]
            self.n_actions = self.env.action_spaces[first_ts].n

        # 🔥 CAMBIO 1: El espacio de estados ahora es: Mi observación completa + Las fases del vecino
        # En sumo_rl, el estado de la fase se codifica como un vector binario de tamaño igual a n_actions
        extended_obs_dim = self.obs_dim + self.n_actions
        
        state_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(extended_obs_dim,),
            dtype=np.float32
        )
    
        joint_action_space = gym.spaces.Discrete(self.n_actions)
    
        self.internal_agents = {}
        for ts_id in self.ts_ids:
            self.internal_agents[ts_id] = TrueOnlineSarsaLambda(
                state_space=state_space,
                action_space=joint_action_space,
                alpha=self.config['Alpha'],
                gamma=self.config['Gamma'],
                epsilon=self.config['Epsilon'],
                fourier_order=self.config['FourierOrder'],
                lamb=self.config['Lambda']
            )

    def _build_phase_sharing_observations(self, obs_dict):
        """
        Construye el estado inyectando UNICAMENTE el vector de fase activa del vecino.
        En sumo_rl, los últimos 'n_actions' elementos de la observación corresponden a la fase activa.
        """
        import numpy as np
        ts_a, ts_b = self.ts_ids[0], self.ts_ids[1]
        
        # Extraer solo el vector One-Hot de la fase del final de la observación
        fase_a = obs_dict[ts_a][-self.n_actions:]
        fase_b = obs_dict[ts_b][-self.n_actions:]
        
        # Agente A ve: [Toda mi info de tráfico y mi fase, SOLO la fase de B]
        # Agente B ve: [Toda mi info de tráfico y mi fase, SOLO la fase de A]
        shared_obs = {
            ts_a: np.concatenate([obs_dict[ts_a], fase_b], dtype=np.float32),
            ts_b: np.concatenate([obs_dict[ts_b], fase_a], dtype=np.float32)
        }
        return shared_obs

    def run(self, learn: bool, out_path: str) -> str:
        if not hasattr(self, 'internal_agents') or not self.internal_agents:
            self._init_agent()

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            obs = self.env.reset()
            done = False
            
            while not done:
                action = {}
                
                # 🔥 CAMBIO 2: Construir el estado con fase compartida
                extended_obs = self._build_phase_sharing_observations(obs)
                
                for ts_id in self.ts_ids:
                    state = extended_obs[ts_id]
                    action[ts_id] = self.internal_agents[ts_id].act(state)
            
                next_obs, reward, dones, _ = self.env.step(action)
                done = dones['__all__']
            
                if learn:
                    # Construir el estado siguiente para el cálculo de SARSA
                    extended_next_obs = self._build_phase_sharing_observations(next_obs)
                    
                    for ts_id in self.ts_ids:
                        state = extended_obs[ts_id]
                        next_state = extended_next_obs[ts_id]
                        
                        self.internal_agents[ts_id].learn(
                            state=state,
                            action=action[ts_id],
                            reward=reward[ts_id],  
                            next_state=next_state,
                            done=dones[ts_id]
                        )
            
                obs = next_obs

            self.env.save_csv(out_file, curr_run)
        self.env.close()

        return out_path
    
    def save(self, path: str) -> None:
        data = {}
        for ts_id, agent in self.internal_agents.items():
            data[ts_id] = {
                'alpha': agent.alpha,
                'gamma': agent.gamma,
                'epsilon': agent.epsilon,
                'lamb': agent.lamb,
                'fourier_order': agent.basis.order,
            }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, env: SumoEnvironment) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.env = env
        self._init_agent()
        
        for ts_id in self.ts_ids:
            if ts_id in data:
                self.internal_agents[ts_id].alpha = data[ts_id]['alpha']
                self.internal_agents[ts_id].gamma = data[ts_id]['gamma']
                self.internal_agents[ts_id].epsilon = data[ts_id]['epsilon']
                self.internal_agents[ts_id].lamb = data[ts_id]['lamb']