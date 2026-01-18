from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch

class EpisodeRunner:

    def __init__(self, args, logger, adversarial_manager):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.adv_manager = adversarial_manager

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()
        if self.adv_manager is not None:
            self.adv_manager.reset(self.batch_size)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, agent_outs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, return_agent_outs=True)

            if self.adv_manager is not None:
                # 准备数据 (Tensor/Numpy 转换)
                state_t = torch.tensor(self.env.get_state(), dtype=torch.float32).to(self.args.device).unsqueeze(0)
                avail_t = torch.tensor(self.env.get_avail_actions(), dtype=torch.float32).to(self.args.device).unsqueeze(0)
                obs_list = [self.env.get_obs()] # 保持 list 结构或转 numpy
                
                # 即使是 test_mode，如果 manager.mode != 'none'，这里也会返回被攻击的动作
                actions = self.adv_manager.perturb_actions(actions, state_t, avail_t, obs_list, test_mode, agent_outs)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            if self.adv_manager is not None:
                next_state = self.env.get_state()
                next_obs = self.env.get_obs()
                
                # 1. 判断是否是超时截断 (Time Limit)
                # SMAC 中 episode_limit 到达时 terminated=True，但这不算输
                truncated = env_info.get("episode_limit", False)
                
                # 2. 只有真正结束且不是超时，对于 PPO 来说 done 才是 True
                # 如果 truncated=True，done 应该为 False，这样 Critic 才会去预测下一状态价值 (Bootstrap)
                real_done = terminated and not truncated

                # 3. 只存储数据，绝对不要在这里 Update！
                self.adv_manager.after_step(
                    reward, 
                    real_done,       # 传给 Buffer 的 done
                    truncated,       # 标记是否截断
                    np.array([next_state]), 
                    np.array([next_obs])
                )

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # ================= [替换开始] =================
        # 1. 安全地累加 env_info (只累加数值类型，跳过列表)
        for k, v in env_info.items():
            # 判断 v 是否为数字 (兼容 numpy 类型)
            if isinstance(v, (int, float, bool, np.number)):
                if k not in cur_stats:
                    cur_stats[k] = 0.0
                cur_stats[k] += float(v)

        # 2. 记录基础统计
        if "n_episodes" not in cur_stats:
            cur_stats["n_episodes"] = 0
        cur_stats["n_episodes"] += 1

        if "ep_length" not in cur_stats:
            cur_stats["ep_length"] = 0
        cur_stats["ep_length"] += self.t

        # 3. 记录 Return 到 cur_stats (为了让 benchmark 函数能读取到)
        if "returns" not in cur_stats:
            cur_stats["returns"] = []
        cur_stats["returns"].append(episode_return)
        # ================= [替换结束] =================

        if not test_mode:
            self.t_env += self.t

        # PyMARL 原生逻辑保留 (self.test_returns 也存一份)
        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
             self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
             self._log(cur_returns, cur_stats, log_prefix)
             if hasattr(self.mac.action_selector, "epsilon"):
                 self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
             self.log_train_stats_t = self.t_env


        if self.adv_manager is not None and not test_mode:
            last_state = self.env.get_state()
            last_obs = self.env.get_obs()
            
            # 1. 接收统计数据
            adv_stats = self.adv_manager.on_episode_end(
                np.array([last_state]), 
                np.array([last_obs])
            )
            
            # 2. 如果有数据（意味着发生了 PPO 更新），则记录日志
            if adv_stats and len(adv_stats) > 0:
                # 这里的 self.t_env 是当前的时间步
                for k, v in adv_stats.items():
                    self.logger.log_stat(k, v, self.t_env)

        return self.batch

    # def _log(self, returns, stats, prefix):
    #     self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
    #     self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
    #     returns.clear()

    #     for k, v in stats.items():
    #         if k != "n_episodes":
    #             self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
    #     stats.clear()
    def _log(self, returns, stats, prefix):
        if self.args.test_nepisode > 10000:
            # 清空缓存但不打印
            returns.clear()
            stats.clear()
            return
        # 1. 记录标准的回报均值和方差 (PyMARL 原生逻辑)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        # 2. 遍历其他统计数据
        for k, v in stats.items():
            # 跳过 n_episodes 自身
            if k == "n_episodes":
                continue
            
            # [新增] 跳过列表类型的数据 (比如我们刚才加的 "returns")
            # 避免 TypeError: list / int
            if isinstance(v, list):
                continue
            
            # [新增] 确保 v 是数字才做除法
            if isinstance(v, (int, float, np.number)):
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        
        stats.clear()
