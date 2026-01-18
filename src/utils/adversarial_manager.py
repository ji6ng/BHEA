import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import copy

# 导入两种攻击 Agent
from modules.agents.adversary_agent import AdversaryAgent        # 旧版 DQN
from modules.agents.budget_ppo_agent import BudgetPPOAgent      # 新版 PPO
import torch.nn as nn
def load_models(path: str, models, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    assert "models" in ckpt, "Invalid checkpoint format"

    # 兼容：如果传入的是 BudgetPPOAgent / nn.Module，就自动转成 dict
    if not isinstance(models, dict):
        # 尝试从对象上取子模块（按你保存时的 key）
        models = {k: getattr(models, k) for k in ckpt["models"].keys() if hasattr(models, k)}

    for name, sd in ckpt["models"].items():
        if name in models:
            models[name].load_state_dict(sd, strict=True)
            print(f"[Load] {name} loaded from {path}")
        else:
            print(f"[Warn] '{name}' not found in current agent; skipped")

    return ckpt.get("episode", None)


class AdversarialManager:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.use_cuda else "cpu")
        
        # --- 通用参数 ---
        self.mode = getattr(args, "adv_mode", "none")           # "random", "rarl", "rap", "none"
        self.method = getattr(args, "adv_method", "dqn")        # "dqn" (Count-based) or "budget_ppo" (Token-based)
        self.attack_prob = getattr(args, "adv_attack_prob", 0.1)
        self.pop_size = getattr(args, "adv_pop_size", 1) if self.mode == "rap" or self.mode == "vic" or self.mode == "vic1" else 1

        # --- DQN 专用参数 (Count Based) ---
        self.max_attacks_per_ep = getattr(args, "adv_max_attacks_per_episode", 10)
        self.max_agents_per_attack = getattr(args, "adv_max_agents_per_attack", 1)
        self.current_attack_counts = None  # 在 reset 中初始化

        self.epsilon = getattr(args, "adv_epsilon_start", 1.0)
        self.epsilon_min = getattr(args, "adv_epsilon_min", 0.05)
        self.epsilon_decay = getattr(args, "adv_epsilon_decay", 0.999) # 每个 episode 衰减

        self.target_update_interval = getattr(args, "adv_target_update_interval", 200)
        self.train_steps = 0  # 记录训练次数

        # --- PPO 专用参数 (Budget Based) ---
        self.max_budget_total = getattr(args, "adv_max_attack_budget", 10.0)
        self.current_episode_budgets = None # 在 reset 中初始化

        self.selection_mode = getattr(args, "budgetppo_selection_mode", "model")
        self.action_mode    = getattr(args, "budgetppo_action_mode", "model")
        self.heuristic_attack_prob = float(
            getattr(args, "heuristic_attack_prob", 0.4)
        )


        # --- 内部状态 ---
        self.adversaries = []
        self.optimizers = []
        self.active_adv_idx = 0
        self.phase = "victim"  # "victim" or "adversary"
    
    def _sample_random_valid_action(self, avail_1d: torch.Tensor) -> int:
        """
        avail_1d: [n_actions] 0/1
        返回一个随机可用动作 id
        """
        valid = torch.nonzero(avail_1d > 0, as_tuple=False).view(-1)
        if valid.numel() == 0:
            return 0  # 极端情况兜底
        j = torch.randint(0, valid.numel(), (1,), device=avail_1d.device).item()
        return int(valid[j].item())


    def setup(self, state_shape, n_agents, n_actions):
        """
        延迟初始化：等待 Runner 获取环境 Shape 后调用
        """
        self.args.state_shape = state_shape
        self.args.n_agents = n_agents
        self.args.n_actions = n_actions
        
        # 清空列表防止重复 setup
        self.adversaries = []
        self.optimizers = []
        self.target_adversaries = [] # [新增] 清空
        # print(self.mode)

        if self.mode in ["rarl", "rap", "vic", "vic1"]:
            for _ in range(self.pop_size):
                if self.method == "dqn":
                    # 初始化 DQN Agent
                    adv = AdversaryAgent(state_shape, self.args).to(self.device)
                    opt = optim.Adam(adv.parameters(), lr=self.args.lr)
                    self.adversaries.append(adv)
                    self.optimizers.append(opt)

                    target_adv = copy.deepcopy(adv)
                    self.target_adversaries.append(target_adv)
                
                elif self.method == "budget_ppo":
                    # 初始化 Budget PPO Agent
                    # PPO Agent 内部维护了自己的 optimizer 和 buffer
                    adv = BudgetPPOAgent(self.args).to(self.device)
                    self.adversaries.append(adv)
                    self.optimizers.append(None) # PPO 不需要外部 optimizer
                
        
        # print(f"[AdvManager] Init Done. Mode={self.mode}, Method={self.method}, PopSize={self.pop_size}")

    def add_adversary(self):
        adv = BudgetPPOAgent(self.args).to(self.device)
        self.adversaries.append(adv.to(self.device))
        self.optimizers.append(None) 
        print(f"[AdvManager] Added adversary. Total now: {len(self.adversaries)}")
    
    def load_adv(self, paths):
        # print(self.adversaries, len(self.adversaries))
        # print(paths, len(paths))
        for i in range(len(paths)):
            f_path = paths[i]
            # self.adversaries[i].load_state_dict(torch.load(f_path, map_location=self.device))
            load_models(
                f_path,
                self.adversaries[i],
                device=self.device
                )
            print(f"[AdvManager] Loaded adversary {i} from {f_path}")



    def reset(self, batch_size):
        """
        每个 Episode 开始前调用，重置计数器、预算，并决定本局是否攻击 (Episode-level Mask)。
        """
        # 1. 重置计数器和预算
        # DQN 计数器: [batch_size]
        self.current_attack_counts = torch.zeros(batch_size, device=self.device)
        # PPO 预算: [batch_size]
        self.current_episode_budgets = torch.full((batch_size,), self.max_budget_total, device=self.device)

        # 2. 【新增】初始化 Episode Active Mask (默认为 True，即攻击)
        self.episode_active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # 3. 【新增】Victim 训练阶段的混合策略
        # 只有在 victim 阶段，我们才随机让一些 episode 变成纯净环境 (Clean)
        # 如果是 "adversary" 阶段，则始终保持 True (全攻击)
        # if self.mode != "none" and self.phase == "victim":
        #     # 生成 0.5 的概率矩阵 (50% 概率攻击，50% 概率不攻击)
        #     # 你可以调整这个 0.5，比如 0.2 表示只有 20% 的局有攻击
        #     probs = torch.full((batch_size,), 1)
        #     self.episode_active_mask = torch.bernoulli(probs).bool().to(self.device)

    def sample_adversary(self):
        """
        决定使用哪个 Adversary (RAP 模式下随机采样)
        """
        if self.mode not in ["rarl", "rap", "vic", "vic1"] or not self.adversaries:
            return
        
        if self.mode == "rap":
            self.active_adv_idx = random.randint(0, len(self.adversaries) - 1)
        else:
            self.active_adv_idx = 0
        is_training = (self.mode in ["rarl", "rap"]) and (self.phase == "adversary")

        if is_training:
            self.active_adv_idx = -1
        elif self.mode=="vic":
            if len(self.adversaries)<=3:
                self.active_adv_idx = random.randint(0, len(self.adversaries) - 1)
            else:
                n = len(self.adversaries)
                start = int(0.6 * n)   # 只从后 40% 选
                self.active_adv_idx = random.randint(start, n - 1)

        elif self.mode=="vic1":
            self.active_adv_idx = -1
        # print(f"[AdvManager] Active Adversary Index: {self.active_adv_idx}")
    def perturb_actions(self, actions, state, avail_actions, obs, test_mode=False, agent_outs=None):
        """
        核心函数：根据策略篡改动作
        actions: [batch, n_agents] (Tensor)
        state: [batch, state_dim] (Tensor)
        avail_actions: [batch, n_agents, n_actions] (Tensor)
        obs: [batch, n_agents, obs_dim] (List/Numpy usually from Runner, need check)
        """
        if self.mode == "none":
            return actions

        batch_size, n_agents = actions.shape
        perturbed_actions = actions.clone()

        # 确保计数器已初始化
        if self.current_attack_counts is None or self.current_attack_counts.shape[0] != batch_size:
            self.reset(batch_size)

        # ======================================================
        # 策略 1: Random Attack
        # ======================================================
        if self.mode == "random":
            # 简单的随机攻击逻辑 (沿用之前的)
            mask = (torch.rand_like(actions.float()) < self.attack_prob).long()
            # print("mask:", mask)
            has_budget = (self.current_attack_counts < self.max_attacks_per_ep)
            # print("has_budget:", has_budget, has_budget, self.current_attack_counts)
            should_attack = mask & has_budget.unsqueeze(1) # Broadcast check
            # print("should_attack:", should_attack)

            if should_attack.sum() == 0:
                return actions

            random_vals = torch.rand(batch_size, n_agents, self.args.n_actions).to(self.device)
            random_vals[avail_actions == 0] = -1e10
            noise_actions = random_vals.argmax(dim=-1)
            
            perturbed_actions = torch.where(should_attack == 1, noise_actions, actions)
            
            # 更新计数 (只要这一步有任意 agent 被攻击，该 batch 的计数 +1)
            # batch_attacked = should_attack.any(dim=1)
            # print("should_attack:", should_attack)
            # print("should_attack:", should_attack.bool(),should_attack.bool().any(dim=1))
            batch_attacked = should_attack.bool().any(dim=1)
            # print(1,self.current_attack_counts)
            self.current_attack_counts[batch_attacked] += 1
            # print(2,self.current_attack_counts)           
            return perturbed_actions

        # ======================================================
        # 策略 2: Adversarial Attack (RARL / RAP)
        # ======================================================
        if not self.adversaries:
            return actions

        # ---------------- A. DQN 方法 (Count Based) ----------------
        if self.method == "dqn":
            # 1. 预算检查
            has_budget = (self.current_attack_counts < self.max_attacks_per_ep) # [batch]
            triggered = (torch.rand(batch_size, device=self.device) < self.attack_prob) # [batch]
            active_batch_mask = has_budget & triggered

            if not active_batch_mask.any():
                return actions

            # 2. 生成攻击动作
            adv_net = self.adversaries[self.active_adv_idx]

            if not test_mode and self.phase == "adversary":
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            with torch.no_grad():
                q_vals = adv_net(state) # [batch, n_agents, n_actions]
            
            q_vals[avail_actions == 0] = -1e10

            use_greedy = test_mode or (self.phase == "victim") or (random.random() > self.epsilon)
            
            if use_greedy:
                adv_actions = q_vals.argmax(dim=-1)
            else:
                # 随机选择可用动作
                # 这里的随机逻辑需要处理 avail_actions
                random_vals = torch.rand_like(q_vals)
                random_vals[avail_actions == 0] = -1e10
                adv_actions = random_vals.argmax(dim=-1)

            # 3. 随机选 k 个智能体进行攻击 (防止全员被控)
            # 这里简化处理：全员攻击，然后用 Mask 过滤
            target_mask = torch.zeros_like(actions, dtype=torch.bool)
            
            # 对每个 active 的 batch，随机选 k 个
            for b in torch.where(active_batch_mask)[0]:
                indices = torch.randperm(n_agents)[:self.max_agents_per_attack]
                target_mask[b, indices] = True

            # 4. 应用
            perturbed_actions = torch.where(target_mask, adv_actions, actions)
            
            # 5. 更新计数
            self.current_attack_counts[active_batch_mask] += 1
            
            return perturbed_actions

        # ---------------- B. Budget PPO 方法 (Token Based) ----------------
        # elif self.method == "budget_ppo":
        #     adv_agent = self.adversaries[self.active_adv_idx]
        #     is_training = (self.mode in ["rarl", "rap"]) and (self.phase == "adversary") and not test_mode
        #     # print("is_training:", is_training, self.mode, self.phase, test_mode)
            
        #     # 注意：BudgetPPOAgent 设计为处理单步交互流。
        #     # 如果 batch_size > 1，我们需要循环处理 (或修改 Agent 支持 Batch)
        #     # PyMARL EpisodeRunner 默认 batch_size 通常为 1 (Running 1 env)
        #     # 为安全起见，这里做一个简单的循环处理
            
        #     obs_np = np.array(obs) # Runner 传入的可能是 list
            
        #     for b in range(batch_size):
        #         # 检查预算
        #         curr_bud = self.current_episode_budgets[b].item()
        #         # if curr_bud <= 0:
        #         #     continue

        #         # 从 Agent 获取决策
        #         # Agent 内部会处理 GNN, Selector 等逻辑
        #         attacked_agents, adv_actions = adv_agent.sample_action(
        #             obs_np[b],                      # [n_agents, obs_dim]
        #             state[b].cpu().numpy(),         # [state_dim]
        #             avail_actions[b].cpu().numpy(), # [n_agents, n_actions]
        #             curr_bud,
        #             self.max_budget_total,
        #             training_mode=is_training,
        #         )

        #         # 如果这一 step 有攻击，则对所有被攻击的 agent 替换动作为对抗动作
        #         if len(attacked_agents) > 0:
        #             for idx in attacked_agents:
        #                 # adv_actions 是长度为 n_agents 的数组，未攻击位置为 -1
        #                 if adv_actions[idx] >= 0:
        #                     perturbed_actions[b, idx] = adv_actions[idx]

        #             # ❗每个 step 只扣一次预算，与之前语义保持一致
        #             self.current_episode_budgets[b] -= 1
            
        #     return perturbed_actions
        elif self.method == "budget_ppo":
            adv_agent = self.adversaries[self.active_adv_idx]
            is_training = (self.mode in ["rarl", "rap"]) and (self.phase == "adversary") and (not test_mode)

            obs_np = np.array(obs)  # [batch, n_agents, obs_dim]

            def sample_random_valid_action(avail_1d: torch.Tensor) -> int:
                valid = torch.nonzero(avail_1d > 0, as_tuple=False).view(-1)
                if valid.numel() == 0:
                    return 0
                j = torch.randint(0, valid.numel(), (1,), device=avail_1d.device).item()
                return int(valid[j].item())

            for b in range(batch_size):
                curr_bud = int(self.current_episode_budgets[b].item())

                # ========== budget 约束 ==========
                if curr_bud <= 0:
                    continue

                attacked_any = False

                # ======================================================
                # 1️⃣ selector 消融：random2（随机选攻击对象）
                # ======================================================
                if self.selection_mode == "random2":
                    if np.random.rand() < self.heuristic_attack_prob:
                        sel_targets = [int(np.random.randint(0, n_agents))]
                    else:
                        sel_targets = [n_agents]  # noop

                # ======================================================
                # 2️⃣ 正常 selector：Budget-PPO 模型
                # ======================================================
                else:  # selection_mode == "model"
                    attacked_agents, adv_actions = adv_agent.sample_action(
                        obs_np[b],
                        state[b].cpu().numpy(),
                        avail_actions[b].cpu().numpy(),
                        curr_bud,
                        self.max_budget_total,
                        training_mode=is_training,
                    )
                    if len(attacked_agents) == 0:
                        continue
                    sel_targets = attacked_agents

                # ======================================================
                # 3️⃣ 对选中的 target 执行动作攻击
                # ======================================================
                for idx in sel_targets:
                    if idx >= n_agents:
                        break

                    t = int(idx)
                    avail_1d = avail_actions[b, t]

                    # -------- action 消融：random --------
                    if self.action_mode == "random":
                        attack_action = sample_random_valid_action(avail_1d)
                    
                    elif self.action_mode == "qmix":
                        # agent_outs: [batch, n_agents, n_actions] = QMIX victim 的 Q
                        q_vals_agent = agent_outs[b, t].clone()

                        # mask 不可行动作
                        q_vals_agent[avail_1d == 0] = 1e10

                        # worst action = argmin Q
                        attack_action = int(torch.argmin(q_vals_agent).item())

                    # -------- 正常 action：Budget-PPO / attacker --------
                    else:  # action_mode == "model"
                        # random2 时 adv_actions 可能不存在，需兜底
                        if self.selection_mode == "model" and adv_actions[t] >= 0:
                            attack_action = int(adv_actions[t])
                        else:
                            attack_action = sample_random_valid_action(avail_1d)

                    # 没有可行动作，不算成功攻击
                    if (avail_1d > 0).sum().item() == 0:
                        continue

                    perturbed_actions[b, t] = attack_action
                    attacked_any = True

                    # attack_n == 1 时只攻击一个
                    if int(self.max_agents_per_attack) == 1:
                        break

                # ======================================================
                # 4️⃣ budget 消耗（严格一次）
                # ======================================================
                if attacked_any:
                    self.current_episode_budgets[b] -= 1

            return perturbed_actions

        return actions
        
   
    def update_budget(self, new_budget):
        """
        用于外部 (main.py) 更新当前的 Budget 上限。
        该值的改变会在下一次 reset() 时生效。
        """
        self.max_budget_total = new_budget
        # 如果需要，也可以同步更新 args 中的值，以防其他地方用到
        self.args.adv_max_attack_budget = new_budget
    # def after_step(self, reward, terminated, next_state, next_obs):
    def after_step(self, reward, done, truncated, next_state, next_obs):
        """
        [PPO 专用] 在环境 Step 后调用。
        
        """

        if self.method != "budget_ppo" or self.mode not in ["rarl", "rap"] or not self.adversaries:
            return
        adv_agent = self.adversaries[self.active_adv_idx]
        # 传递 truncated 标志给 agent
        adv_agent.store_reward(reward, done, truncated)

        # 更新 Budget (如果有 Budget 限制逻辑)
        if hasattr(self, "current_budget"):
            # 假设你这里有扣除 budget 的逻辑
            pass

       
    def on_episode_end(self, last_state, last_obs):
            # if self.mode == "none" or self.phase != "adversary":
            #     return {} # 返回空字典
            if self.method != "budget_ppo" or self.mode not in ["rarl", "rap"] or self.phase != "adversary":
                return {}

            # 调用 Agent，并接收返回值
            adv_agent = self.adversaries[self.active_adv_idx]
            train_stats = adv_agent.process_episode(
                last_state, 
                last_obs, 
                max_budget=getattr(self.args, "adv_budget", 10.0),
                current_budget=getattr(self, "current_budget", 0.0)
            )
            
            return train_stats # 继续往上传


    def train_adversary(self, batch):
        if self.method != "dqn" or self.mode not in ["rarl", "rap"] or self.phase != "adversary":
            return

        # 1. 准备数据
        # state: [batch, seq_len, state_dim]
        state = batch["state"][:, :-1].to(self.device)
        state_next = batch["state"][:, 1:].to(self.device)
        reward = batch["reward"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        actions = batch["actions"][:, :-1].to(self.device)
        
        # [新增] 获取 avail_actions 用于 Masking
        # 注意：PyMARL 中 avail_actions 通常存储在 batch 中
        avail_actions = batch["avail_actions"][:, :-1].to(self.device)
        avail_actions_next = batch["avail_actions"][:, 1:].to(self.device)

        # Adversary Reward (建议缩放，例如 / 10.0 或标准化)
        adv_reward = -1.0 * reward 

        adv_net = self.adversaries[self.active_adv_idx]
        target_adv_net = self.target_adversaries[self.active_adv_idx]
        optimizer = self.optimizers[self.active_adv_idx]

        # ==============================================================================
        # 2. 计算 Q_Eval (当前状态的 Q 值)
        # ==============================================================================
        # [处理维度] PyMARL 网络通常接受 [batch * time, input_dim]
        batch_size = state.shape[0]
        time_len = state.shape[1]
        
        # 将 state 展平: [32, 60, 100] -> [1920, 100]
        state_flat = state.reshape(-1, state.shape[-1])
        
        # [注意] 如果 adv_net 是 RNN，这里必须传入 hidden_state 并循环处理
        # 这里假设 adv_net 是 MLP (全连接网络)，可以直接处理整个 batch
        q_evals_all_flat = adv_net(state_flat) 
        
        # 恢复维度: [1920, 11] -> [32, 60, 5, 11] (假设 n_agents=5)
        # 注意：如果 Adversary 是控制单个 Agent 的，这里的维度需要根据你的架构调整
        # 假设 Adversary 输出针对所有 agent 的动作 Q 值
        n_agents = q_evals_all_flat.shape[1] if len(q_evals_all_flat.shape) > 2 else 1
        n_actions = q_evals_all_flat.shape[-1]
        
        # 重新 Reshape 回 [batch, time, n_agents, n_actions]
        # 如果 adv_net 输出已经是 [batch*time, n_agents, n_actions]，则如下：
        q_evals_all = q_evals_all_flat.view(batch_size, time_len, n_agents, n_actions)

        # 取出实际执行动作对应的 Q 值
        q_eval = torch.gather(q_evals_all, dim=3, index=actions).squeeze(3) # [batch, time, n_agents]
        
        # VDN 聚合: Sum over agents
        q_eval_tot = q_eval.sum(dim=2, keepdim=True) # [batch, time, 1]

        # ==============================================================================
        # 3. 计算 Q_Target (目标状态的 Q 值) - Double DQN + Masking
        # ==============================================================================
        with torch.no_grad():
            state_next_flat = state_next.reshape(-1, state_next.shape[-1])
            
            # A. 使用 Eval 网络选择动作 (Double DQN)
            q_next_eval_flat = adv_net(state_next_flat)
            q_next_eval = q_next_eval_flat.view(batch_size, time_len, n_agents, n_actions)
            
            # [关键修补] Mask 掉不可用动作
            q_next_eval[avail_actions_next == 0] = -1e10
            
            # 获得最大 Q 值对应的动作 Index
            max_action_next = q_next_eval.argmax(dim=3, keepdim=True) # [batch, time, n_agents, 1]

            # B. 使用 Target 网络评估动作
            q_next_target_flat = target_adv_net(state_next_flat)
            q_next_target = q_next_target_flat.view(batch_size, time_len, n_agents, n_actions)

            # 根据刚才选出的动作，在 Target 网络中查表
            q_next_taken = torch.gather(q_next_target, dim=3, index=max_action_next).squeeze(3)
            
            # VDN 聚合
            q_next_taken_tot = q_next_taken.sum(dim=2, keepdim=True)

            # C. 计算 TD Target
            q_target = adv_reward + self.args.gamma * (1 - terminated) * q_next_taken_tot

        # ==============================================================================
        # 4. 计算 Loss 并更新
        # ==============================================================================
        td_error = (q_eval_tot - q_target)
        
        # mask 用于处理填充的序列 (pad)
        loss = (td_error ** 2 * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adv_net.parameters(), self.args.grad_norm_clip)
        optimizer.step()

        # Update Target Net
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            target_adv_net.load_state_dict(adv_net.state_dict())

    def save_models(self, path, name):
        """
        保存攻击者模型
        """
        if self.mode not in ["rarl", "rap"] or not self.adversaries:
            return
        
        import os
        save_path = os.path.join(path, "adversary")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for i, adv in enumerate(self.adversaries):
            # 保存模型参数
            torch.save(adv.state_dict(), os.path.join(save_path, f"{name}_adv_{i}.th"))
        
        print(f"[AdvManager] Saved {len(self.adversaries)} adversaries to {save_path}")

    def load_models(self, path):
        """
        加载攻击者模型
        """
        if self.mode not in ["rarl", "rap"]:
            return

        import os
        import glob
        
        # 假设路径是 results/models/.../adversary/
        # 或者用户传入的是 checkpoint 根目录
        load_path = os.path.join(path, "adversary")
        if not os.path.exists(load_path):
            print(f"[AdvManager] Warning: No adversary model found at {load_path}")
            return

        # 寻找模型文件
        files = sorted(glob.glob(os.path.join(load_path, "*_adv_*.th")))
        if not files:
            print(f"[AdvManager] Warning: No .th files found in {load_path}")
            return

        # 重新初始化列表 (如果数量不匹配)
        # 这里假设 pop_size 和保存的一致
        for i, f_path in enumerate(files):
            if i >= len(self.adversaries):
                break
            
            self.adversaries[i].load_state_dict(torch.load(f_path, map_location=self.device))
            print(f"[AdvManager] Loaded adversary {i} from {f_path}")
            
    def can_safely_switch(self):
        """
        检查当前是否可以安全切换阶段。
        对于 PPO，必须等 Batch 训完才能切。
        对于 DQN，随时可以切。
        """
        if self.method != "budget_ppo":
            return True
        
        # 只有当所有 Attacker 的 buffer 都空了，才允许切换
        for adv in self.adversaries:
            if adv.has_pending_data():
                return False
        
        return True

