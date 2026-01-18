# # -*- coding: utf-8 -*-
# # Drop-in replacement for: src/modules/agents/budget_ppo_agent.py
# # EXACT training logic alignment with your top script:
# #   - attack_n==1 : CODE-A (gate+k1, strict logp recompute, bootstrap next_value)
# #   - attack_n>1  : CODE-B (FORCE_FILL sequential, sel_seq, autoreg logp, next_value=None)
# #   - Analyst update RNG alignment: A full update (no randperm), B randperm subset
# #
# # Patch in this version:
# #   - FORCE selector/attacker sampling + PPO Categorical to run in FP32 and with AMP disabled
# #     (prevents multinomial CUDA assert val>=0 caused by NaN/Inf under autocast / fp16 masking)

# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical
# from contextlib import nullcontext

# EPS = 1e-8


# def amp_off(device):
#     # Disable autocast locally (important if outer learner enables AMP)
#     if str(device).startswith("cuda"):
#         return torch.cuda.amp.autocast(enabled=False)
#     return nullcontext()


# def to_tensor(x, device):
#     return torch.as_tensor(x, dtype=torch.float32, device=device)


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


# def build_budget_action_mask(n_agents: int, current_budget: int, device):
#     """[N+1] budget-only mask: budget<=0 => only noop allowed"""
#     mask = torch.ones(n_agents + 1, dtype=torch.float32, device=device)
#     if current_budget <= 0:
#         mask[:-1] = 0.0
#     return mask


# # =========================
# # Networks (same as your script)
# # =========================
# class GNNEncoder(nn.Module):
#     def __init__(self, obs_dim, embed_dim):
#         super().__init__()
#         self.self_mlp = layer_init(nn.Linear(obs_dim, embed_dim))
#         self.nei_mlp = layer_init(nn.Linear(obs_dim, embed_dim))

#     def forward(self, obs_all):
#         # obs_all: [B,N,obs]
#         x_self = self.self_mlp(obs_all)
#         mean_nei = obs_all.mean(dim=1, keepdim=True)
#         msg = self.nei_mlp(mean_nei)
#         h = torch.relu(x_self + msg)
#         return h


# class VictimSelectorBudget(nn.Module):
#     def __init__(self, embed_dim, state_dim, hidden_dim=128):
#         super().__init__()
#         self.global_feature_net = nn.Sequential(
#             layer_init(nn.Linear(embed_dim + state_dim + 1, hidden_dim)),
#             nn.ReLU(),
#         )
#         self.agent_policy_net = nn.Sequential(
#             layer_init(nn.Linear(embed_dim + hidden_dim, hidden_dim)),
#             nn.ReLU(),
#             layer_init(nn.Linear(hidden_dim, 1), std=0.01),
#         )
#         self.noop_policy_net = nn.Sequential(
#             layer_init(nn.Linear(hidden_dim, hidden_dim)),
#             nn.ReLU(),
#             layer_init(nn.Linear(hidden_dim, 1), std=0.01),
#         )
#         self.v_net = nn.Sequential(
#             layer_init(nn.Linear(hidden_dim, hidden_dim)),
#             nn.ReLU(),
#             layer_init(nn.Linear(hidden_dim, 1), std=1.0),
#         )

#     def forward(self, agent_embeds, state_batch, budget_ratio):
#         # agent_embeds:[B,N,E], state_batch:[B,S], budget_ratio:[B,1] or [B]
#         B, N, E = agent_embeds.shape
#         global_embed = agent_embeds.mean(dim=1)
#         if budget_ratio.dim() == 1:
#             budget_ratio = budget_ratio.unsqueeze(1)

#         global_input = torch.cat([global_embed, state_batch, budget_ratio], dim=-1)
#         global_feat = self.global_feature_net(global_input)

#         global_feat_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
#         agent_input = torch.cat([agent_embeds, global_feat_expanded], dim=-1)

#         agent_logits = self.agent_policy_net(agent_input).squeeze(-1)  # [B,N]
#         noop_logit = self.noop_policy_net(global_feat)                 # [B,1]
#         total_logits = torch.cat([agent_logits, noop_logit], dim=-1)   # [B,N+1]

#         v_value = self.v_net(global_feat).squeeze(-1)                  # [B]
#         return total_logits, v_value


# class ActionAttackerMulti(nn.Module):
#     def __init__(self, obs_dim, embed_dim, n_actions, hidden_dim=128):
#         super().__init__()
#         self.policy_mlp = nn.Sequential(
#             layer_init(nn.Linear(obs_dim + embed_dim, hidden_dim)),
#             nn.ReLU(),
#             layer_init(nn.Linear(hidden_dim, n_actions), std=0.01),
#         )
#         self.value_mlp = nn.Sequential(
#             layer_init(nn.Linear(obs_dim + embed_dim, hidden_dim)),
#             nn.ReLU(),
#             layer_init(nn.Linear(hidden_dim, 1), std=1.0),
#         )

#     def forward(self, obs_all, agent_embeds, avail_actions_all=None):
#         # obs_all:[B,N,obs], agent_embeds:[B,N,E], avail_actions_all:[B,N,A]
#         x = torch.cat([obs_all, agent_embeds], dim=-1)
#         logits = self.policy_mlp(x)  # [B,N,A]
#         if avail_actions_all is not None:
#             logits = logits.masked_fill(avail_actions_all == 0, -1e10)
#         values = self.value_mlp(x).squeeze(-1)  # [B,N]
#         return logits, values

#     def act_single(self, obs_i, embed_i, avail_actions_i, device):
#         obs_t = torch.as_tensor(obs_i, dtype=torch.float32, device=device).view(1, 1, -1)
#         emb_t = torch.as_tensor(embed_i, dtype=torch.float32, device=device).view(1, 1, -1)
#         avail_t = torch.as_tensor(avail_actions_i, dtype=torch.float32, device=device).view(1, 1, -1)

#         # AMP OFF + FP32 logits before Categorical
#         with amp_off(device):
#             logits, values = self.forward(obs_t, emb_t, avail_t)  # logits [1,1,A]
#             logits = logits.float()
#             dist = Categorical(logits=logits)
#             action = dist.sample()
#             log_prob = dist.log_prob(action)
#         return int(action.item()), log_prob.detach(), values.detach()


# class BudgetAnalyst(nn.Module):
#     def __init__(self, embed_dim, state_dim, hidden_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             layer_init(nn.Linear(embed_dim + state_dim + 1, hidden_dim)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_dim, hidden_dim)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_dim, 1), std=1.0),
#         )
#         self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         self.loss_fn = nn.MSELoss()

#     def forward(self, global_embed, state, budget_ratio):
#         x = torch.cat([global_embed, state, budget_ratio], dim=-1)
#         return self.net(x)

#     def update(self, global_embed, state, budget_ratio, returns):
#         pred = self.forward(global_embed, state, budget_ratio).squeeze(-1)
#         loss = self.loss_fn(pred, returns)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return float(loss.item())


# # =========================
# # CODE-A selector sampling (attack_n==1)
# # =========================
# def sample_selector_gate_k1(logits_sel_1x, budget_mask_1d, eps=EPS):
#     masked_logits = logits_sel_1x.clone()
#     masked_logits[budget_mask_1d == 0] = -1e10

#     probs_full = F.softmax(masked_logits, dim=-1)
#     p_noop = probs_full[-1].clamp(min=eps, max=1.0)
#     p_attack = (1.0 - p_noop).clamp(min=eps, max=1.0)

#     if budget_mask_1d[:-1].sum().item() < 0.5:
#         return False, -1, float(torch.log(p_noop + eps).item())

#     if random.random() > float(p_attack.item()):
#         return False, -1, float(torch.log(p_noop + eps).item())

#     agent_probs = probs_full[:-1].clamp(min=eps)
#     agent_probs = agent_probs / agent_probs.sum()

#     vid = int(torch.multinomial(agent_probs, num_samples=1, replacement=False).item())
#     lp = float(torch.log(p_attack + eps).item() + torch.log(agent_probs[vid] + eps).item())
#     return True, vid, lp


# def recompute_logp_selector_gate_k1(masked_logits_bn1, idx_seq_bk, attacked_bool, eps=EPS):
#     probs_full = F.softmax(masked_logits_bn1, dim=-1)
#     p_noop = probs_full[:, -1].clamp(min=eps, max=1.0)
#     p_attack = (1.0 - p_noop).clamp(min=eps, max=1.0)

#     agent_logits = masked_logits_bn1[:, :-1]
#     agent_logp = F.log_softmax(agent_logits, dim=-1)

#     logp = torch.log(p_noop + eps)
#     if attacked_bool.any():
#         vid = idx_seq_bk[:, 0].clamp(min=0)
#         lp_attack = torch.log(p_attack + eps) + agent_logp.gather(1, vid.view(-1, 1)).squeeze(1)
#         logp = torch.where(attacked_bool, lp_attack, logp)

#     gate_ent = -(p_noop * torch.log(p_noop) + p_attack * torch.log(p_attack)).mean()
#     agent_probs = F.softmax(agent_logits, dim=-1)
#     agent_ent = -(agent_probs * agent_logp).sum(dim=1).mean()
#     entropy = gate_ent + agent_ent
#     return logp, entropy


# # =========================
# # CODE-B selector sampling (attack_n>1): FORCE_FILL
# # =========================
# def sample_with_budget_mask_strict(logits_1x, current_budget, device, exclude_mask=None, forbid_noop=False):
#     logits = logits_1x.squeeze(0)  # [N+1]
#     base_mask = torch.ones_like(logits, device=device)
#     if current_budget <= 0:
#         base_mask[:-1] = 0.0

#     combined_mask = base_mask.clone()
#     if exclude_mask is not None:
#         combined_mask[:-1] *= (1.0 - exclude_mask.squeeze(0))
#     if forbid_noop:
#         combined_mask[-1] = 0.0

#     # ===== 兜底：如果一个都不能选，就强制返回 noop（或解除 forbid_noop）=====
#     if combined_mask.sum().item() < 0.5:
#         noop_idx = logits.numel() - 1
#         action_idx = torch.tensor(noop_idx, device=device, dtype=torch.long)
#         log_prob = torch.tensor(0.0, device=device, dtype=torch.float32)
#         return action_idx, log_prob, base_mask

#     # ===== fp16-safe：别用 -1e10，改成 dtype 安全的负大数 =====
#     NEG = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e10
#     masked_logits = logits.clone()
#     masked_logits[combined_mask == 0] = NEG

#     dist = Categorical(logits=masked_logits)
#     action_idx = dist.sample()
#     log_prob = dist.log_prob(action_idx)
#     return action_idx, log_prob, base_mask


# def compute_new_lp_sel_autoregressive(logits_sel, sel_seq, act_masks):
#     device = logits_sel.device
#     B, NA1 = logits_sel.shape
#     N = NA1 - 1
#     K = sel_seq.shape[1]

#     new_lp = torch.zeros(B, device=device)
#     exclude = torch.zeros(B, N, device=device)
#     attacked_already = torch.zeros(B, dtype=torch.bool, device=device)

#     for k in range(K):
#         idx = sel_seq[:, k]
#         valid = (idx >= 0)
#         if valid.sum() == 0:
#             continue

#         masked_logits_k = logits_sel.clone()
#         masked_logits_k[act_masks == 0] = -1e10
#         masked_logits_k[:, :N] = masked_logits_k[:, :N].masked_fill(exclude > 0.5, -1e10)

#         forbid = attacked_already
#         if forbid.any():
#             masked_logits_k[forbid, -1] = -1e10

#         logp_all = F.log_softmax(masked_logits_k, dim=-1)
#         new_lp[valid] += logp_all[valid, idx[valid]]

#         is_victim = valid & (idx < N)
#         attacked_already = attacked_already | is_victim
#         if is_victim.any():
#             exclude[is_victim, idx[is_victim]] = 1.0

#     return new_lp


# # =========================
# # Buffer: stores BOTH idx_seq (A) and sel_seq (B)
# # =========================
# class PPORolloutBuffer:
#     def __init__(self, max_k: int):
#         self.max_k = int(max_k)
#         self.clear()

#     def clear(self):
#         self.states, self.obs_all, self.avail_all = [], [], []
#         self.victim_mask, self.adv_actions = [], []
#         self.idx_seq, self.sel_seq = [], []          # [K] padded -1
#         self.logprob_sel, self.logprob_att = [], []
#         self.value_sel, self.value_att = [], []
#         self.rewards, self.dones = [], []
#         self.budgets, self.action_masks = [], []     # [N+1]

#     def add(self, state, obs, avail, v_mask, adv_act,
#             idx_seq, sel_seq,
#             lp_sel, lp_att, v_sel, v_att, r, d, bud, act_mask):
#         self.states.append(state)
#         self.obs_all.append(obs)
#         self.avail_all.append(avail)
#         self.victim_mask.append(v_mask)
#         self.adv_actions.append(adv_act)

#         self.idx_seq.append(idx_seq)
#         self.sel_seq.append(sel_seq)

#         self.logprob_sel.append(lp_sel)
#         self.logprob_att.append(lp_att)
#         self.value_sel.append(v_sel)
#         self.value_att.append(v_att)
#         self.rewards.append(r)
#         self.dones.append(d)
#         self.budgets.append(bud)
#         self.action_masks.append(act_mask)

#     def get_data(self, device, gamma=0.99, gae_lambda=0.95, next_value=None):
#         states = to_tensor(np.array(self.states), device)                     # [T,S]
#         obs = to_tensor(np.array(self.obs_all), device)                       # [T,N,obs]
#         avail = to_tensor(np.array(self.avail_all), device)                   # [T,N,A]
#         v_mask = to_tensor(np.array(self.victim_mask), device)                # [T,N]
#         adv_act = torch.as_tensor(np.array(self.adv_actions), dtype=torch.long, device=device)  # [T,N]

#         idx_seq = torch.as_tensor(np.array(self.idx_seq), dtype=torch.long, device=device)      # [T,K]
#         sel_seq = torch.as_tensor(np.array(self.sel_seq), dtype=torch.long, device=device)      # [T,K]

#         rewards = to_tensor(np.array(self.rewards), device)                   # [T]
#         dones = to_tensor(np.array(self.dones), device)                       # [T]
#         budgets = to_tensor(np.array(self.budgets), device).unsqueeze(1)      # [T,1]
#         act_masks = to_tensor(np.array(self.action_masks), device)            # [T,N+1]

#         old_lp_sel = to_tensor(np.array(self.logprob_sel), device)            # [T]
#         old_lp_att = to_tensor(np.array(self.logprob_att), device)            # [T]
#         old_v_sel = to_tensor(np.array(self.value_sel), device)               # [T]
#         old_v_att = to_tensor(np.array(self.value_att), device)               # [T]

#         # GAE based on selector value (EXACT)
#         with torch.no_grad():
#             values = old_v_sel
#             advantages = torch.zeros_like(rewards)
#             lastgaelam = 0.0
#             T = len(rewards)
#             for t in reversed(range(T)):
#                 if t == T - 1:
#                     nextnonterminal = 1.0 - dones[t]
#                     nextvalues = next_value if next_value is not None else 0.0
#                 else:
#                     nextnonterminal = 1.0 - dones[t]
#                     nextvalues = values[t + 1]
#                 delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
#                 advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
#             returns = advantages + values

#         return {
#             "states": states, "obs": obs, "avail": avail,
#             "v_mask": v_mask, "adv_act": adv_act,
#             "idx_seq": idx_seq, "sel_seq": sel_seq,
#             "budgets": budgets, "act_masks": act_masks,
#             "old_lp_sel": old_lp_sel, "old_lp_att": old_lp_att,
#             "old_v_sel": old_v_sel, "old_v_att": old_v_att,
#             "advantages": advantages, "returns": returns,
#         }


# # =========================
# # Main Agent class
# # =========================
# class BudgetPPOAgent(nn.Module):
#     """
#     PyMARL adversarial agent, training logic aligned to your top script.
#     """
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.device = torch.device("cuda" if getattr(args, "use_cuda", False) and torch.cuda.is_available() else "cpu")

#         self.obs_dim = int(args.obs_shape)
#         self.state_dim = int(args.state_shape)
#         self.n_actions = int(args.n_actions)
#         self.n_agents = int(args.n_agents)

#         self.attack_n = int(getattr(args, "adv_max_agents_per_attack", 1))
#         self.attack_cost = float(getattr(args, "adv_attack_cost", 0.0))

#         # PPO config
#         self.ppo_batch_size = int(getattr(args, "adv_ppo_batch_size", 40))  # episodes per update (PyMARL-style)
#         self.ppo_epochs = int(getattr(args, "adv_ppo_epochs", 4))
#         self.mini_batch_size = int(getattr(args, "adv_mini_batch_size", 512))
#         self.clip_coef = float(getattr(args, "adv_clip_coef", 0.2))
#         self.ent_coef = float(getattr(args, "adv_ent_coef", 0.05))
#         self.vf_coef = float(getattr(args, "adv_vf_coef", 0.5))
#         self.gamma = float(getattr(args, "adv_gamma", 0.99))
#         self.gae_lambda = float(getattr(args, "adv_gae_lambda", 0.95))

#         # Delta-V
#         self.use_delta_v = bool(getattr(args, "adv_use_delta_v", False))
#         self.deltaV_coef = float(getattr(args, "adv_deltaV_coef", 0.0))
#         self.deltaV_delta_b = float(getattr(args, "adv_deltaV_delta_b", 0.1))

#         self.embed_dim = int(getattr(args, "adv_embed_dim", 64))
#         lr = float(getattr(args, "lr", 3e-4))

#         self.gnn = GNNEncoder(self.obs_dim, self.embed_dim).to(self.device)
#         self.sel = VictimSelectorBudget(self.embed_dim, self.state_dim).to(self.device)
#         self.att = ActionAttackerMulti(self.obs_dim, self.embed_dim, self.n_actions).to(self.device)

#         self.opt_gnn = optim.Adam(self.gnn.parameters(), lr=lr, eps=1e-5)
#         self.opt_sel = optim.Adam(self.sel.parameters(), lr=lr, eps=1e-5)
#         self.opt_att = optim.Adam(self.att.parameters(), lr=lr, eps=1e-5)

#         # Analyst (EXACT behavior)
#         self.analyst = BudgetAnalyst(self.embed_dim, self.state_dim).to(self.device)

#         # Buffer stores both sequences
#         self.buffer = PPORolloutBuffer(max_k=max(1, self.attack_n))

#         # accumulate episodes until update (PyMARL-style)
#         self.episode_counter = 0
#         self.training_batch = {k: [] for k in [
#             "states", "obs", "avail", "v_mask", "adv_act",
#             "idx_seq", "sel_seq",
#             "budgets", "act_masks",
#             "old_lp_sel", "old_lp_att",
#             "old_v_sel", "old_v_att",
#             "advantages", "returns",
#         ]}
#         self.temp_transition = {}

#         print(f"[BudgetPPOAgent-ALIGN-FP32] attack_n={self.attack_n}, attack_cost={self.attack_cost}, device={self.device}")

#     # -------------------------------------------------
#     # Step sampling (EXACT A/B)  +  FP32 + AMP OFF
#     # -------------------------------------------------
#     def sample_action(self, obs, state, avail_actions, current_budget, max_budget, training_mode=True):
#         obs_t = to_tensor(obs, self.device)
#         state_t = to_tensor(state, self.device)
#         avail_t = to_tensor(avail_actions, self.device)
#         if obs_t.dim() == 2: obs_t = obs_t.unsqueeze(0)
#         if state_t.dim() == 1: state_t = state_t.unsqueeze(0)
#         if avail_t.dim() == 2: avail_t = avail_t.unsqueeze(0)

#         # 强制 FP32（防止外部 autocast 把它变成 fp16/bf16）
#         obs_t = obs_t.float()
#         state_t = state_t.float()
#         avail_t = avail_t.float()

#         v_mask_np = np.zeros(self.n_agents, dtype=np.float32)
#         adv_actions_np = np.full(self.n_agents, -1, dtype=np.int64)

#         K = max(1, int(self.attack_n))
#         idx_seq_np = np.full((K,), -1, dtype=np.int64)   # for A
#         sel_seq_np = np.full((K,), -1, dtype=np.int64)   # for B

#         lp_sel_total = 0.0
#         lp_att_total = 0.0
#         val_att_step = 0.0
#         step_has_attacked = False
#         num_attacked_this_step = 0

#         denom = float(max_budget) if max_budget > 0 else 1.0
#         bud_ratio = float(current_budget) / denom
#         bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=self.device)

#         act_mask_store = None

#         with torch.no_grad():
#             with amp_off(self.device):
#                 embeds = self.gnn(obs_t)                    # [1,N,E] FP32
#                 emb = embeds.squeeze(0)                     # [N,E]
#                 logits_sel_1x, val_sel = self.sel(emb.unsqueeze(0), state_t, bud_t)
#                 logits_sel_1x = logits_sel_1x.float()
#                 val_sel = val_sel.float()
#                 logits_sel = logits_sel_1x.squeeze(0)       # [N+1]

#                 if int(self.attack_n) == 1:
#                     budget_mask = build_budget_action_mask(self.n_agents, current_budget, self.device)
#                     step_has_attacked, vid, lp = sample_selector_gate_k1(logits_sel, budget_mask, eps=EPS)
#                     lp_sel_total = lp
#                     act_mask_store = budget_mask.detach().cpu().numpy()

#                     if step_has_attacked:
#                         v_mask_np[vid] = 1.0
#                         idx_seq_np[0] = vid
#                         num_attacked_this_step = 1
#                 else:
#                     exclude_mask = torch.zeros((1, self.n_agents), device=self.device, dtype=torch.float32)
#                     base_mask_for_store = None

#                     for k in range(int(self.attack_n)):
#                         forbid_noop = (k > 0) and step_has_attacked
#                         idx_t, lp_sel_k, base_mask = sample_with_budget_mask_strict(
#                             logits_sel_1x, current_budget, self.device,
#                             exclude_mask=exclude_mask,
#                             forbid_noop=forbid_noop
#                         )
#                         idx = int(idx_t.item())

#                         if base_mask_for_store is None:
#                             base_mask_for_store = base_mask.detach().clone()

#                         sel_seq_np[k] = idx

#                         if idx >= self.n_agents:
#                             break

#                         step_has_attacked = True
#                         v_mask_np[idx] = 1.0
#                         exclude_mask[0, idx] = 1.0
#                         num_attacked_this_step += 1
#                         lp_sel_total += float(lp_sel_k.item())

#                     if base_mask_for_store is None:
#                         base_mask_for_store = build_budget_action_mask(self.n_agents, current_budget, self.device)
#                     act_mask_store = base_mask_for_store.detach().cpu().numpy()

#         # attacker actions: FP32 + AMP OFF
#         if step_has_attacked and num_attacked_this_step > 0:
#             with torch.no_grad():
#                 with amp_off(self.device):
#                     obs_np = obs_t.squeeze(0).detach().cpu().numpy()
#                     avail_np = avail_t.squeeze(0).detach().cpu().numpy()
#                     emb_np = emb.detach().cpu().numpy()

#                     v_acc = 0.0
#                     for target in np.where(v_mask_np > 0.5)[0]:
#                         act_i, lp_a, v_a = self.att.act_single(
#                             obs_np[target], emb_np[target], avail_np[target], self.device
#                         )
#                         adv_actions_np[target] = act_i
#                         lp_att_total += float(lp_a.item())
#                         v_acc += float(v_a.item())
#                     val_att_step = v_acc / max(1, num_attacked_this_step)

#         if training_mode:
#             self.temp_transition = {
#                 "state": state,
#                 "obs": obs,
#                 "avail": avail_actions,
#                 "v_mask": v_mask_np,
#                 "adv_act": adv_actions_np,
#                 "idx_seq": idx_seq_np,
#                 "sel_seq": sel_seq_np,
#                 "lp_sel": float(lp_sel_total),
#                 "lp_att": float(lp_att_total),
#                 "v_sel": float(val_sel.squeeze().item()),
#                 "v_att": float(val_att_step),
#                 "bud_ratio": float(bud_ratio),
#                 "act_mask": act_mask_store,
#                 "step_has_attacked": bool(step_has_attacked),
#                 "num_attacked": int(num_attacked_this_step),
#             }

#         attacked_agents = np.where(v_mask_np > 0.5)[0].tolist()
#         return attacked_agents, adv_actions_np

#     # -------------------------------------------------
#     # Store reward
#     # -------------------------------------------------
#     def store_reward(self, reward, done, truncated=False):
#         if not self.temp_transition:
#             return
#         t = self.temp_transition

#         adv_r = -float(reward)
#         if t.get("step_has_attacked", False):
#             adv_r -= float(self.attack_cost)

#         self.buffer.add(
#             t["state"], t["obs"], t["avail"],
#             t["v_mask"], t["adv_act"],
#             t["idx_seq"], t["sel_seq"],
#             t["lp_sel"], t["lp_att"],
#             t["v_sel"], t["v_att"],
#             float(adv_r), float(done),
#             float(t["bud_ratio"]), t["act_mask"],
#         )
#         self.temp_transition = {}

#     # -------------------------------------------------
#     # End episode: bootstrap next_value  + FP32 + AMP OFF
#     # -------------------------------------------------
#     def process_episode(self, last_state, last_obs, max_budget, current_budget):
#         if len(self.buffer.rewards) == 0:
#             return {}

#         next_value = None
#         if int(self.attack_n) == 1:
#             if not bool(self.buffer.dones[-1]):
#                 with torch.no_grad():
#                     with amp_off(self.device):
#                         ns_t = to_tensor(last_state, self.device)
#                         if ns_t.dim() == 1: ns_t = ns_t.unsqueeze(0)
#                         no_t = to_tensor(last_obs, self.device)
#                         if no_t.dim() == 2: no_t = no_t.unsqueeze(0)

#                         ns_t = ns_t.float()
#                         no_t = no_t.float()

#                         denom = float(max_budget) if max_budget > 0 else 1.0
#                         bud_ratio = float(current_budget) / denom
#                         bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=self.device)

#                         emb = self.gnn(no_t).squeeze(0)  # [N,E]
#                         _, next_val_sel = self.sel(emb.unsqueeze(0), ns_t, bud_t)
#                         next_value = float(next_val_sel.squeeze().float().item())
#         else:
#             next_value = None

#         episode_data = self.buffer.get_data(self.device, gamma=self.gamma, gae_lambda=self.gae_lambda, next_value=next_value)
#         for k, v in episode_data.items():
#             if k in self.training_batch:
#                 self.training_batch[k].append(v)

#         self.buffer.clear()
#         self.episode_counter += 1

#         if self.episode_counter >= self.ppo_batch_size:
#             return self.update()
#         return {}

#     def update(self):
#         full_batch = {}
#         has_data = False
#         for k, v_list in self.training_batch.items():
#             if len(v_list) > 0:
#                 full_batch[k] = torch.cat(v_list, dim=0)
#                 has_data = True
#             else:
#                 full_batch[k] = torch.tensor([], device=self.device)

#         stats = self._ppo_optimize(full_batch) if has_data else {}

#         # analyst update (unchanged; but we keep tensors FP32)
#         if has_data and ("obs" in full_batch) and full_batch["obs"].numel() > 0:
#             with torch.no_grad():
#                 with amp_off(self.device):
#                     obs_t = full_batch["obs"].detach().float()
#                     states_t = full_batch["states"].detach().float()
#                     budgets_t = full_batch["budgets"].detach().float()
#                     returns_t = full_batch["returns"].detach().float()

#                     embeds_t = self.gnn(obs_t).detach()          # [T,N,E]
#                     global_emb_t = embeds_t.mean(dim=1)          # [T,E]

#             if int(self.attack_n) == 1:
#                 analyst_loss = self.analyst.update(global_emb_t, states_t, budgets_t, returns_t)
#             else:
#                 m = min(512, global_emb_t.shape[0])
#                 idxs = torch.randperm(global_emb_t.shape[0], device=self.device)[:m]
#                 analyst_loss = self.analyst.update(global_emb_t[idxs], states_t[idxs], budgets_t[idxs], returns_t[idxs])

#             stats = dict(stats)
#             stats["analyst_loss"] = float(analyst_loss)

#         self.clear_memory()
#         return stats

#     def clear_memory(self):
#         self.buffer.clear()
#         for k in self.training_batch:
#             self.training_batch[k] = []
#         self.episode_counter = 0
#         self.temp_transition = {}

#     # -------------------------------------------------
#     # PPO optimize  + FP32 + AMP OFF around ALL Categorical usage
#     # -------------------------------------------------
#     def _ppo_optimize(self, data):
#         states, obs, avail = data["states"], data["obs"], data["avail"]
#         v_mask = data["v_mask"]
#         idx_seq = data["idx_seq"]
#         sel_seq = data["sel_seq"]
#         adv_act = data["adv_act"]
#         budgets, act_masks = data["budgets"], data["act_masks"]
#         old_lp_sel, old_lp_att = data["old_lp_sel"], data["old_lp_att"]
#         returns, advantages = data["returns"], data["advantages"]

#         batch_size = states.shape[0]
#         if batch_size < self.mini_batch_size:
#             return {}

#         b_inds = np.arange(batch_size)
#         loss_pi_list, loss_v_list, loss_ent_list, loss_meta_list = [], [], [], []

#         for _ in range(self.ppo_epochs):
#             np.random.shuffle(b_inds)
#             for start in range(0, batch_size, self.mini_batch_size):
#                 end = start + self.mini_batch_size
#                 mb_inds = b_inds[start:end]

#                 mb_states = states[mb_inds].float()
#                 mb_obs = obs[mb_inds].float()
#                 mb_avail = avail[mb_inds].float()
#                 mb_vmask = v_mask[mb_inds].float()
#                 mb_idx_seq = idx_seq[mb_inds]
#                 mb_sel_seq = sel_seq[mb_inds]
#                 mb_advact = adv_act[mb_inds]
#                 mb_budgets = budgets[mb_inds].float()
#                 mb_actmask = act_masks[mb_inds].float()
#                 mb_old_lp_sel = old_lp_sel[mb_inds].float()
#                 mb_old_lp_att = old_lp_att[mb_inds].float()
#                 mb_returns = returns[mb_inds].float()
#                 mb_adv = advantages[mb_inds].float()
#                 mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

#                 with amp_off(self.device):
#                     mb_embeds = self.gnn(mb_obs)  # [B,N,E]
#                     logits_sel, val_sel = self.sel(mb_embeds, mb_states, mb_budgets)  # [B,N+1],[B]
#                     logits_sel = logits_sel.float()
#                     val_sel = val_sel.float()
#                     attacked = (mb_vmask.sum(dim=1) > 0.5)

#                     if int(self.attack_n) == 1:
#                         masked_logits = logits_sel.clone()
#                         masked_logits[mb_actmask == 0] = -1e10
#                         new_lp_sel, entropy_sel = recompute_logp_selector_gate_k1(
#                             masked_logits, mb_idx_seq, attacked, eps=EPS
#                         )
#                         new_lp_sel = new_lp_sel.float()
#                         entropy_sel = entropy_sel.float()
#                     else:
#                         masked_logits_ent = logits_sel.clone()
#                         masked_logits_ent[mb_actmask == 0] = -1e10
#                         dist_sel_ent = Categorical(logits=masked_logits_ent.float())
#                         entropy_sel = dist_sel_ent.entropy().mean()

#                         new_lp_sel = compute_new_lp_sel_autoregressive(
#                             logits_sel=logits_sel,
#                             sel_seq=mb_sel_seq,
#                             act_masks=mb_actmask
#                         ).float()

#                     ratio_sel = torch.exp(new_lp_sel - mb_old_lp_sel)
#                     surr1 = ratio_sel * mb_adv
#                     surr2 = torch.clamp(ratio_sel, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
#                     loss_sel_pi = -torch.min(surr1, surr2).mean()

#                     loss_sel_v = 0.5 * ((val_sel - mb_returns) ** 2).mean()
#                     loss_sel = loss_sel_pi + self.vf_coef * loss_sel_v - self.ent_coef * entropy_sel

#                     loss_meta = torch.tensor(0.0, device=self.device)
#                     if self.use_delta_v and self.deltaV_coef > 0.0:
#                         attack_indicator = attacked.float()
#                         b_curr = mb_budgets
#                         b_prev = torch.clamp(b_curr - self.deltaV_delta_b, 0.0, 1.0)
#                         _, v_prev = self.sel(mb_embeds, mb_states, b_prev)
#                         delta_v = torch.relu(val_sel - v_prev.float())
#                         loss_meta = (delta_v * attack_indicator).mean()
#                         loss_sel = loss_sel + self.deltaV_coef * loss_meta
#                         loss_meta_list.append(float(loss_meta.item()))

#                     loss_att = torch.tensor(0.0, device=self.device)
#                     attack_bool = (mb_vmask > 0.5) & (mb_advact >= 0)

#                     if attack_bool.sum() > 0:
#                         logits_att, val_att_all = self.att(mb_obs, mb_embeds, mb_avail)  # [B,N,A], [B,N]
#                         logits_att = logits_att.float()
#                         val_att_all = val_att_all.float()

#                         mask_sum = mb_vmask.sum(dim=1, keepdim=True).clamp(min=1.0)
#                         val_att = (val_att_all * mb_vmask).sum(dim=1) / mask_sum.squeeze(-1)

#                         dist_att = Categorical(logits=logits_att)
#                         all_lp_att = dist_att.log_prob(torch.clamp(mb_advact, min=0))          # [B,N]
#                         new_lp_att_step = (all_lp_att * mb_vmask).sum(dim=1)                   # [B]

#                         row_is_attack = attack_bool.any(dim=1)
#                         active_mb_idx = torch.where(row_is_attack)[0]

#                         if len(active_mb_idx) > 0:
#                             ratio_att = torch.exp(new_lp_att_step[active_mb_idx] - mb_old_lp_att[active_mb_idx])
#                             surr1_a = ratio_att * mb_adv[active_mb_idx]
#                             surr2_a = torch.clamp(ratio_att, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv[active_mb_idx]
#                             loss_att_pi = -torch.min(surr1_a, surr2_a).mean()

#                             loss_att_v = 0.5 * ((val_att[active_mb_idx] - mb_returns[active_mb_idx]) ** 2).mean()

#                             ent_att_all = dist_att.entropy()  # [B,N]
#                             loss_att_ent = (ent_att_all * mb_vmask).sum() / (attack_bool.sum() + 1e-8)

#                             loss_att = loss_att_pi + self.vf_coef * loss_att_v - self.ent_coef * loss_att_ent

#                     total_loss = loss_sel + loss_att

#                 self.opt_gnn.zero_grad()
#                 self.opt_sel.zero_grad()
#                 self.opt_att.zero_grad()
#                 total_loss.backward()

#                 torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 0.5)
#                 torch.nn.utils.clip_grad_norm_(self.sel.parameters(), 0.5)
#                 torch.nn.utils.clip_grad_norm_(self.att.parameters(), 0.5)

#                 self.opt_gnn.step()
#                 self.opt_sel.step()
#                 self.opt_att.step()

#                 loss_pi_list.append(float(loss_sel_pi.item()))
#                 loss_v_list.append(float(loss_sel_v.item()))
#                 loss_ent_list.append(float(entropy_sel.item()))

#         return {
#             "policy_loss": float(np.mean(loss_pi_list)) if loss_pi_list else 0.0,
#             "value_loss": float(np.mean(loss_v_list)) if loss_v_list else 0.0,
#             "entropy": float(np.mean(loss_ent_list)) if loss_ent_list else 0.0,
#             "meta_loss": float(np.mean(loss_meta_list)) if loss_meta_list else 0.0,
#         }

#     # -------------------------------------------------
#     # IO
#     # -------------------------------------------------
#     def load(self, checkpoint_path):
#         ckpt = torch.load(checkpoint_path, map_location=self.device)
#         self.gnn.load_state_dict(ckpt["models"]["gnn"])
#         self.sel.load_state_dict(ckpt["models"]["sel"])
#         self.att.load_state_dict(ckpt["models"]["att"])
#         if "analyst" in ckpt.get("models", {}):
#             self.analyst.load_state_dict(ckpt["models"]["analyst"])
#         print(f"[BudgetPPOAgent] Loaded checkpoint from {checkpoint_path}")

#     def has_pending_data(self):
#         if getattr(self, "temp_transition", None):
#             return True
#         if getattr(self, "buffer", None) is not None and len(self.buffer.rewards) > 0:
#             return True
#         if getattr(self, "episode_counter", 0) > 0:
#             return True
#         tb = getattr(self, "training_batch", None)
#         if tb is not None:
#             for k, v in tb.items():
#                 if isinstance(v, list) and len(v) > 0:
#                     return True
#         return False


# -*- coding: utf-8 -*-
# Drop-in replacement for: src/modules/agents/budget_ppo_agent.py
# EXACT training logic alignment with your top script:
#   - attack_n==1 : CODE-A (gate+k1, strict logp recompute, bootstrap next_value)
#   - attack_n>1  : CODE-B (FORCE_FILL sequential, sel_seq, autoreg logp, next_value=None)
#   - Analyst update RNG alignment: A full update (no randperm), B randperm subset
#
# Patch in this version:
#   - Strategy B Implemented: Dynamic Mini-Batch Size (Total / 4).
#   - CRITICAL FIX: "Drop Last" logic added to discard tiny tail batches (prevents NaN/CUDA asserts).
#   - FORCE selector/attacker sampling + PPO Categorical to run in FP32 and with AMP disabled.

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from contextlib import nullcontext

EPS = 1e-8


def amp_off(device):
    # Disable autocast locally (important if outer learner enables AMP)
    if str(device).startswith("cuda"):
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_budget_action_mask(n_agents: int, current_budget: int, device):
    """[N+1] budget-only mask: budget<=0 => only noop allowed"""
    mask = torch.ones(n_agents + 1, dtype=torch.float32, device=device)
    if current_budget <= 0:
        mask[:-1] = 0.0
    return mask


# =========================
# Networks (same as your script)
# =========================
class GNNEncoder(nn.Module):
    def __init__(self, obs_dim, embed_dim):
        super().__init__()
        self.self_mlp = layer_init(nn.Linear(obs_dim, embed_dim))
        self.nei_mlp = layer_init(nn.Linear(obs_dim, embed_dim))

    def forward(self, obs_all):
        # obs_all: [B,N,obs]
        x_self = self.self_mlp(obs_all)
        mean_nei = obs_all.mean(dim=1, keepdim=True)
        msg = self.nei_mlp(mean_nei)
        h = torch.relu(x_self + msg)
        return h


class VictimSelectorBudget(nn.Module):
    def __init__(self, embed_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.global_feature_net = nn.Sequential(
            layer_init(nn.Linear(embed_dim + state_dim + 1, hidden_dim)),
            nn.ReLU(),
        )
        self.agent_policy_net = nn.Sequential(
            layer_init(nn.Linear(embed_dim + hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )
        self.noop_policy_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )
        self.v_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, agent_embeds, state_batch, budget_ratio):
        # agent_embeds:[B,N,E], state_batch:[B,S], budget_ratio:[B,1] or [B]
        B, N, E = agent_embeds.shape
        global_embed = agent_embeds.mean(dim=1)
        if budget_ratio.dim() == 1:
            budget_ratio = budget_ratio.unsqueeze(1)

        global_input = torch.cat([global_embed, state_batch, budget_ratio], dim=-1)
        global_feat = self.global_feature_net(global_input)

        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        agent_input = torch.cat([agent_embeds, global_feat_expanded], dim=-1)

        agent_logits = self.agent_policy_net(agent_input).squeeze(-1)  # [B,N]
        noop_logit = self.noop_policy_net(global_feat)                 # [B,1]
        total_logits = torch.cat([agent_logits, noop_logit], dim=-1)   # [B,N+1]

        v_value = self.v_net(global_feat).squeeze(-1)                  # [B]
        return total_logits, v_value


class ActionAttackerMulti(nn.Module):
    def __init__(self, obs_dim, embed_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(obs_dim + embed_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, n_actions), std=0.01),
        )
        self.value_mlp = nn.Sequential(
            layer_init(nn.Linear(obs_dim + embed_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs_all, agent_embeds, avail_actions_all=None):
        # obs_all:[B,N,obs], agent_embeds:[B,N,E], avail_actions_all:[B,N,A]
        x = torch.cat([obs_all, agent_embeds], dim=-1)
        logits = self.policy_mlp(x)  # [B,N,A]
        if avail_actions_all is not None:
            logits = logits.masked_fill(avail_actions_all == 0, -1e10)
        values = self.value_mlp(x).squeeze(-1)  # [B,N]
        return logits, values

    def act_single(self, obs_i, embed_i, avail_actions_i, device):
        obs_t = torch.as_tensor(obs_i, dtype=torch.float32, device=device).view(1, 1, -1)
        emb_t = torch.as_tensor(embed_i, dtype=torch.float32, device=device).view(1, 1, -1)
        avail_t = torch.as_tensor(avail_actions_i, dtype=torch.float32, device=device).view(1, 1, -1)

        # AMP OFF + FP32 logits before Categorical
        with amp_off(device):
            logits, values = self.forward(obs_t, emb_t, avail_t)  # logits [1,1,A]
            logits = logits.float()
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.detach(), values.detach()


class BudgetAnalyst(nn.Module):
    def __init__(self, embed_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(embed_dim + state_dim + 1, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, global_embed, state, budget_ratio):
        x = torch.cat([global_embed, state, budget_ratio], dim=-1)
        return self.net(x)

    def update(self, global_embed, state, budget_ratio, returns):
        pred = self.forward(global_embed, state, budget_ratio).squeeze(-1)
        loss = self.loss_fn(pred, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


# =========================
# CODE-A selector sampling (attack_n==1)
# =========================
def sample_selector_gate_k1(logits_sel_1x, budget_mask_1d, eps=EPS):
    masked_logits = logits_sel_1x.clone()
    masked_logits[budget_mask_1d == 0] = -1e10

    probs_full = F.softmax(masked_logits, dim=-1)
    p_noop = probs_full[-1].clamp(min=eps, max=1.0)
    p_attack = (1.0 - p_noop).clamp(min=eps, max=1.0)

    if budget_mask_1d[:-1].sum().item() < 0.5:
        return False, -1, float(torch.log(p_noop + eps).item())

    if random.random() > float(p_attack.item()):
        return False, -1, float(torch.log(p_noop + eps).item())

    agent_probs = probs_full[:-1].clamp(min=eps)
    agent_probs = agent_probs / agent_probs.sum()

    vid = int(torch.multinomial(agent_probs, num_samples=1, replacement=False).item())
    lp = float(torch.log(p_attack + eps).item() + torch.log(agent_probs[vid] + eps).item())
    return True, vid, lp


def recompute_logp_selector_gate_k1(masked_logits_bn1, idx_seq_bk, attacked_bool, eps=EPS):
    probs_full = F.softmax(masked_logits_bn1, dim=-1)
    p_noop = probs_full[:, -1].clamp(min=eps, max=1.0)
    p_attack = (1.0 - p_noop).clamp(min=eps, max=1.0)

    agent_logits = masked_logits_bn1[:, :-1]
    agent_logp = F.log_softmax(agent_logits, dim=-1)

    logp = torch.log(p_noop + eps)
    if attacked_bool.any():
        vid = idx_seq_bk[:, 0].clamp(min=0)
        lp_attack = torch.log(p_attack + eps) + agent_logp.gather(1, vid.view(-1, 1)).squeeze(1)
        logp = torch.where(attacked_bool, lp_attack, logp)

    gate_ent = -(p_noop * torch.log(p_noop) + p_attack * torch.log(p_attack)).mean()
    agent_probs = F.softmax(agent_logits, dim=-1)
    agent_ent = -(agent_probs * agent_logp).sum(dim=1).mean()
    entropy = gate_ent + agent_ent
    return logp, entropy


# =========================
# CODE-B selector sampling (attack_n>1): FORCE_FILL
# =========================
def sample_with_budget_mask_strict(logits_1x, current_budget, device, exclude_mask=None, forbid_noop=False):
    logits = logits_1x.squeeze(0)  # [N+1]
    base_mask = torch.ones_like(logits, device=device)
    if current_budget <= 0:
        base_mask[:-1] = 0.0

    combined_mask = base_mask.clone()
    if exclude_mask is not None:
        combined_mask[:-1] *= (1.0 - exclude_mask.squeeze(0))
    if forbid_noop:
        combined_mask[-1] = 0.0

    # ===== 兜底：如果一个都不能选，就强制返回 noop（或解除 forbid_noop）=====
    if combined_mask.sum().item() < 0.5:
        noop_idx = logits.numel() - 1
        action_idx = torch.tensor(noop_idx, device=device, dtype=torch.long)
        log_prob = torch.tensor(0.0, device=device, dtype=torch.float32)
        return action_idx, log_prob, base_mask

    # ===== fp16-safe：别用 -1e10，改成 dtype 安全的负大数 =====
    NEG = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e10
    masked_logits = logits.clone()
    masked_logits[combined_mask == 0] = NEG

    dist = Categorical(logits=masked_logits)
    action_idx = dist.sample()
    log_prob = dist.log_prob(action_idx)
    return action_idx, log_prob, base_mask


def compute_new_lp_sel_autoregressive(logits_sel, sel_seq, act_masks):
    device = logits_sel.device
    B, NA1 = logits_sel.shape
    N = NA1 - 1
    K = sel_seq.shape[1]

    new_lp = torch.zeros(B, device=device)
    exclude = torch.zeros(B, N, device=device)
    attacked_already = torch.zeros(B, dtype=torch.bool, device=device)

    for k in range(K):
        idx = sel_seq[:, k]
        valid = (idx >= 0)
        if valid.sum() == 0:
            continue

        masked_logits_k = logits_sel.clone()
        masked_logits_k[act_masks == 0] = -1e10
        masked_logits_k[:, :N] = masked_logits_k[:, :N].masked_fill(exclude > 0.5, -1e10)

        forbid = attacked_already
        if forbid.any():
            masked_logits_k[forbid, -1] = -1e10

        logp_all = F.log_softmax(masked_logits_k, dim=-1)
        new_lp[valid] += logp_all[valid, idx[valid]]

        is_victim = valid & (idx < N)
        attacked_already = attacked_already | is_victim
        if is_victim.any():
            exclude[is_victim, idx[is_victim]] = 1.0

    return new_lp


# =========================
# Buffer: stores BOTH idx_seq (A) and sel_seq (B)
# =========================
class PPORolloutBuffer:
    def __init__(self, max_k: int):
        self.max_k = int(max_k)
        self.clear()

    def clear(self):
        self.states, self.obs_all, self.avail_all = [], [], []
        self.victim_mask, self.adv_actions = [], []
        self.idx_seq, self.sel_seq = [], []          # [K] padded -1
        self.logprob_sel, self.logprob_att = [], []
        self.value_sel, self.value_att = [], []
        self.rewards, self.dones = [], []
        self.budgets, self.action_masks = [], []     # [N+1]

    def add(self, state, obs, avail, v_mask, adv_act,
            idx_seq, sel_seq,
            lp_sel, lp_att, v_sel, v_att, r, d, bud, act_mask):
        self.states.append(state)
        self.obs_all.append(obs)
        self.avail_all.append(avail)
        self.victim_mask.append(v_mask)
        self.adv_actions.append(adv_act)

        self.idx_seq.append(idx_seq)
        self.sel_seq.append(sel_seq)

        self.logprob_sel.append(lp_sel)
        self.logprob_att.append(lp_att)
        self.value_sel.append(v_sel)
        self.value_att.append(v_att)
        self.rewards.append(r)
        self.dones.append(d)
        self.budgets.append(bud)
        self.action_masks.append(act_mask)

    def get_data(self, device, gamma=0.99, gae_lambda=0.95, next_value=None):
        states = to_tensor(np.array(self.states), device)                     # [T,S]
        obs = to_tensor(np.array(self.obs_all), device)                       # [T,N,obs]
        avail = to_tensor(np.array(self.avail_all), device)                   # [T,N,A]
        v_mask = to_tensor(np.array(self.victim_mask), device)                # [T,N]
        adv_act = torch.as_tensor(np.array(self.adv_actions), dtype=torch.long, device=device)  # [T,N]

        idx_seq = torch.as_tensor(np.array(self.idx_seq), dtype=torch.long, device=device)      # [T,K]
        sel_seq = torch.as_tensor(np.array(self.sel_seq), dtype=torch.long, device=device)      # [T,K]

        rewards = to_tensor(np.array(self.rewards), device)                   # [T]
        dones = to_tensor(np.array(self.dones), device)                       # [T]
        budgets = to_tensor(np.array(self.budgets), device).unsqueeze(1)      # [T,1]
        act_masks = to_tensor(np.array(self.action_masks), device)            # [T,N+1]

        old_lp_sel = to_tensor(np.array(self.logprob_sel), device)            # [T]
        old_lp_att = to_tensor(np.array(self.logprob_att), device)            # [T]
        old_v_sel = to_tensor(np.array(self.value_sel), device)               # [T]
        old_v_att = to_tensor(np.array(self.value_att), device)               # [T]

        # GAE based on selector value (EXACT)
        with torch.no_grad():
            values = old_v_sel
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            T = len(rewards)
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value if next_value is not None else 0.0
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        return {
            "states": states, "obs": obs, "avail": avail,
            "v_mask": v_mask, "adv_act": adv_act,
            "idx_seq": idx_seq, "sel_seq": sel_seq,
            "budgets": budgets, "act_masks": act_masks,
            "old_lp_sel": old_lp_sel, "old_lp_att": old_lp_att,
            "old_v_sel": old_v_sel, "old_v_att": old_v_att,
            "advantages": advantages, "returns": returns,
        }


# =========================
# Main Agent class
# =========================
class BudgetPPOAgent(nn.Module):
    """
    PyMARL adversarial agent, training logic aligned to your top script.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "use_cuda", False) and torch.cuda.is_available() else "cpu")

        self.obs_dim = int(args.obs_shape)
        self.state_dim = int(args.state_shape)
        self.n_actions = int(args.n_actions)
        self.n_agents = int(args.n_agents)

        self.attack_n = int(getattr(args, "adv_max_agents_per_attack", 1))
        self.attack_cost = float(getattr(args, "adv_attack_cost", 0.0))

        # PPO config
        self.ppo_batch_size = int(getattr(args, "adv_ppo_batch_size", 40))  # episodes per update (PyMARL-style)
        self.ppo_epochs = int(getattr(args, "adv_ppo_epochs", 4))
        self.mini_batch_size = int(getattr(args, "adv_mini_batch_size", 512)) # Will be used as a "floor" or ignored in Strategy B logic
        self.clip_coef = float(getattr(args, "adv_clip_coef", 0.2))
        self.ent_coef = float(getattr(args, "adv_ent_coef", 0.05))
        self.vf_coef = float(getattr(args, "adv_vf_coef", 0.5))
        self.gamma = float(getattr(args, "adv_gamma", 0.99))
        self.gae_lambda = float(getattr(args, "adv_gae_lambda", 0.95))

        # Delta-V
        self.use_delta_v = bool(getattr(args, "adv_use_delta_v", False))
        self.deltaV_coef = float(getattr(args, "adv_deltaV_coef", 0.0))
        self.deltaV_delta_b = float(getattr(args, "adv_deltaV_delta_b", 0.0))

        self.embed_dim = int(getattr(args, "adv_embed_dim", 64))
        lr = 3e-4

        self.gnn = GNNEncoder(self.obs_dim, self.embed_dim).to(self.device)
        self.sel = VictimSelectorBudget(self.embed_dim, self.state_dim).to(self.device)
        self.att = ActionAttackerMulti(self.obs_dim, self.embed_dim, self.n_actions).to(self.device)

        self.opt_gnn = optim.Adam(self.gnn.parameters(), lr=lr, eps=1e-5)
        self.opt_sel = optim.Adam(self.sel.parameters(), lr=lr, eps=1e-5)
        self.opt_att = optim.Adam(self.att.parameters(), lr=lr, eps=1e-5)

        # Analyst (EXACT behavior)
        self.analyst = BudgetAnalyst(self.embed_dim, self.state_dim).to(self.device)

        # Buffer stores both sequences
        self.buffer = PPORolloutBuffer(max_k=max(1, self.attack_n))

        # accumulate episodes until update (PyMARL-style)
        self.episode_counter = 0
        self.training_batch = {k: [] for k in [
            "states", "obs", "avail", "v_mask", "adv_act",
            "idx_seq", "sel_seq",
            "budgets", "act_masks",
            "old_lp_sel", "old_lp_att",
            "old_v_sel", "old_v_att",
            "advantages", "returns",
        ]}
        self.temp_transition = {}

        print(f"[BudgetPPOAgent-ALIGN-FP32] attack_n={self.attack_n}, attack_cost={self.attack_cost}, device={self.device}")
        print(f"[BudgetPPOAgent-StrategyB-FIXED] Dynamic mini-batch + Drop Last (<16) to prevent NaN.")

    # -------------------------------------------------
    # Step sampling (EXACT A/B)  +  FP32 + AMP OFF
    # -------------------------------------------------
    def sample_action(self, obs, state, avail_actions, current_budget, max_budget, training_mode=True):
        obs_t = to_tensor(obs, self.device)
        state_t = to_tensor(state, self.device)
        avail_t = to_tensor(avail_actions, self.device)
        if obs_t.dim() == 2: obs_t = obs_t.unsqueeze(0)
        if state_t.dim() == 1: state_t = state_t.unsqueeze(0)
        if avail_t.dim() == 2: avail_t = avail_t.unsqueeze(0)

        # 强制 FP32（防止外部 autocast 把它变成 fp16/bf16）
        obs_t = obs_t.float()
        state_t = state_t.float()
        avail_t = avail_t.float()

        v_mask_np = np.zeros(self.n_agents, dtype=np.float32)
        adv_actions_np = np.full(self.n_agents, -1, dtype=np.int64)

        K = max(1, int(self.attack_n))
        idx_seq_np = np.full((K,), -1, dtype=np.int64)   # for A
        sel_seq_np = np.full((K,), -1, dtype=np.int64)   # for B

        lp_sel_total = 0.0
        lp_att_total = 0.0
        val_att_step = 0.0
        step_has_attacked = False
        num_attacked_this_step = 0

        denom = float(max_budget) if max_budget > 0 else 1.0
        bud_ratio = float(current_budget) / denom
        bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=self.device)

        act_mask_store = None

        with torch.no_grad():
            with amp_off(self.device):
                embeds = self.gnn(obs_t)                    # [1,N,E] FP32
                emb = embeds.squeeze(0)                     # [N,E]
                logits_sel_1x, val_sel = self.sel(emb.unsqueeze(0), state_t, bud_t)
                logits_sel_1x = logits_sel_1x.float()
                val_sel = val_sel.float()
                logits_sel = logits_sel_1x.squeeze(0)       # [N+1]

                if int(self.attack_n) == 1:
                    budget_mask = build_budget_action_mask(self.n_agents, current_budget, self.device)
                    step_has_attacked, vid, lp = sample_selector_gate_k1(logits_sel, budget_mask, eps=EPS)
                    lp_sel_total = lp
                    act_mask_store = budget_mask.detach().cpu().numpy()

                    if step_has_attacked:
                        v_mask_np[vid] = 1.0
                        idx_seq_np[0] = vid
                        num_attacked_this_step = 1
                else:
                    exclude_mask = torch.zeros((1, self.n_agents), device=self.device, dtype=torch.float32)
                    base_mask_for_store = None

                    for k in range(int(self.attack_n)):
                        forbid_noop = (k > 0) and step_has_attacked
                        idx_t, lp_sel_k, base_mask = sample_with_budget_mask_strict(
                            logits_sel_1x, current_budget, self.device,
                            exclude_mask=exclude_mask,
                            forbid_noop=forbid_noop
                        )
                        idx = int(idx_t.item())

                        if base_mask_for_store is None:
                            base_mask_for_store = base_mask.detach().clone()

                        sel_seq_np[k] = idx

                        if idx >= self.n_agents:
                            break

                        step_has_attacked = True
                        v_mask_np[idx] = 1.0
                        exclude_mask[0, idx] = 1.0
                        num_attacked_this_step += 1
                        lp_sel_total += float(lp_sel_k.item())

                    if base_mask_for_store is None:
                        base_mask_for_store = build_budget_action_mask(self.n_agents, current_budget, self.device)
                    act_mask_store = base_mask_for_store.detach().cpu().numpy()

        # attacker actions: FP32 + AMP OFF
        if step_has_attacked and num_attacked_this_step > 0:
            with torch.no_grad():
                with amp_off(self.device):
                    obs_np = obs_t.squeeze(0).detach().cpu().numpy()
                    avail_np = avail_t.squeeze(0).detach().cpu().numpy()
                    emb_np = emb.detach().cpu().numpy()

                    v_acc = 0.0
                    for target in np.where(v_mask_np > 0.5)[0]:
                        act_i, lp_a, v_a = self.att.act_single(
                            obs_np[target], emb_np[target], avail_np[target], self.device
                        )
                        adv_actions_np[target] = act_i
                        lp_att_total += float(lp_a.item())
                        v_acc += float(v_a.item())
                    val_att_step = v_acc / max(1, num_attacked_this_step)

        if training_mode:
            self.temp_transition = {
                "state": state,
                "obs": obs,
                "avail": avail_actions,
                "v_mask": v_mask_np,
                "adv_act": adv_actions_np,
                "idx_seq": idx_seq_np,
                "sel_seq": sel_seq_np,
                "lp_sel": float(lp_sel_total),
                "lp_att": float(lp_att_total),
                "v_sel": float(val_sel.squeeze().item()),
                "v_att": float(val_att_step),
                "bud_ratio": float(bud_ratio),
                "act_mask": act_mask_store,
                "step_has_attacked": bool(step_has_attacked),
                "num_attacked": int(num_attacked_this_step),
            }

        attacked_agents = np.where(v_mask_np > 0.5)[0].tolist()
        return attacked_agents, adv_actions_np

    # -------------------------------------------------
    # Store reward
    # -------------------------------------------------
    def store_reward(self, reward, done, truncated=False):
        if not self.temp_transition:
            return
        t = self.temp_transition

        adv_r = -float(reward)
        if t.get("step_has_attacked", False):
            adv_r -= float(self.attack_cost)

        self.buffer.add(
            t["state"], t["obs"], t["avail"],
            t["v_mask"], t["adv_act"],
            t["idx_seq"], t["sel_seq"],
            t["lp_sel"], t["lp_att"],
            t["v_sel"], t["v_att"],
            float(adv_r), float(done),
            float(t["bud_ratio"]), t["act_mask"],
        )
        self.temp_transition = {}

    # -------------------------------------------------
    # End episode: bootstrap next_value  + FP32 + AMP OFF
    # -------------------------------------------------
    def process_episode(self, last_state, last_obs, max_budget, current_budget):
        if len(self.buffer.rewards) == 0:
            return {}

        next_value = None
        if int(self.attack_n) == 1:
            if not bool(self.buffer.dones[-1]):
                with torch.no_grad():
                    with amp_off(self.device):
                        ns_t = to_tensor(last_state, self.device)
                        if ns_t.dim() == 1: ns_t = ns_t.unsqueeze(0)
                        no_t = to_tensor(last_obs, self.device)
                        if no_t.dim() == 2: no_t = no_t.unsqueeze(0)

                        ns_t = ns_t.float()
                        no_t = no_t.float()

                        denom = float(max_budget) if max_budget > 0 else 1.0
                        bud_ratio = float(current_budget) / denom
                        bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=self.device)

                        emb = self.gnn(no_t).squeeze(0)  # [N,E]
                        _, next_val_sel = self.sel(emb.unsqueeze(0), ns_t, bud_t)
                        next_value = float(next_val_sel.squeeze().float().item())
        else:
            next_value = None

        episode_data = self.buffer.get_data(self.device, gamma=self.gamma, gae_lambda=self.gae_lambda, next_value=next_value)
        for k, v in episode_data.items():
            if k in self.training_batch:
                self.training_batch[k].append(v)

        self.buffer.clear()
        self.episode_counter += 1

        if self.episode_counter >= self.ppo_batch_size:
            return self.update()
        return {}

    def update(self):
        full_batch = {}
        has_data = False
        for k, v_list in self.training_batch.items():
            if len(v_list) > 0:
                full_batch[k] = torch.cat(v_list, dim=0)
                has_data = True
            else:
                full_batch[k] = torch.tensor([], device=self.device)

        stats = self._ppo_optimize(full_batch) if has_data else {}

        # analyst update (unchanged; but we keep tensors FP32)
        if has_data and ("obs" in full_batch) and full_batch["obs"].numel() > 0:
            with torch.no_grad():
                with amp_off(self.device):
                    obs_t = full_batch["obs"].detach().float()
                    states_t = full_batch["states"].detach().float()
                    budgets_t = full_batch["budgets"].detach().float()
                    returns_t = full_batch["returns"].detach().float()

                    embeds_t = self.gnn(obs_t).detach()          # [T,N,E]
                    global_emb_t = embeds_t.mean(dim=1)          # [T,E]

            if int(self.attack_n) == 1:
                analyst_loss = self.analyst.update(global_emb_t, states_t, budgets_t, returns_t)
            else:
                m = min(512, global_emb_t.shape[0])
                idxs = torch.randperm(global_emb_t.shape[0], device=self.device)[:m]
                analyst_loss = self.analyst.update(global_emb_t[idxs], states_t[idxs], budgets_t[idxs], returns_t[idxs])

            stats = dict(stats)
            stats["analyst_loss"] = float(analyst_loss)

        self.clear_memory()
        return stats

    def clear_memory(self):
        self.buffer.clear()
        for k in self.training_batch:
            self.training_batch[k] = []
        self.episode_counter = 0
        self.temp_transition = {}

    # -------------------------------------------------
    # PPO optimize  + FP32 + AMP OFF around ALL Categorical usage
    # -------------------------------------------------
    def _ppo_optimize(self, data):
        states, obs, avail = data["states"], data["obs"], data["avail"]
        v_mask = data["v_mask"]
        idx_seq = data["idx_seq"]
        sel_seq = data["sel_seq"]
        adv_act = data["adv_act"]
        budgets, act_masks = data["budgets"], data["act_masks"]
        old_lp_sel, old_lp_att = data["old_lp_sel"], data["old_lp_att"]
        returns, advantages = data["returns"], data["advantages"]

        batch_size = states.shape[0]

        # [Strategy B] Dynamic Batch Size Calculation
        # Target: Split the collected buffer into `n_minibatches` chunks.
        n_minibatches = 4
        dynamic_mb_size = batch_size // n_minibatches

        # Safety: If buffer is too small, use full batch
        if dynamic_mb_size < 64:
            dynamic_mb_size = batch_size

        # If total data is extremely small (e.g., < 16 steps), PPO is unstable, skip
        if batch_size < 256:
            return {}

        b_inds = np.arange(batch_size)
        loss_pi_list, loss_v_list, loss_ent_list, loss_meta_list = [], [], [], []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, dynamic_mb_size):
                end = start + dynamic_mb_size
                mb_inds = b_inds[start:end]

                # ===============================================
                # 【关键修复】Drop Last / 丢弃尾部残次品
                # 彻底防止 batch_size=1 导致的 NaN / CUDA Assert
                # ===============================================
                if len(mb_inds) < 16:
                    continue 

                mb_states = states[mb_inds].float()
                mb_obs = obs[mb_inds].float()
                mb_avail = avail[mb_inds].float()
                mb_vmask = v_mask[mb_inds].float()
                mb_idx_seq = idx_seq[mb_inds]
                mb_sel_seq = sel_seq[mb_inds]
                mb_advact = adv_act[mb_inds]
                mb_budgets = budgets[mb_inds].float()
                mb_actmask = act_masks[mb_inds].float()
                mb_old_lp_sel = old_lp_sel[mb_inds].float()
                mb_old_lp_att = old_lp_att[mb_inds].float()
                mb_returns = returns[mb_inds].float()
                mb_adv = advantages[mb_inds].float()
                
                # Double safety: Normalize advantage only if batch size > 1
                if mb_adv.shape[0] > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with amp_off(self.device):
                    mb_embeds = self.gnn(mb_obs)  # [B,N,E]
                    logits_sel, val_sel = self.sel(mb_embeds, mb_states, mb_budgets)  # [B,N+1],[B]
                    logits_sel = logits_sel.float()
                    val_sel = val_sel.float()
                    attacked = (mb_vmask.sum(dim=1) > 0.5)

                    if int(self.attack_n) == 1:
                        masked_logits = logits_sel.clone()
                        masked_logits[mb_actmask == 0] = -1e10
                        new_lp_sel, entropy_sel = recompute_logp_selector_gate_k1(
                            masked_logits, mb_idx_seq, attacked, eps=EPS
                        )
                        new_lp_sel = new_lp_sel.float()
                        entropy_sel = entropy_sel.float()
                    else:
                        masked_logits_ent = logits_sel.clone()
                        masked_logits_ent[mb_actmask == 0] = -1e10
                        dist_sel_ent = Categorical(logits=masked_logits_ent.float())
                        entropy_sel = dist_sel_ent.entropy().mean()

                        new_lp_sel = compute_new_lp_sel_autoregressive(
                            logits_sel=logits_sel,
                            sel_seq=mb_sel_seq,
                            act_masks=mb_actmask
                        ).float()

                    ratio_sel = torch.exp(new_lp_sel - mb_old_lp_sel)
                    surr1 = ratio_sel * mb_adv
                    surr2 = torch.clamp(ratio_sel, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                    loss_sel_pi = -torch.min(surr1, surr2).mean()

                    loss_sel_v = 0.5 * ((val_sel - mb_returns) ** 2).mean()
                    loss_sel = loss_sel_pi + self.vf_coef * loss_sel_v - self.ent_coef * entropy_sel

                    loss_meta = torch.tensor(0.0, device=self.device)
                    if self.use_delta_v and self.deltaV_coef > 0.0:
                        attack_indicator = attacked.float()
                        b_curr = mb_budgets
                        b_prev = torch.clamp(b_curr - self.deltaV_delta_b, 0.0, 1.0)
                        _, v_prev = self.sel(mb_embeds, mb_states, b_prev)
                        delta_v = torch.relu(val_sel - v_prev.float())
                        loss_meta = (delta_v * attack_indicator).mean()
                        loss_sel = loss_sel + self.deltaV_coef * loss_meta
                        loss_meta_list.append(float(loss_meta.item()))

                    loss_att = torch.tensor(0.0, device=self.device)
                    attack_bool = (mb_vmask > 0.5) & (mb_advact >= 0)

                    if attack_bool.sum() > 0:
                        logits_att, val_att_all = self.att(mb_obs, mb_embeds, mb_avail)  # [B,N,A], [B,N]
                        logits_att = logits_att.float()
                        val_att_all = val_att_all.float()

                        mask_sum = mb_vmask.sum(dim=1, keepdim=True).clamp(min=1.0)
                        val_att = (val_att_all * mb_vmask).sum(dim=1) / mask_sum.squeeze(-1)

                        dist_att = Categorical(logits=logits_att)
                        all_lp_att = dist_att.log_prob(torch.clamp(mb_advact, min=0))          # [B,N]
                        new_lp_att_step = (all_lp_att * mb_vmask).sum(dim=1)                   # [B]

                        row_is_attack = attack_bool.any(dim=1)
                        active_mb_idx = torch.where(row_is_attack)[0]

                        if len(active_mb_idx) > 0:
                            ratio_att = torch.exp(new_lp_att_step[active_mb_idx] - mb_old_lp_att[active_mb_idx])
                            surr1_a = ratio_att * mb_adv[active_mb_idx]
                            surr2_a = torch.clamp(ratio_att, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv[active_mb_idx]
                            loss_att_pi = -torch.min(surr1_a, surr2_a).mean()

                            loss_att_v = 0.5 * ((val_att[active_mb_idx] - mb_returns[active_mb_idx]) ** 2).mean()

                            ent_att_all = dist_att.entropy()  # [B,N]
                            loss_att_ent = (ent_att_all * mb_vmask).sum() / (attack_bool.sum() + 1e-8)

                            loss_att = loss_att_pi + self.vf_coef * loss_att_v - self.ent_coef * loss_att_ent

                    total_loss = loss_sel + loss_att

                self.opt_gnn.zero_grad()
                self.opt_sel.zero_grad()
                self.opt_att.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.sel.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.att.parameters(), 0.5)

                self.opt_gnn.step()
                self.opt_sel.step()
                self.opt_att.step()

                loss_pi_list.append(float(loss_sel_pi.item()))
                loss_v_list.append(float(loss_sel_v.item()))
                loss_ent_list.append(float(entropy_sel.item()))

        return {
            "policy_loss": float(np.mean(loss_pi_list)) if loss_pi_list else 0.0,
            "value_loss": float(np.mean(loss_v_list)) if loss_v_list else 0.0,
            "entropy": float(np.mean(loss_ent_list)) if loss_ent_list else 0.0,
            "meta_loss": float(np.mean(loss_meta_list)) if loss_meta_list else 0.0,
        }

    # -------------------------------------------------
    # IO
    # -------------------------------------------------
    def load(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.gnn.load_state_dict(ckpt["models"]["gnn"])
        self.sel.load_state_dict(ckpt["models"]["sel"])
        self.att.load_state_dict(ckpt["models"]["att"])
        if "analyst" in ckpt.get("models", {}):
            self.analyst.load_state_dict(ckpt["models"]["analyst"])
        print(f"[BudgetPPOAgent] Loaded checkpoint from {checkpoint_path}")

    def has_pending_data(self):
        if getattr(self, "temp_transition", None):
            return True
        if getattr(self, "buffer", None) is not None and len(self.buffer.rewards) > 0:
            return True
        if getattr(self, "episode_counter", 0) > 0:
            return True
        tb = getattr(self, "training_batch", None)
        if tb is not None:
            for k, v in tb.items():
                if isinstance(v, list) and len(v) > 0:
                    return True
        return False
