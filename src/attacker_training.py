import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from types import SimpleNamespace as SN
from os.path import join
import argparse
import random
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]   
SC2_DIR = PROJECT_ROOT / "3rdparty" / "StarCraftII"
print(str(SC2_DIR))
os.environ["SC2PATH"] = str(SC2_DIR)
BASE_DIR = "./src"
SAVE_DIR = "./results"

USE_DELTA_V = True
PRETRAIN_MODEL_DIR = ""

def set_global_seeds(seed: int, device: str = "cuda"):
    print(f"[Seed] Setting global seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from controllers import REGISTRY as mac_REGISTRY
    from smac.env import StarCraft2Env
except ImportError as e:
    print("【错误】无法导入 PyMARL 组件。")
    raise e


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            recursive_dict_update(d[k], v)
        else:
            d[k] = v
    return d

def load_args(env_config_name="sc2", alg_config_name="qmix"):
    config_dir = join(BASE_DIR, "config")
    with open(join(config_dir, "default.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    with open(join(config_dir, "envs", f"{env_config_name}.yaml"), "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(join(config_dir, "algs", f"{alg_config_name}.yaml"), "r") as f:
        alg_cfg = yaml.safe_load(f)
    cfg = recursive_dict_update(cfg, env_cfg)
    cfg = recursive_dict_update(cfg, alg_cfg)
    args = SN(**cfg)
    args.device = "cuda" if getattr(args, "use_cuda", False) and torch.cuda.is_available() else "cpu"
    return args

def build_scheme(args, env_info):
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    scheme = {
        "state": {"vshape": args.state_shape},
        "obs": {"vshape": args.obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "actions_onehot": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.float32},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    return args, scheme, groups, preprocess

def load_pretrained_mac(args, scheme, groups, pretrain_dir):
    mac = mac_REGISTRY[args.mac](scheme, groups, args)
    agent_path = os.path.join(pretrain_dir, "agent.th")
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"预训练模型未找到: {agent_path}")
    state_dict = torch.load(agent_path, map_location=args.device)
    mac.agent.load_state_dict(state_dict)
    mac.agent.to(args.device)
    mac.agent.eval()
    return mac

class LinearSchedule:
    def __init__(self, start, end, duration):
        self.start = start
        self.end = end
        self.duration = duration
    def get_value(self, t):
        if self.duration <= 0: return self.end
        if t >= self.duration: return self.end
        return self.start + (self.end - self.start) * (t / self.duration)

def create_lr_scheduler(optimizer, total_episodes):
    def lr_lambda(ep):
        return max(0.0, float(total_episodes - ep) / float(total_episodes))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class GNNEncoder(nn.Module):
    def __init__(self, obs_dim, embed_dim):
        super().__init__()
        self.self_mlp = layer_init(nn.Linear(obs_dim, embed_dim))
        self.nei_mlp = layer_init(nn.Linear(obs_dim, embed_dim))
    def forward(self, obs_all):
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
            nn.ReLU()
        )
        self.agent_policy_net = nn.Sequential(
            layer_init(nn.Linear(embed_dim + hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01)
        )
        self.noop_policy_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01)
        )
        self.v_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, agent_embeds, state_batch, budget_ratio):
        B, N, E = agent_embeds.shape
        global_embed = agent_embeds.mean(dim=1)
        if budget_ratio.dim() == 1:
            budget_ratio = budget_ratio.unsqueeze(1)
        global_input = torch.cat([global_embed, state_batch, budget_ratio], dim=-1)
        global_feat = self.global_feature_net(global_input)

        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        agent_input = torch.cat([agent_embeds, global_feat_expanded], dim=-1)
        agent_logits = self.agent_policy_net(agent_input).squeeze(-1)
        noop_logit = self.noop_policy_net(global_feat)
        total_logits = torch.cat([agent_logits, noop_logit], dim=-1)

        v_value = self.v_net(global_feat).squeeze(-1)
        return total_logits, v_value

class BudgetAnalyst(nn.Module):
    def __init__(self, embed_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(embed_dim + state_dim + 1, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
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
        return loss.item()

class ActionAttackerMulti(nn.Module):
    def __init__(self, obs_dim, embed_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.policy_mlp = nn.Sequential(
            layer_init(nn.Linear(obs_dim + embed_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        )
        self.value_mlp = nn.Sequential(
            layer_init(nn.Linear(obs_dim + embed_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, obs_all, agent_embeds, avail_actions_all=None):
        x = torch.cat([obs_all, agent_embeds], dim=-1)
        logits = self.policy_mlp(x)
        if avail_actions_all is not None:
            logits = logits.masked_fill(avail_actions_all == 0, -1e10)
        values = self.value_mlp(x).squeeze(-1)
        return logits, values

    def act_single(self, obs_i, embed_i, avail_actions_i, device):
        obs_t = torch.as_tensor(obs_i, dtype=torch.float32, device=device).view(1, 1, -1)
        emb_t = torch.as_tensor(embed_i, dtype=torch.float32, device=device).view(1, 1, -1)
        avail_t = torch.as_tensor(avail_actions_i, dtype=torch.float32, device=device).view(1, 1, -1)
        logits, values = self.forward(obs_t, emb_t, avail_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach(), values.detach()


def build_budget_action_mask(n_agents: int, current_budget: int, device):
    mask = torch.ones(n_agents + 1, dtype=torch.float32, device=device)
    if current_budget <= 0:
        mask[:-1] = 0.0
    return mask

def sample_selector_gate_k1(logits_sel_1x, budget_mask_1d, eps=1e-8):
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

def recompute_logp_selector_gate_k1(masked_logits_bn1, idx_seq_bk, attacked_bool, eps=1e-8):
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

    masked_logits = logits.clone()
    masked_logits[combined_mask == 0] = -1e10
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


class RolloutBuffer:
    def __init__(self, max_k: int):
        self.max_k = max_k

        self.states, self.obs_all, self.avail_all = [], [], []
        self.victim_mask, self.adv_actions = [], []

        self.idx_seq = []
        self.sel_seq = []

        self.logprob_sel, self.logprob_att = [], []
        self.value_sel, self.value_att = [], []
        self.rewards, self.dones, self.budgets, self.action_masks = [], [], [], []

        self.global_embeds = []
        self.opp_costs = []

    def add(
        self, state, obs, avail,
        v_mask, adv_act,
        idx_seq, sel_seq,
        lp_sel, lp_att, v_sel, v_att,
        r, d, bud, act_mask,
        global_embed=None, opp_cost=0.0
    ):
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

        if global_embed is None:
            self.global_embeds.append(None)
        else:
            self.global_embeds.append(global_embed.astype(np.float32))
        self.opp_costs.append(float(opp_cost))

    def clear(self):
        self.__init__(self.max_k)

    def get_data(self, device, gamma=0.99, gae_lambda=0.95, next_value=None):
        states = to_tensor(np.array(self.states), device)
        obs = to_tensor(np.array(self.obs_all), device)
        avail = to_tensor(np.array(self.avail_all), device)
        v_mask = to_tensor(np.array(self.victim_mask), device)
        adv_act = torch.tensor(np.array(self.adv_actions), dtype=torch.long, device=device)

        idx_seq = torch.tensor(np.array(self.idx_seq), dtype=torch.long, device=device)  # [T,K]
        sel_seq = torch.tensor(np.array(self.sel_seq), dtype=torch.long, device=device)  # [T,K]

        rewards = to_tensor(np.array(self.rewards), device)
        dones = to_tensor(np.array(self.dones), device)
        budgets = to_tensor(np.array(self.budgets), device).unsqueeze(1)
        act_masks = to_tensor(np.array(self.action_masks), device)  # [T,N+1]

        old_lp_sel = to_tensor(np.array(self.logprob_sel), device)
        old_lp_att = to_tensor(np.array(self.logprob_att), device)
        old_v_sel = to_tensor(np.array(self.value_sel), device)
        old_v_att = to_tensor(np.array(self.value_att), device)

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

def ppo_update(
    models, optimizers, buffer,
    gamma, clip_coef, ent_coef, vf_coef,
    ppo_epochs, mini_batch_size, device,
    attack_n: int,
    ablation_worst_action=False,
    deltaV_coef=0.0, deltaV_delta_b=0.1,
    next_value=None
):
    global USE_DELTA_V

    eps = 1e-8
    gnn, sel = models["gnn"], models["sel"]
    opt_gnn, opt_sel = optimizers["gnn"], optimizers["sel"]
    att = models.get("att", None)
    opt_att = optimizers.get("att", None)

    data = buffer.get_data(device, gamma, gae_lambda=0.95, next_value=next_value)
    states, obs, avail = data["states"], data["obs"], data["avail"]
    v_mask = data["v_mask"]
    idx_seq = data["idx_seq"]
    sel_seq = data["sel_seq"]
    budgets, act_masks = data["budgets"], data["act_masks"]
    adv_act = data["adv_act"]
    old_lp_sel, old_lp_att = data["old_lp_sel"], data["old_lp_att"]
    advantages, returns = data["advantages"], data["returns"]

    batch_size = states.shape[0]
    b_inds = np.arange(batch_size)

    loss_pi_list, loss_v_list, loss_ent_list, loss_meta_list = [], [], [], []

    for _ in range(ppo_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = b_inds[start:end]

            mb_obs = obs[mb_inds]
            mb_states = states[mb_inds]
            mb_budgets = budgets[mb_inds]
            mb_vmask = v_mask[mb_inds]
            mb_act_masks = act_masks[mb_inds]
            mb_idx_seq = idx_seq[mb_inds]
            mb_sel_seq = sel_seq[mb_inds]

            mb_embeds = gnn(mb_obs)
            logits_sel, val_sel = sel(mb_embeds, mb_states, mb_budgets)

            attacked = (mb_vmask.sum(dim=1) > 0.5)

            if int(attack_n) == 1:
                masked_logits = logits_sel.clone()
                masked_logits[mb_act_masks == 0] = -1e10
                new_lp_sel, entropy_sel = recompute_logp_selector_gate_k1(
                    masked_logits, mb_idx_seq, attacked, eps=eps
                )
            else:
                masked_logits_ent = logits_sel.clone()
                masked_logits_ent[mb_act_masks == 0] = -1e10
                dist_sel_ent = Categorical(logits=masked_logits_ent)
                entropy_sel = dist_sel_ent.entropy().mean()

                new_lp_sel = compute_new_lp_sel_autoregressive(
                    logits_sel=logits_sel,
                    sel_seq=mb_sel_seq,
                    act_masks=mb_act_masks
                )

            # normalize adv
            mb_adv = advantages[mb_inds]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            ratio_sel = torch.exp(new_lp_sel - old_lp_sel[mb_inds])
            surr1 = ratio_sel * mb_adv
            surr2 = torch.clamp(ratio_sel, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
            loss_sel_pi = -torch.min(surr1, surr2).mean()

            loss_sel_v = 0.5 * ((val_sel - returns[mb_inds]) ** 2).mean()
            loss_sel = loss_sel_pi + vf_coef * loss_sel_v - ent_coef * entropy_sel

            # Delta-V meta loss
            loss_meta = torch.tensor(0.0, device=device)
            if USE_DELTA_V and deltaV_coef > 0.0:
                attack_indicator = attacked.float()
                b_curr = mb_budgets
                b_prev = torch.clamp(b_curr - deltaV_delta_b, 0.0, 1.0)
                _, v_prev = sel(mb_embeds, mb_states, b_prev)
                delta_v = torch.relu(val_sel - v_prev)
                loss_meta = (delta_v * attack_indicator).mean()
                loss_sel = loss_sel + deltaV_coef * loss_meta
                loss_meta_list.append(loss_meta.item())

            loss_att = torch.tensor(0.0, device=device)
            if (not ablation_worst_action) and (att is not None):
                logits_att, val_att_all = att(mb_obs, mb_embeds, avail[mb_inds])

                mask_sum = mb_vmask.sum(dim=1, keepdim=True).clamp(min=1.0)
                val_att = (val_att_all * mb_vmask).sum(dim=1) / mask_sum.squeeze(-1)

                mb_adv_act = adv_act[mb_inds]
                attack_bool = (mb_vmask > 0.5) & (mb_adv_act >= 0)

                if attack_bool.sum() > 0:
                    dist_att = Categorical(logits=logits_att)
                    all_lp_att = dist_att.log_prob(torch.clamp(mb_adv_act, min=0))
                    new_lp_att_step = (all_lp_att * mb_vmask).sum(dim=1)

                    row_is_attack = attack_bool.any(dim=1)
                    active_mb_idx = torch.where(row_is_attack)[0]

                    if len(active_mb_idx) > 0:
                        ratio_att = torch.exp(new_lp_att_step[active_mb_idx] - old_lp_att[mb_inds][active_mb_idx])
                        surr1_a = ratio_att * mb_adv[active_mb_idx]
                        surr2_a = torch.clamp(ratio_att, 1 - clip_coef, 1 + clip_coef) * mb_adv[active_mb_idx]
                        loss_att_pi = -torch.min(surr1_a, surr2_a).mean()

                        loss_att_v = 0.5 * ((val_att[active_mb_idx] - returns[mb_inds][active_mb_idx]) ** 2).mean()

                        ent_att_all = dist_att.entropy()
                        loss_att_ent = (ent_att_all * mb_vmask).sum() / (attack_bool.sum() + 1e-8)

                        loss_att = loss_att_pi + vf_coef * loss_att_v - ent_coef * loss_att_ent

            total_loss = loss_sel + loss_att

            loss_pi_list.append(loss_sel_pi.item())
            loss_v_list.append(loss_sel_v.item())
            loss_ent_list.append(float(entropy_sel.item()))
            opt_gnn.zero_grad()
            opt_sel.zero_grad()
            if (not ablation_worst_action) and (att is not None):
                opt_att.zero_grad()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(sel.parameters(), 0.5)
            if (not ablation_worst_action) and (att is not None):
                torch.nn.utils.clip_grad_norm_(att.parameters(), 0.5)

            opt_gnn.step()
            opt_sel.step()
            if (not ablation_worst_action) and (att is not None):
                opt_att.step()

    # print("autocast:", torch.is_autocast_enabled())
    # mb_embeds = gnn(mb_obs)
    # print("mb_obs dtype:", mb_obs.dtype, "mb_embeds dtype:", mb_embeds.dtype)
    # logits_sel, _ = sel(mb_embeds, mb_states, mb_budgets)
    # print("logits_sel dtype:", logits_sel.dtype)


    return {
        "policy_loss": float(np.mean(loss_pi_list)) if loss_pi_list else 0.0,
        "value_loss": float(np.mean(loss_v_list)) if loss_v_list else 0.0,
        "entropy": float(np.mean(loss_ent_list)) if loss_ent_list else 0.0,
        "meta_loss": float(np.mean(loss_meta_list)) if loss_meta_list else 0.0,
    }


def train_budget_aware_adversary(
    map_name="2s3z",
    total_episodes=5000,
    max_attack_budget_start=0,
    max_attack_budget_end=10,
    budget_anneal_episodes=2000,
    rollout_size=2048,
    mini_batch_size=512,
    device="cuda",
    load_model=False,
    save_interval=50000,
    print_interval=50,
    ablation_worst_action=False,
    seed=0,
    deltaV_coef=0.0,
    attack_cost=0.0,
    attack_n=1,
    eval=False
):
    global PRETRAIN_MODEL_DIR
    EVAL = eval

    env = StarCraft2Env(map_name=map_name, difficulty="7", seed=seed, reward_death_value=10)
    env_test = StarCraft2Env(map_name=map_name, difficulty="7", seed=seed, reward_death_value=10)
    env_info = env.get_env_info()

    args = load_args()
    args.env_args["map_name"] = map_name
    args.device = device
    args, scheme, groups, preprocess = build_scheme(args, env_info)
    mac = load_pretrained_mac(args, scheme, groups, PRETRAIN_MODEL_DIR)
    mac_test = load_pretrained_mac(args, scheme, groups, PRETRAIN_MODEL_DIR)

    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]
    n_actions = env_info["n_actions"]
    embed_dim = 64
    dev = torch.device(device)

    models = {
        "gnn": GNNEncoder(obs_dim, embed_dim).to(dev),
        "sel": VictimSelectorBudget(embed_dim, state_dim).to(dev),
    }
    optimizers = {
        "gnn": optim.Adam(models["gnn"].parameters(), lr=3e-4, eps=1e-5),
        "sel": optim.Adam(models["sel"].parameters(), lr=3e-4, eps=1e-5),
    }
    schedulers = {
        "gnn": create_lr_scheduler(optimizers["gnn"], total_episodes),
        "sel": create_lr_scheduler(optimizers["sel"], total_episodes),
    }
    if not ablation_worst_action:
        models["att"] = ActionAttackerMulti(obs_dim, embed_dim, n_actions).to(dev)
        optimizers["att"] = optim.Adam(models["att"].parameters(), lr=3e-4, eps=1e-5)

    analyst = BudgetAnalyst(embed_dim, state_dim).to(dev)
    models["analyst"] = analyst
    optimizers["analyst"] = analyst.optimizer

    # load_models("/home/data/sdb5/jiangjunyong/results/MMM/adv_model_loop2/models_ep0.pt",models)

    budget_scheduler = LinearSchedule(max_attack_budget_start, max_attack_budget_end, budget_anneal_episodes)

    buffer = RolloutBuffer(max_k=max(1, int(attack_n)))

    ep_rewards = deque(maxlen=100)
    ep_lens = deque(maxlen=100)
    win_rates = deque(maxlen=100)
    attack_nums = deque(maxlen=100)
    agents_attacked_nums = deque(maxlen=100)

    est_values = deque(maxlen=2000)
    opp_costs = deque(maxlen=2000)

    last_train_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "meta_loss": 0.0}
    last_analyst_loss = 0.0

    print(f"Start Training. Seed={seed}, attack_n={attack_n}")
    if int(attack_n) == 1:
        print("  -> attack_n==1 uses CODE-A (gate+k1 strict logp)")
    else:
        print("  -> attack_n>1 uses CODE-B (FORCE_FILL + autoreg logp + analyst randperm)")

    ep = 0
    while ep < total_episodes:
        env.reset()
        mac.init_hidden(batch_size=1)
        
        terminated = False
        ep_reward = 0.0
        step = 0
        attack_num = 0
        ep_agent_attacks = 0

        cur_max_budget = int(budget_scheduler.get_value(ep))
        current_budget = cur_max_budget

        ep_batch = EpisodeBatch(
            scheme, groups, 1, env_info["episode_limit"] + 1,
            device=device, preprocess=preprocess
        )

        while not terminated:
            obs_list = env.get_obs()
            state = env.get_state()
            avail = [env.get_avail_agent_actions(i) for i in range(args.n_agents)]

            obs_np = np.stack(obs_list, axis=0)
            avail_np = np.stack(avail, axis=0)

            obs_t = to_tensor(obs_np, device).unsqueeze(0)
            state_t = to_tensor(state, device).unsqueeze(0)
            avail_t = to_tensor(avail_np, device).unsqueeze(0)

            ep_batch.update({"obs": obs_t, "avail_actions": avail_t}, bs=0, ts=step)

            agent_outs = mac.forward(ep_batch, t=step, test_mode=False)
            agent_outs[avail_t == 0] = -1e10
            base_actions = agent_outs.argmax(dim=-1).cpu().numpy().flatten()
            final_actions = base_actions.copy()

            v_mask_np = np.zeros(args.n_agents, dtype=np.float32)
            adv_actions_step = np.full(args.n_agents, -1, dtype=np.int64)

            # unified sequence storage
            idx_seq_np = np.full((max(1, int(attack_n)),), -1, dtype=np.int64)   # A uses
            sel_seq_np = np.full((max(1, int(attack_n)),), -1, dtype=np.int64)   # B uses (noop=N)

            lp_sel_total = 0.0
            lp_att_total = 0.0
            val_att_step = 0.0
            step_has_attacked = False
            num_attacked_this_step = 0

            # ---- forward ----
            with torch.no_grad():
                emb = models["gnn"](obs_t).squeeze(0)  # [N,E]
                global_embed = emb.mean(dim=0, keepdim=True)  # [1,E]

                if int(attack_n) == 1:
                    # ===== CODE-A =====
                    denom_budget = float(cur_max_budget) if cur_max_budget > 0 else 1.0
                    bud_ratio = current_budget / denom_budget
                    bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=device)

                    logits_sel, val_sel = models["sel"](emb.unsqueeze(0), state_t, bud_t)
                    logits_sel = logits_sel.squeeze(0)

                    budget_mask = build_budget_action_mask(args.n_agents, current_budget, device)

                    step_has_attacked, vid, lp = sample_selector_gate_k1(logits_sel, budget_mask)
                    lp_sel_total = lp

                    act_mask_store = budget_mask.detach().cpu().numpy()
                    bud_store = float(bud_ratio)

                    if step_has_attacked:
                        v_mask_np[vid] = 1.0
                        idx_seq_np[0] = vid
                        num_attacked_this_step = 1

                        # attacker
                        obs_i = obs_np[vid]
                        emb_i = emb[vid].detach().cpu().numpy()
                        avail_i = avail_np[vid]

                        adv_a, lp_a, v_a = models["att"].act_single(obs_i, emb_i, avail_i, device)
                        final_actions[vid] = adv_a
                        adv_actions_step[vid] = adv_a

                        lp_att_total += float(lp_a.item())
                        val_att_step = float(v_a.item())

                        if current_budget > 0:
                            current_budget -= 1
                        attack_num += 1
                        ep_agent_attacks += 1

                else:
                    bud_ratio_pre = current_budget / max(1.0, cur_max_budget)
                    bud_t = torch.tensor([[bud_ratio_pre]], device=device)

                    logits_sel, val_sel = models["sel"](emb.unsqueeze(0), state_t, bud_t)
                    est_v = models["analyst"](global_embed, state_t, bud_t).item()
                    est_values.append(est_v)

                    exclude_mask = torch.zeros((1, args.n_agents), device=device)
                    noop_idx = args.n_agents

                    base_mask_for_store = None
                    val_att_acc = 0.0

                    for k in range(int(attack_n)):
                        forbid_noop = (k > 0) and step_has_attacked
                        idx_t, lp_sel_k, base_mask = sample_with_budget_mask_strict(
                            logits_sel, current_budget, device,
                            exclude_mask=exclude_mask,
                            forbid_noop=forbid_noop
                        )
                        idx = idx_t.item()

                        if base_mask_for_store is None:
                            base_mask_for_store = base_mask.detach().clone()

                        sel_seq_np[k] = idx

                        if idx >= args.n_agents:
                            if k == 0:
                                break
                            else:
                                break

                        # victim
                        target = idx
                        step_has_attacked = True
                        v_mask_np[target] = 1.0
                        exclude_mask[0, target] = 1.0

                        obs_i = obs_np[target]
                        emb_i = emb[target].detach().cpu().numpy()
                        avail_i = avail_np[target]

                        adv_a, lp_a, v_a = models["att"].act_single(obs_i, emb_i, avail_i, device)

                        final_actions[target] = adv_a
                        adv_actions_step[target] = adv_a

                        lp_sel_total += float(lp_sel_k.item())
                        lp_att_total += float(lp_a.item())
                        val_att_acc += float(v_a.item())
                        num_attacked_this_step += 1

                    if num_attacked_this_step > 0:
                        val_att_step = val_att_acc / num_attacked_this_step
                        ep_agent_attacks += num_attacked_this_step
                        current_budget -= 1
                        attack_num += 1

                    act_mask_store = base_mask_for_store.cpu().numpy()
                    bud_store = float(bud_ratio_pre)

            if map_name=="2s3z":
                actions_t = to_tensor(final_actions, device).unsqueeze(1).unsqueeze(0) 
                ep_batch.update({"actions": actions_t}, bs=0, ts=step)
            reward, terminated, info = env.step(final_actions)

            opp_cost = (attack_cost if step_has_attacked else 0.0)
            opp_costs.append(opp_cost)

            adv_r = -reward - opp_cost
            ep_reward += adv_r

            buffer.add(
                state, obs_np, avail_np,
                v_mask_np, adv_actions_step,
                idx_seq_np, sel_seq_np,
                float(lp_sel_total), float(lp_att_total),
                float(val_sel.item()), float(val_att_step),
                float(adv_r), float(terminated),
                float(bud_store),
                act_mask_store,
                global_embed=global_embed.squeeze(0).detach().cpu().numpy(),
                opp_cost=opp_cost
            )

            step += 1

            if len(buffer.rewards) >= rollout_size:
                if int(attack_n) == 1:
                    next_value = None
                    if not terminated:
                        with torch.no_grad():
                            next_obs_list = env.get_obs()
                            next_state = env.get_state()
                            next_obs_np = np.stack(next_obs_list, axis=0)
                            next_obs_t = to_tensor(next_obs_np, device).unsqueeze(0)
                            next_state_t = to_tensor(next_state, device).unsqueeze(0)

                            emb_next = models["gnn"](next_obs_t).squeeze(0)
                            denom_budget = float(cur_max_budget) if cur_max_budget > 0 else 1.0
                            bud_ratio_next = current_budget / denom_budget
                            bud_t_next = torch.tensor([[bud_ratio_next]], dtype=torch.float32, device=device)
                            _, next_val_sel = models["sel"](emb_next.unsqueeze(0), next_state_t, bud_t_next)
                            next_value = float(next_val_sel.squeeze().item())
                else:
                    next_value = None


                last_train_metrics = ppo_update(
                    models, optimizers, buffer,
                    gamma=0.99, clip_coef=0.2, ent_coef=0.05, vf_coef=0.5,
                    ppo_epochs=4, mini_batch_size=mini_batch_size,
                    device=device,
                    attack_n=int(attack_n),
                    ablation_worst_action=ablation_worst_action,
                    deltaV_coef=deltaV_coef,
                    next_value=next_value
                )

                with torch.no_grad():
                    d = buffer.get_data(device, gamma=0.99, gae_lambda=0.95, next_value=next_value)
                    returns_t = d["returns"].detach()
                    states_t = d["states"].detach()
                    budgets_t = d["budgets"].detach()
                    obs_t2 = d["obs"].detach()
                    embeds_t = models["gnn"](obs_t2).detach()
                    global_emb_t = embeds_t.mean(dim=1)

                if int(attack_n) == 1:
                    last_analyst_loss = models["analyst"].update(
                        global_emb_t, states_t, budgets_t, returns_t
                    )
                else:
                    idxs = torch.randperm(global_emb_t.shape[0], device=device)[:min(512, global_emb_t.shape[0])]
                    last_analyst_loss = models["analyst"].update(
                        global_emb_t[idxs], states_t[idxs], budgets_t[idxs], returns_t[idxs]
                    )

                buffer.clear()

        ep += 1
        ep_rewards.append(ep_reward)
        ep_lens.append(step)
        attack_nums.append(attack_num)
        agents_attacked_nums.append(ep_agent_attacks)
        win_rates.append(1 if info.get("battle_won", False) else 0)

        if (ep % print_interval) == 0:
            if not EVAL:
                avg_est_value = float(np.mean(est_values)) if len(est_values) > 0 else 0.0
                avg_opp_cost = float(np.mean(opp_costs)) if len(opp_costs) > 0 else 0.0

                print(f"\n---------------------------------")
                print(f"| rollout/          |                      |")
                print(f"|    ep_len_mean    | {np.mean(ep_lens) if ep_lens else 0:.1f}        |")
                # print(f"|    ep_rew_mean    | {np.mean(ep_rewards) if ep_rewards else 0:.2f}  |")
                # print(f"|    win_rate       | {np.mean(win_rates) if win_rates else 0:.2%}    |")
                print(f"|    attack_num     | {np.mean(attack_nums) if attack_nums else 0:.2f}|")
                print(f"|    attack agents  | {np.mean(agents_attacked_nums) if agents_attacked_nums else 0:.2f}|")
                print(f"|    cur_max_budget | {cur_max_budget}                  |")
                print(f"| explanation/      |                      |")
                print(f"|    avg_val_est    | {avg_est_value:.4f}         |")
                print(f"|    avg_opp_cost   | {avg_opp_cost:.4f}          |")
                print(f"| train/            |                      |")
                print(f"|    policy_loss    | {last_train_metrics['policy_loss']:.4f}        |")
                print(f"|    value_loss     | {last_train_metrics['value_loss']:.4f}         |")
                print(f"|    meta_loss      | {last_train_metrics['meta_loss']:.6f}          |")
                print(f"|    analyst_loss   | {float(last_analyst_loss):.4f}         |")
                print(f"---------------------------------")
            else:
                # ===== EVAL after each rollout update =====
                mac_test.agent.load_state_dict(mac.agent.state_dict())
                mac_test.agent.eval()
                eval_metrics = eval_adversary(
                    models=models,
                    mac=mac_test,
                    args=args,
                    map_name=map_name,
                    device=device,
                    n_eval_episodes=50,         
                    attack_n=int(attack_n),
                    fixed_budget=max_attack_budget_end, 
                    attack_cost=attack_cost,
                    env=env_test
                )
                print(
                    f"[EVAL] Win_Rate={eval_metrics['eval_win_rate']:.2%} | "
                    f"adv_return={eval_metrics['eval_adv_return']:.2f} | "
                    f"ep_len={eval_metrics['eval_ep_len']:.1f} | "
                    f"attack_num={eval_metrics['eval_attack_num']:.2f} | "
                    f"attack_agents={eval_metrics['eval_attack_agents']:.2f}"
                    )

    # ===== 保存模型 =====
        if (save_interval is not None) and (save_interval > 0) and (ep % save_interval == 0):
            save_models(save_dir=SAVE_DIR, tag="models", ep=ep, models=models)

    env.close()
    save_models(save_dir=SAVE_DIR, tag="models", ep=0, models=models)

def save_models(save_dir: str, tag: str, ep: int, models: dict):
    os.makedirs(save_dir, exist_ok=True)
    payload = {"episode": ep, "models": {}}
    for k, m in models.items():
        if isinstance(m, nn.Module):
            payload["models"][k] = m.state_dict()
    path = os.path.join(save_dir, f"{tag}_ep{ep}.pt")
    torch.save(payload, path)
    print(f"[Save] models -> {path}")

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


@torch.no_grad()
def greedy_selector_attack_n1_gate_k1(logits_sel_1d, budget_mask_1d, eps=1e-8):
    """
    attack_n == 1 的确定性策略：
    - 先算 p_attack = 1 - p_noop
    - 若 p_attack >= p_noop 且预算允许，则攻击，并在 agent_probs 里选 argmax
    - 否则 noop
    返回: (step_has_attacked: bool, vid: int)
    """
    masked_logits = logits_sel_1d.clone()
    masked_logits[budget_mask_1d == 0] = -1e10

    probs = F.softmax(masked_logits, dim=-1)              # [N+1]
    p_noop = probs[-1].clamp(min=eps, max=1.0)
    p_attack = (1.0 - p_noop).clamp(min=eps, max=1.0)

    # 没预算：必 noop
    if budget_mask_1d[:-1].sum().item() < 0.5:
        return False, -1

    # gate：确定性判别（attack vs noop）
    if float(p_attack.item()) < float(p_noop.item()):
        return False, -1

    # 选 victim：agent_probs argmax
    agent_probs = probs[:-1]
    if agent_probs.sum().item() <= eps:
        return False, -1
    vid = int(torch.argmax(agent_probs).item())
    return True, vid


@torch.no_grad()
def greedy_selector_attack_n_gt1_forcefill(logits_sel_1x, current_budget, device, n_agents, attack_n: int):
    """
    attack_n > 1 的确定性策略（FORCE_FILL 风格）：
    - k=0..K-1 逐步选 idx
    - 预算<=0：只能 noop
    - 第一次攻击后 forbid_noop（和你训练分支一致）
    - victim 选择使用 masked_logits 的 argmax
    返回:
      sel_seq_np: [K] (noop = N)
      victim_list: list[int]
      step_has_attacked: bool
      base_mask_for_store: [N+1] (budget-only mask)
    """
    K = int(attack_n)
    noop_idx = n_agents

    logits = logits_sel_1x.squeeze(0)  # [N+1]
    base_mask = torch.ones_like(logits, device=device)
    if current_budget <= 0:
        base_mask[:-1] = 0.0
    base_mask_for_store = base_mask.detach().clone()

    sel_seq = [noop_idx] * K
    victim_list = []
    exclude = torch.zeros(n_agents, device=device)  # 0/1
    step_has_attacked = False

    for k in range(K):
        combined_mask = base_mask.clone()
        # exclude 已选 victim
        combined_mask[:-1] = combined_mask[:-1] * (1.0 - exclude)

        # forbid noop if already attacked (k>0 且 step_has_attacked)
        if (k > 0) and step_has_attacked:
            combined_mask[-1] = 0.0

        masked_logits = logits.clone()
        masked_logits[combined_mask == 0] = -1e10

        idx = int(torch.argmax(masked_logits).item())
        sel_seq[k] = idx

        if idx >= n_agents:
            # noop
            break

        # victim
        step_has_attacked = True
        victim_list.append(idx)
        exclude[idx] = 1.0

    return np.array(sel_seq, dtype=np.int64), victim_list, step_has_attacked, base_mask_for_store


@torch.no_grad()
def greedy_attacker_action(models, obs_np, emb, avail_np, target, device):

    obs_t = torch.as_tensor(obs_np[target], dtype=torch.float32, device=device).view(1, 1, -1)
    emb_t = emb[target].view(1, 1, -1)
    avail_t = torch.as_tensor(avail_np[target], dtype=torch.float32, device=device).view(1, 1, -1)
    logits, _ = models["att"](obs_t, emb_t, avail_t)  
    a = int(torch.argmax(logits.squeeze(0).squeeze(0)).item())
    return a


def eval_adversary(
    models, mac, args,
    map_name: str,
    device: str = "cuda",
    n_eval_episodes: int = 20,
    attack_n: int = 1,
    fixed_budget: int = 10,          
    attack_cost: float = 0.0,
    env=None
):
    """
    每次 rollout update 后跑一次评估（确定性 greedy）：
    输出：
      - eval_win_rate: victim 是否赢（battle_won）
      - eval_adv_return: adversary return（你训练里是 -reward - cost 的累积）
      - eval_attack_num: 每局攻击 step 次数均值（step_has_attacked 计数）
      - eval_attack_agents: 每局被攻击 agent 数均值（累积 attacked victims 数）
      - eval_ep_len
    """
    dev = torch.device(device)

    env_info = env.get_env_info()

    args_eval, scheme, groups, preprocess = build_scheme(args, env_info)
    args_eval.device = device

    win_list = []
    adv_ret_list = []
    ep_len_list = []
    attack_num_list = []
    attack_agents_list = []


    for k, m in models.items():
        if isinstance(m, nn.Module):
            m.eval()

    for _ in range(n_eval_episodes):
        env.reset()
        mac.init_hidden(batch_size=1)
        terminated = False
        step = 0

        cur_max_budget = int(fixed_budget)
        current_budget = int(fixed_budget)

        adv_return = 0.0
        attack_num = 0
        attacked_agents = 0

        ep_batch = EpisodeBatch(
            scheme, groups, 1, env_info["episode_limit"] + 1,
            device=device, preprocess=preprocess
        )

        while not terminated:
            obs_list = env.get_obs()
            state = env.get_state()
            avail = [env.get_avail_agent_actions(i) for i in range(args_eval.n_agents)]

            obs_np = np.stack(obs_list, axis=0)
            avail_np = np.stack(avail, axis=0)

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev).unsqueeze(0)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
            avail_t = torch.as_tensor(avail_np, dtype=torch.float32, device=dev).unsqueeze(0)

            ep_batch.update({"obs": obs_t, "avail_actions": avail_t}, bs=0, ts=step)

            # victim policy
            agent_outs = mac.forward(ep_batch, t=step, test_mode=True)   # eval 用 test_mode=True
            agent_outs[avail_t == 0] = -1e10
            base_actions = agent_outs.argmax(dim=-1).cpu().numpy().flatten()
            final_actions = base_actions.copy()

            step_has_attacked = False
            num_attacked_this_step = 0

            # embeds
            emb = models["gnn"](obs_t).squeeze(0)  # [N,E]

            if int(attack_n) == 1:
                denom_budget = float(cur_max_budget) if cur_max_budget > 0 else 1.0
                bud_ratio = current_budget / denom_budget
                bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=dev)

                logits_sel, _ = models["sel"](emb.unsqueeze(0), state_t, bud_t)
                logits_sel = logits_sel.squeeze(0)  # [N+1]

                budget_mask = build_budget_action_mask(args_eval.n_agents, current_budget, dev)
                step_has_attacked, vid = greedy_selector_attack_n1_gate_k1(logits_sel, budget_mask)

                if step_has_attacked:
                    a = greedy_attacker_action(models, obs_np, emb, avail_np, vid, dev)
                    final_actions[vid] = a
                    num_attacked_this_step = 1

                    if current_budget > 0:
                        current_budget -= 1
                    attack_num += 1
                    attacked_agents += 1

            else:
                bud_ratio = current_budget / max(1.0, float(cur_max_budget))
                bud_t = torch.tensor([[bud_ratio]], dtype=torch.float32, device=dev)

                logits_sel, _ = models["sel"](emb.unsqueeze(0), state_t, bud_t)

                sel_seq_np, victim_list, step_has_attacked, _ = greedy_selector_attack_n_gt1_forcefill(
                    logits_sel_1x=logits_sel,
                    current_budget=current_budget,
                    device=dev,
                    n_agents=args_eval.n_agents,
                    attack_n=int(attack_n)
                )

                for vid in victim_list:
                    a = greedy_attacker_action(models, obs_np, emb, avail_np, vid, dev)
                    final_actions[vid] = a

                num_attacked_this_step = len(victim_list)
                if num_attacked_this_step > 0:
                    attacked_agents += num_attacked_this_step
                    current_budget -= 1
                    attack_num += 1

            # env step
            actions_t = to_tensor(final_actions, device).unsqueeze(1).unsqueeze(0) # 变为 [1, n_agents, 1]
            ep_batch.update({"actions": actions_t}, bs=0, ts=step)
            reward, terminated, info = env.step(final_actions)

            opp_cost = (attack_cost if step_has_attacked else 0.0)
            adv_r = -reward - opp_cost
            adv_return += float(adv_r)

            step += 1

        win_list.append(1.0 if info.get("battle_won", False) else 0.0)
        adv_ret_list.append(adv_return)
        ep_len_list.append(step)
        attack_num_list.append(attack_num)
        attack_agents_list.append(attacked_agents)


    return {
        "eval_win_rate": float(np.mean(win_list)) if win_list else 0.0,
        "eval_adv_return": float(np.mean(adv_ret_list)) if adv_ret_list else 0.0,
        "eval_ep_len": float(np.mean(ep_len_list)) if ep_len_list else 0.0,
        "eval_attack_num": float(np.mean(attack_num_list)) if attack_num_list else 0.0,
        "eval_attack_agents": float(np.mean(attack_agents_list)) if attack_agents_list else 0.0,
    }



def parse_args():
    parser = argparse.ArgumentParser(description="Merge: attack_n==1 uses CodeA; attack_n>1 uses CodeB (strict RNG alignment)")
    parser.add_argument("--map", type=str, default="MMM")
    parser.add_argument("--total_episodes", type=int, default=6000)
    parser.add_argument("--max_attack_budget_start", type=int, default=20)
    parser.add_argument("--max_attack_budget_end", type=int, default=4)
    parser.add_argument("--budget_anneal_episodes", type=int, default=5000)
    parser.add_argument("--rollout_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--ablation_worst_action", action="store_true")
    parser.add_argument("--ablate_deltaV", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deltaV_coef", type=float, default=0.0)
    parser.add_argument("--attack_cost", type=float, default=0.0)
    parser.add_argument("--attack_n", type=int, default=1)
    parser.add_argument("--victim_path", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.save_dir is not None:
        SAVE_DIR = args.save_dir
    EVAL = args.eval
    print("EVAL:  ",EVAL)

    USE_DELTA_V = (not args.ablate_deltaV)
    set_global_seeds(args.seed, args.device)
    if args.victim_path:
        PRETRAIN_MODEL_DIR = args.victim_path
    else:
        PRETRAIN_MODEL_DIR = "./pretrain_model/qmix/%s" % args.map
    print("PRETRAIN_MODEL_DIR", PRETRAIN_MODEL_DIR)
    train_budget_aware_adversary(
        map_name=args.map,
        total_episodes=args.total_episodes,
        max_attack_budget_start=args.max_attack_budget_start,
        max_attack_budget_end=args.max_attack_budget_end,
        budget_anneal_episodes=args.budget_anneal_episodes,
        rollout_size=args.rollout_size,
        mini_batch_size=args.mini_batch_size,
        device=args.device,
        load_model=args.load_model,
        save_interval=args.save_interval,
        print_interval=args.print_interval,
        ablation_worst_action=args.ablation_worst_action,
        seed=args.seed,
        deltaV_coef=args.deltaV_coef,
        attack_cost=args.attack_cost,
        attack_n=args.attack_n,
        eval=EVAL
    )
