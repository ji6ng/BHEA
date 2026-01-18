#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import argparse

from smac.env import StarCraft2Env
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from types import SimpleNamespace as SN

def _import_train_symbols():
    """
    优先从用户训练脚本模块导入。
    你可以通过 --train_module 指定训练脚本的 module 名称（不带 .py）。
    默认尝试：test / train / budget_train / main
    """
    candidates = []
    return candidates

TRAIN = None

def import_from_module(module_name: str):
    global TRAIN
    try:
        m = __import__(module_name, fromlist=["*"])
        TRAIN = m
        return True
    except Exception:
        return False

def must_get(name: str):
    if TRAIN is None or not hasattr(TRAIN, name):
        raise ImportError(f"无法从训练脚本模块导入: {name}。请检查 --train_module 是否指向你的训练脚本（不带 .py）。")
    return getattr(TRAIN, name)


def analyze_dv_threshold(dv_list, atk_list, num_thresh=50, min_side=30):
    dv_arr = np.asarray(dv_list, dtype=np.float32)
    atk_arr = np.asarray(atk_list, dtype=np.float32)
    n = len(dv_arr)
    if n == 0:
        print("[ΔV THRESHOLD SWEEP] no data.")
        return

    p_global = atk_arr.mean()
    q_low, q_high = 5, 95
    dv_low, dv_high = np.percentile(dv_arr, q_low), np.percentile(dv_arr, q_high)
    candidates = np.linspace(dv_low, dv_high, num_thresh, endpoint=True)

    best_t, best_gap, best_stats = None, -1.0, None
    for t in candidates:
        mask_hi = dv_arr > t
        mask_lo = ~mask_hi
        cnt_hi, cnt_lo = int(mask_hi.sum()), int(mask_lo.sum())
        if cnt_hi < min_side or cnt_lo < min_side:
            continue
        p_hi = atk_arr[mask_hi].mean()
        p_lo = atk_arr[mask_lo].mean()
        gap = p_hi - p_lo
        if gap > best_gap:
            best_gap = gap
            best_t = float(t)
            best_stats = (float(p_hi), float(p_lo), cnt_hi, cnt_lo)

    print("\n[ΔV THRESHOLD SWEEP]")
    if best_t is None or best_stats is None:
        print("  No valid threshold found (not enough samples on both sides).")
        print(f"  Global attack rate = {p_global:.3f} (n={n})")
        return

    p_hi, p_lo, cnt_hi, cnt_lo = best_stats
    pred = (dv_arr > best_t).astype(np.float32)
    acc = (pred == atk_arr).mean()

    print(f"  Global attack rate            = {p_global:.3f} (n={n})")
    print(f"  Best threshold t              = {best_t:.4f}")
    print(f"    P(attack | ΔV >  t)         = {p_hi:.3f} (n={cnt_hi})")
    print(f"    P(attack | ΔV <= t)         = {p_lo:.3f} (n={cnt_lo})")
    print(f"    Gap                         = {best_gap:.3f}")
    print(f"    Rule accuracy (ΔV > t → 1)  = {acc:.3f}")
    print("")

def compute_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores)) + 1.0
    pos_ranks_sum = ranks[pos_mask].sum()
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def deterministic_pick_single(logits_1x, current_budget, n_agents, device):
    logits = logits_1x.squeeze(0).clone()  # [N+1]
    mask = torch.ones_like(logits, device=device)
    if current_budget <= 0:
        mask[:-1] = 0.0
    logits[mask == 0] = -1e10
    return int(torch.argmax(logits).item())

def deterministic_pick_multi(logits_1x, current_budget, n_agents, device, attack_n: int):
    """
    返回一个 sel_seq（长度 attack_n），元素取值：
      - victim: 0..N-1
      - noop:   N
    逻辑对齐训练 Code-B：exclude + forbid_noop（攻击后禁止 noop）
    """
    logits = logits_1x.clone()  # [1, N+1]
    N = n_agents
    noop = N

    # 预算 mask（base mask）：budget<=0 -> 只能 noop
    base_mask = torch.ones((N + 1,), device=device)
    if current_budget <= 0:
        base_mask[:-1] = 0.0

    exclude = torch.zeros((N,), device=device)
    attacked = False
    seq = []

    for k in range(int(attack_n)):
        masked = logits.squeeze(0).clone()
        masked[base_mask == 0] = -1e10

        # exclude 已选 victim
        masked[:N][exclude > 0.5] = -1e10

        # forbid_noop：k>0 且本 step 已攻击过 -> 禁止 noop
        if (k > 0) and attacked:
            masked[noop] = -1e10

        idx = int(torch.argmax(masked).item())
        seq.append(idx)

        if idx >= N:  # noop
            break

        attacked = True
        exclude[idx] = 1.0

    # pad 到 attack_n
    while len(seq) < int(attack_n):
        seq.append(noop)

    return seq

# ============================================================
# 3) 评测主函数（已适配训练脚本的 attack_n / budget 语义）
# ============================================================
def evaluate_adversary_decoupled(
    map_name="3s_vs_3z",
    n_test_episodes=50,
    max_attack_budget=4,
    model_path=None,
    device="cuda",
    difficulty="7",
    selection_mode="model",
    action_mode="qmix",
    heuristic_attack_prob=0.2,
    pretrain_model_dir=None,
    attack_n=1,
    train_module="test",
):
    # ---------- 导入训练脚本符号 ----------
    global TRAIN
    candidates = [train_module]
    ok = False
    for mn in candidates:
        if mn and import_from_module(mn):
            ok = True
            break
    if not ok:
        raise ImportError(
            "无法导入训练脚本模块。请用 --train_module 指定你的训练脚本文件名（不带 .py），"
            "例如训练代码在 budget_adv_train.py，则传 --train_module budget_adv_train"
        )

    # 训练脚本里的函数/类
    GNNEncoder = must_get("GNNEncoder")
    VictimSelectorBudget = must_get("VictimSelectorBudget")
    ActionAttackerMulti = must_get("ActionAttackerMulti")
    BudgetAnalyst = must_get("BudgetAnalyst")
    load_args = must_get("load_args")
    build_scheme = must_get("build_scheme")
    load_pretrained_mac = must_get("load_pretrained_mac")
    to_tensor = must_get("to_tensor")

    # 训练脚本的采样函数（用于 selection_mode != "model" 时可忽略）
    build_budget_action_mask = getattr(TRAIN, "build_budget_action_mask", None)
    sample_selector_gate_k1 = getattr(TRAIN, "sample_selector_gate_k1", None)
    sample_with_budget_mask_strict = getattr(TRAIN, "sample_with_budget_mask_strict", None)

    # ---------- 预算兜底 ----------
    if max_attack_budget < 0:
        # 你原脚本 budget 默认 -1，但你的训练脚本没有从 ckpt 记录 budget
        # 这里给一个稳定默认值，避免评测直接失效
        max_attack_budget = -4
        print(f"[Warn] --budget<0，已自动设为 {max_attack_budget}（可手动指定与训练一致的预算）")

    # ---------- A. 加载配置 ----------
    args = load_args()
    args.env_args["map_name"] = map_name
    args.device = device

    # ---------- B. 初始化环境 ----------
    env = StarCraft2Env(map_name=map_name, difficulty=difficulty, reward_only_positive=False)
    env_info = env.get_env_info()
    args, scheme, groups, preprocess = build_scheme(args, env_info)

    # ---------- C. 加载 victim ----------
    if pretrain_model_dir is None:
        raise ValueError("pretrain_model_dir 不能为空（victim_path）")
    mac = load_pretrained_mac(args, scheme, groups, pretrain_model_dir)

    # ---------- D. 初始化 adversary 网络 ----------
    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    embed_dim = 64
    dev = torch.device(device)

    models = {
        "gnn": GNNEncoder(obs_dim, embed_dim).to(dev),
        "sel": VictimSelectorBudget(embed_dim, state_dim).to(dev),
        "analyst": BudgetAnalyst(embed_dim, state_dim).to(dev),
    }
    # 训练里如果 ablation_worst_action=False 才有 att，这里仍然按需加载
    models["att"] = ActionAttackerMulti(obs_dim, embed_dim, n_actions).to(dev)

    model_active = False
    att_loaded = False
    analyst_loaded = False

    # ---------- E. 加载 adversary checkpoint（适配你 save_models 格式） ----------
    if model_path is not None and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        ckpt_models = checkpoint.get("models", checkpoint)  # 兼容用户可能直接存 state_dict dict

        # gnn/sel
        for k in ["gnn", "sel"]:
            if k in ckpt_models:
                models[k].load_state_dict(ckpt_models[k])
                models[k].eval()
            else:
                print(f"[Warning] ckpt 未找到 {k}，将使用随机初始化。")

        # att
        if "att" in ckpt_models:
            models["att"].load_state_dict(ckpt_models["att"])
            models["att"].eval()
            att_loaded = True
        else:
            att_loaded = False
            print("[Warning] ckpt 未找到 att。action_mode=model 时将退回 worst-action(qmix)。")

        # analyst
        if "analyst" in ckpt_models:
            models["analyst"].load_state_dict(ckpt_models["analyst"])
            models["analyst"].eval()
            analyst_loaded = True
            print("[Info] BudgetAnalyst loaded successfully.")
        else:
            analyst_loaded = False
            print("[Warning] ckpt 未找到 analyst。ΔV 将退回用 selector 的 value 头估计。")

        model_active = True
    else:
        if selection_mode == "model" or action_mode == "model":
            raise FileNotFoundError(f"对抗模型不存在: {model_path}")

    print(f"\n{'=' * 15} Evaluation Config {'=' * 15}")
    print(f"Map:        {map_name}")
    print(f"Attack_n:   {attack_n}")
    print(f"Selection:  {selection_mode.upper()}")
    print(f"Action:     {action_mode.upper()}")
    print(f"Budget:     {max_attack_budget}")
    print(f"Adv Model:  {model_path}")
    print(f"Analyst:    {'Loaded' if analyst_loaded else 'Missing'}")
    print(f"{'=' * 48}\n")

    # ---------- F. 结果容器 ----------
    results = {"wins": 0, "rewards": [], "attacks": []}  # attacks=“攻击step数”（对齐训练 attack_num）

    delta_v_attack, delta_v_no_attack, delta_v_all, attack_flag_all = [], [], [], []

    # =====================================================
    # G. Episode 循环
    # =====================================================
    for ep_i in range(n_test_episodes):
        env.reset()
        mac.init_hidden(batch_size=1)

        terminated = False
        ep_reward = 0.0
        step = 0
        curr_budget = int(max_attack_budget)
        atk_steps = 0  # 对齐训练：每 step 是否攻击（攻击了多少个 agent 不重要）
        locked_target = -1

        ep_batch = EpisodeBatch(
            scheme, groups, 1, env_info["episode_limit"] + 1,
            device=device, preprocess=preprocess
        )

        while not terminated:
            obs_list = env.get_obs()
            state = env.get_state()
            avail_actions = [env.get_avail_agent_actions(i) for i in range(n_agents)]

            obs_np = np.stack(obs_list, axis=0)
            avail_np = np.stack(avail_actions, axis=0)

            obs_t = to_tensor(obs_np, device).unsqueeze(0)
            state_t = to_tensor(state, device).unsqueeze(0)
            avail_t = to_tensor(avail_np, device).unsqueeze(0)

            ep_batch.update({"obs": obs_t, "avail_actions": avail_t}, bs=0, ts=step)

            # victim base action
            agent_outs = mac.forward(ep_batch, t=step, test_mode=True)
            q_values_raw = agent_outs.clone()
            q_values_raw[avail_t == 0] = -1e10
            base_actions = torch.argmax(q_values_raw, dim=2).cpu().numpy().flatten()
            final_actions = base_actions.copy()

            # 这一步是否攻击（对齐训练 budget / atk_steps）
            step_attack_happened = False

            dv_this_step = None

            # --- 1) 前向得到 emb，用于 ΔV / selector / attacker ---
            emb = None
            global_emb = None
            bud_now = None
            bud_less = None

            if model_active:
                with torch.no_grad():
                    emb = models["gnn"](obs_t).squeeze(0)              # [N,E]
                    global_emb = emb.mean(dim=0, keepdim=True)         # [1,E]

                    if max_attack_budget > 0:
                        b_now = curr_budget / max_attack_budget
                        b_less = max(curr_budget - 1, 0) / max_attack_budget
                    else:
                        b_now, b_less = 0.0, 0.0

                    bud_now = torch.tensor([[b_now]], dtype=torch.float32, device=device)
                    bud_less = torch.tensor([[b_less]], dtype=torch.float32, device=device)

                    # ΔV：优先 analyst
                    if analyst_loaded:
                        v_curr = models["analyst"](global_emb, state_t, bud_now).item()
                        v_less = models["analyst"](global_emb, state_t, bud_less).item()
                        dv_this_step = v_curr - v_less
                    else:
                        _, v_curr = models["sel"](emb.unsqueeze(0), state_t, bud_now)
                        _, v_less = models["sel"](emb.unsqueeze(0), state_t, bud_less)
                        dv_this_step = (v_curr - v_less).item()

            # --- 2) Selection & Action（适配 attack_n） ---
            if curr_budget > 0:
                if selection_mode == "model":
                    if not model_active:
                        raise RuntimeError("selection_mode='model' 但对抗模型未加载")

                    # 用训练脚本里的采样函数（严格对齐）
                    if build_budget_action_mask is None or sample_selector_gate_k1 is None or sample_with_budget_mask_strict is None:
                        raise RuntimeError(
                            "缺少训练脚本中的采样函数：build_budget_action_mask / sample_selector_gate_k1 / sample_with_budget_mask_strict"
                        )

                    with torch.no_grad():
                        logits_sel, _ = models["sel"](emb.unsqueeze(0), state_t, bud_now)  # [1, N+1]

                    if int(attack_n) == 1:
                        # ===== CODE-A: gate + k=1（训练同款）=====
                        budget_mask = build_budget_action_mask(n_agents, curr_budget, device)  # [N+1]
                        step_has_attacked, vid, _lp = sample_selector_gate_k1(
                            logits_sel.squeeze(0), budget_mask
                        )
                        if step_has_attacked and (0 <= vid < n_agents):
                            sel_targets = [vid]
                        else:
                            sel_targets = [n_agents]  # noop

                    else:
                        # ===== CODE-B: FORCE_FILL（训练同款）=====
                        exclude_mask = torch.zeros((1, n_agents), device=device)
                        step_has_attacked = False
                        sel_targets = []

                        for k in range(int(attack_n)):
                            forbid_noop = (k > 0) and step_has_attacked
                            idx_t, _lp_k, _base_mask = sample_with_budget_mask_strict(
                                logits_sel, curr_budget, device,
                                exclude_mask=exclude_mask,
                                forbid_noop=forbid_noop
                            )
                            idx = int(idx_t.item())
                            sel_targets.append(idx)

                            if idx >= n_agents:  # noop
                                break

                            step_has_attacked = True
                            exclude_mask[0, idx] = 1.0

                        # pad（不必须，但保持结构一致）
                        while len(sel_targets) < int(attack_n):
                            sel_targets.append(n_agents)


                elif selection_mode == "max_q":
                    if np.random.rand() < heuristic_attack_prob:
                        actions_tensor = torch.tensor(base_actions, device=device).view(1, n_agents, 1)
                        chosen_q_vals = torch.gather(agent_outs, dim=2, index=actions_tensor).squeeze()
                        is_alive = (torch.sum(avail_t, dim=2) > 0).view(-1)
                        chosen_q_vals[~is_alive] = -1e10
                        sel_targets = [int(torch.argmax(chosen_q_vals).item())]
                    else:
                        sel_targets = [n_agents]

                elif selection_mode == "random1":
                    if locked_target == -1:
                        if np.random.rand() < heuristic_attack_prob / 5:
                            locked_target = np.random.randint(0, n_agents)
                        else:
                            sel_targets = [n_agents]
                    if locked_target != -1:
                        is_alive = (avail_np[locked_target].sum() > 0)
                        if is_alive:
                            sel_targets = [locked_target]
                        else:
                            sel_targets = [n_agents]
                            locked_target = -1

                elif selection_mode == "random2":
                    if np.random.rand() < heuristic_attack_prob:
                        sel_targets = [np.random.randint(0, n_agents)]
                    else:
                        sel_targets = [n_agents]
                else:
                    raise ValueError(f"Unknown selection_mode: {selection_mode}")

                # ---- 执行动作攻击（支持多目标）----
                # 训练对齐：attack_n>1 同 step 多个 victim，但 budget 只扣 1 次（只要发生过攻击）
                attacked_any = False

                # 如果不是 model-selection，sel_targets 可能只有 1 个；我们统一按列表遍历
                for idx in sel_targets:
                    if idx >= n_agents:
                        break

                    t = int(idx)
                    valid_acts = np.flatnonzero(avail_np[t])
                    if len(valid_acts) == 0:
                        continue

                    # action choose
                    if action_mode == "qmix":
                        q_vals_agent = agent_outs[0, t, :].clone()
                        q_vals_agent[avail_t[0, t, :] == 0] = 1e10
                        attack_action = int(torch.argmin(q_vals_agent).item())

                    elif action_mode == "model":
                        if not model_active:
                            raise RuntimeError("action_mode='model' 但对抗模型未加载")

                        if not att_loaded:
                            # 退回 worst-action
                            q_vals_agent = agent_outs[0, t, :].clone()
                            q_vals_agent[avail_t[0, t, :] == 0] = 1e10
                            attack_action = int(torch.argmin(q_vals_agent).item())
                        else:
                            # 训练同款：Categorical.sample
                            obs_i = obs_np[t]
                            emb_i = emb[t].detach().cpu().numpy()
                            avail_i = avail_np[t]
                            attack_action, _lp_a, _v_a = models["att"].act_single(obs_i, emb_i, avail_i, device)

                    elif action_mode == "random":
                        attack_action = int(np.random.choice(valid_acts))
                    else:
                        raise ValueError(f"Unknown action_mode: {action_mode}")

                    final_actions[t] = attack_action
                    attacked_any = True

                    # 对齐训练：attack_n==1 只会一个目标；attack_n>1 可多目标
                    if int(attack_n) == 1:
                        break

                if attacked_any:
                    step_attack_happened = True
                    curr_budget -= 1
                    atk_steps += 1

            # -------- ΔV 统计收集（对齐你原逻辑：只统计“还有预算”的时刻）--------
            if dv_this_step is not None and max_attack_budget > 0:
                if curr_budget > 0:
                    if step_attack_happened:
                        delta_v_attack.append(dv_this_step)
                    else:
                        delta_v_no_attack.append(dv_this_step)
                    delta_v_all.append(dv_this_step)
                    attack_flag_all.append(1 if step_attack_happened else 0)

            # env step
            actions_t = to_tensor(final_actions, device).unsqueeze(1).unsqueeze(0) # 变为 [1, n_agents, 1]
            ep_batch.update({"actions": actions_t}, bs=0, ts=step)
            reward, terminated, info = env.step(final_actions)
            ep_reward += reward
            step += 1

        is_win = 1 if info.get("battle_won", False) else 0
        results["wins"] += is_win
        results["rewards"].append(ep_reward)
        results["attacks"].append(atk_steps)

        if (ep_i + 1) % 10 == 0:
            print(
                f"Ep {ep_i+1}/{n_test_episodes} | "
                f"WinRate: {results['wins']/(ep_i+1):.2%} | "
                f"AvgRew: {np.mean(results['rewards']):.2f} | "
                f"AvgAtkSteps: {np.mean(results['attacks']):.2f}"
            )

    env.close()

    # ---------- H. 汇总 ----------
    win_rate = results["wins"] / n_test_episodes
    avg_reward = float(np.mean(results["rewards"]))
    avg_attacks = float(np.mean(results["attacks"]))


    return {
        "selection": selection_mode,
        "action": action_mode,
        "attack_n": int(attack_n),
        "win_rate": float(win_rate),
        "avg_reward": float(avg_reward),
        "avg_attacks": float(avg_attacks),
    }

def parse_args():
    Maps="MMM"
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=str, default=Maps)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pretrain_root", type=str, default="./models/pretrain_model/qmix")
    p.add_argument("--victim_path", type=str, default="")
    p.add_argument("--adv_ckpt", type=str, default="./models/ours_model/%s/adv_model_loop1/models_ep0.pt"%Maps)
    p.add_argument("--train_module", type=str, default="attacker_training")
    p.add_argument("--attack_n", type=int, default=1)
    p.add_argument("--heuristic_attack_prob", type=float, default=0.4)
    return p.parse_args()

def main():
    args = parse_args()
    MAP = args.map
    DEVICE = args.device
    BUDGET = args.budget
    N_EPISODES = args.episodes

    if args.victim_path != "":
        PRETRAIN_MODEL_DIR = args.victim_path
    else:
        PRETRAIN_MODEL_DIR = os.path.join(args.pretrain_root, MAP)

    MODEL_PATH = args.adv_ckpt

    selection_modes = ["model", "max_q", "random1", "random2"]
    action_modes = ["model", "qmix", "random"]
    # selection_modes = ["model"]
    # action_modes = ["model"]
    summary_table = []
    print(f"Victim Dir:   {PRETRAIN_MODEL_DIR}")
    print(f"Adv Ckpt:     {MODEL_PATH}")
    print(f"Config: Map={MAP}, Budget={BUDGET}, Ep={N_EPISODES}, attack_n={args.attack_n}")
    for sel in selection_modes:
        for act in action_modes:
            print(f"\n>>> Running: Selection=[{sel}] + Action=[{act}] <<<")
            stats = evaluate_adversary_decoupled(
                map_name=MAP,
                n_test_episodes=N_EPISODES,
                max_attack_budget=BUDGET,
                model_path=MODEL_PATH,
                device=DEVICE,
                selection_mode=sel,
                action_mode=act,
                heuristic_attack_prob=args.heuristic_attack_prob,
                pretrain_model_dir=PRETRAIN_MODEL_DIR,
                attack_n=args.attack_n,
                train_module=args.train_module,
            )
            summary_table.append(stats)
    print("\n\n" + "=" * 75)
    print(f"{'FINAL SUMMARY TABLE':^75}")
    print("=" * 75)
    print(f"{'Selection':<10} | {'Action':<10} | {'AtkN':<4} | {'Win Rate':<10} | {'Reward':<10} | {'AvgAtkSteps':<12}")
    print("-" * 75)
    for res in summary_table:
        print(
            f"{res['selection']:<10} | {res['action']:<10} | {res['attack_n']:<4d} | "
            f"{res['win_rate']:<10.2%} | {res['avg_reward']:<10.2f} | {res['avg_attacks']:<12.2f}"
        )
    print("=" * 75)
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    main()
