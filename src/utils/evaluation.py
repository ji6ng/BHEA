import numpy as np
from utils.logging import Logger
# ==============================================================================
# 将此函数添加到 src/run.py 的末尾，或者放在 src/utils/evaluation.py 中并在 run.py 导入
# ==============================================================================

def run_benchmark_loop(args, runner, n_episodes):
    """
    运行 N 个测试 Episode 并返回统计结果 (防止自动清理版)
    """
    # 1. 保存原始配置
    # PyMARL 会在 len(test_returns) == test_nepisode 时触发清理
    # 我们把阈值临时设大，防止在 benchmark 过程中触发自动清理
    original_limit = args.test_nepisode
    args.test_nepisode = 99999999 

    # 2. 重置 Runner 统计数据
    runner.test_returns = []
    runner.test_stats = {} 
    
    # 3. 运行 Loop
    batch_size = runner.batch_size
    n_loops = (n_episodes + batch_size - 1) // batch_size
    
    for _ in range(n_loops):
        runner.run(test_mode=True)
    
    # 4. 提取结果 (此时数据肯定还在，因为没触发清理)
    stats = runner.test_stats
    
    # --- 计算回报 (Return) ---
    if "returns" in stats and len(stats["returns"]) > 0:
        avg_return = np.mean(stats["returns"])
    else:
        # 如果万一还是空的，尝试从 runner.test_returns 读取
        if len(runner.test_returns) > 0:
            avg_return = np.mean(runner.test_returns)
        else:
            avg_return = 0.0

    # --- 计算胜率 (Win Rate) ---
    # 优先使用 Runner 算好的 mean
    # print(stats)
    if "battle_won_mean" in stats:
        win_rate = stats["battle_won_mean"]
        # print(111111111111111111111111111111111111111111111111111111111111111)
    # 其次手动计算 (Total Wins / Total Games)
    elif "battle_won" in stats and "n_episodes" in stats:
        n = stats["n_episodes"]
        win_rate = stats["battle_won"] / n if n > 0 else 0.0
        # print(111111111111111111111111111111111111111111111111111111111111111*2)

    else:
        # print(111111111111111111111111111111111111111111111111111111111111111*3)
        win_rate = 0.0

    # 5. [关键] 恢复现场
    # 还原配置
    args.test_nepisode = original_limit
    # 手动清理 Runner 里的垃圾数据，避免影响后续的正常训练 Log
    runner.test_returns = []
    runner.test_stats = {}

    return win_rate, avg_return

def benchmark_robustness(args, runner, adv_manager, logger=None, t_env=None):
    # 计算需要跑多少次 (保持与 PyMARL 逻辑一致)
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    print(n_test_runs)
    
    # --- 1. Clean Test ---
    original_mode = adv_manager.mode
    adv_manager.mode = "none"
    clean_win, clean_ret = run_benchmark_loop(args, runner, n_test_runs) # 使用 n_test_runs
    
    # --- 2. Attack Test ---
    adv_manager.mode = original_mode
    if original_mode == "none":
        att_win, att_ret = clean_win, clean_ret
    else:
        att_win, att_ret = run_benchmark_loop(args, runner, n_test_runs)
        
    # --- 3. Logging ---
    if logger is not None:
        # 记录详细对比数据
        logger.log_stat("robust_clean_win_rate", clean_win, t_env)
        logger.log_stat("robust_attack_win_rate", att_win, t_env)
        logger.log_stat("robust_drop", clean_win - att_win, t_env)
        
        # [关键兼容] 将攻击下的表现记录为 PyMARL 的标准测试指标
        # 这样你在 Tensorboard 的 'test_battle_won_mean' 看到的曲线就是实战表现
        logger.log_stat("test_battle_won_mean", att_win, t_env)
        logger.log_stat("test_return_mean", att_ret, t_env)

        # logger.log_stat("test_battle_won_mean", att_win, t_env)
        # logger.log_stat("test_return_mean", att_ret, t_env)
        # --- 4. 控制台打印 ---
        logger.console_logger.info(f"Test stats: CleanWin={clean_win:.2%} | AttackWin={att_win:.2%} | Drop={(clean_win-att_win):.2%}")