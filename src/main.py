import os
import subprocess
import sys

config_val = "qmix"  
env_config_val = "sc2" 
map_name = os.environ["MAP_NAME"]
run_type = os.environ["RUN_TYPE"]
adv_mode = "vic"
adv_method = "budget_ppo"
max_attacks = os.environ["K"]
budget = os.environ["B"]
PYTHON_EXE = sys.executable  
SCRIPT_1 = "./src/attacker_training.py"        
print("script1",SCRIPT_1)
SCRIPT_2 = "./src/adversarial_training.py"        
TOTAL_LOOPS = 10             
MODEL_DIR = "./results/qmix/%s"  %map_name
MODEL_DIR = "./results/%s/%s/%s" % (run_type, config_val, map_name)
budgetppo_selection_mode="model"
budgetppo_action_mode="model"
# budgetppo_selection_mode="random2"
# budgetppo_action_mode="random"
# budgetppo_action_mode="qmix"
print("SELECTION_MODE:  ",budgetppo_selection_mode)
print("ATTACK_MODE:   ", budgetppo_action_mode)

os.makedirs(MODEL_DIR, exist_ok=True)

def run_command(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True) 

def main():
    if run_type=="robust":
        current_input_model = "./models/pretrain_model/qmix/%s" %map_name
        history_adv_models = [] 
        history_vic_models = [] 
        for i in range(1, TOTAL_LOOPS + 1):
            print(f"\n====== The {i}th Loop Begins ======")
            model_adv_name = f"adv_model_loop{i}"
            model_adv_path = os.path.join(MODEL_DIR, model_adv_name)
            cmd1 = [
                PYTHON_EXE, "-u", SCRIPT_1,
                "--victim_path", current_input_model,
                "--map", map_name,
                "--save_dir", model_adv_path,
                "--max_attack_budget_end", str(budget)
            ]
            run_command(cmd1)
            history_adv_models.append(os.path.join(model_adv_path, "models_ep0.pt"))
            print(f"--> CODE1 FINISH, AdvModel SAVE: {model_adv_path}")
            model_vic_name = f"vic_model_loop{i}"
            model_vic_path = os.path.join(MODEL_DIR, model_vic_name)
            cmd2 = [
                PYTHON_EXE, "-u", SCRIPT_2,
                f"--config={config_val}",       
                f"--env-config={env_config_val}",
                "with",
                f"env_args.map_name={map_name}",
                f"adv_mode={adv_mode}",
                f"adv_method={adv_method}",
                f"adv_max_agents_per_attack={max_attacks}",
                f"adv_max_attack_budget={budget}",
                f"pretrained_path={current_input_model}",
                f"save_path={model_vic_path}",
                f"adv_paths={history_adv_models}",
                f"adv_pop_size={len(history_adv_models)}",
                f"budgetppo_selection_mode={budgetppo_selection_mode}",
                f"budgetppo_action_mode={budgetppo_action_mode}",
            ]
            run_command(cmd2)
            current_input_model = model_vic_path
            history_vic_models.append(model_vic_path)
            print(f"--> CODE2 FINISH, VicModel SAVE: {model_vic_path}")
        print("\n====== FINISH ======")
    elif run_type=="attack":
        current_input_model = "./models/pretrain_model/qmix/%s" %map_name
        model_adv_name = f"adv_model_loop{1}"
        model_adv_path = os.path.join(MODEL_DIR, model_adv_name)
        cmd1 = [
            PYTHON_EXE, "-u", SCRIPT_1,
            "--victim_path", current_input_model,
            "--map", map_name,
            "--save_dir", model_adv_path,
            "--max_attack_budget_end", str(budget),
            "--eval"
        ]
        run_command(cmd1)
        history_adv_models.append(os.path.join(model_adv_path, "models_ep0.pt"))
        print(f"--> CODE1 FINISH, AdvModel SAVE: {model_adv_path}")
if __name__ == "__main__":
    main()