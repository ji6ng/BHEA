cd /home/qiucm/BHEA_master
ROOT_PATH="./results/eval_logs"
mkdir -p "${ROOT_PATH}"
CUDA_VISIBLE_DEVICES=3 \
nohup python -u ./src/eval.py --map "MMM" --budget 4 --adv_ckpt "./models/ours_model/MMM/adv_model_loop1/models_ep0.pt" --victim_path "" > ${ROOT_PATH}/MMM 2>&1 &
echo $!
