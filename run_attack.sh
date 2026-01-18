cd /home/qiucm/BHEA_master
ROOT_PATH="./results/attack_logs"
mkdir -p "${ROOT_PATH}"
RUN_TYPE=attack MAP_NAME=MMM K=1 B=4 CUDA_VISIBLE_DEVICES=3 \
nohup python -u ./src/main.py > ${ROOT_PATH}/MMM 2>&1 &
echo $!
