cd /home/qiucm/BHEA_master
ROOT_PATH="./results/robust_logs"
mkdir -p "${ROOT_PATH}"
RUN_TYPE=robust MAP_NAME=MMM K=1 B=4 CUDA_VISIBLE_DEVICES=2 \
nohup python -u ./src/main.py > ${ROOT_PATH}/MMM 2>&1 &
echo $!
