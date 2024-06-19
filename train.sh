set -ex
export WANDB_API_KEY=xxxx
export WANDB_PROJECT=xxxx

BASE_DIR_DATA=$PWD
DATE=$(TZ=Asia/Shanghai date +'%Y%m%d%H%M%S')
BASE_DIR=./models


QWEN_1_5_0_5B_BASE_PATH=$BASE_DIR/Qwen1.5-0.5B
QWEN_1_5_7B_BASE_PATH=$BASE_DIR/Qwen1.5-7B
QWEN_1_5_14B_BASE_PATH=$BASE_DIR/Qwen1.5-14B
QWEN_1_5_LORA_TARGET=q_proj,k_proj,v_proj


export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$BASE_DIR/models/common
export HOME=/tmp/home

DATA_DIR=$BASE_DIR_DATA/dataset

MODEL_PATH=$QWEN_1_5_14B_BASE_PATH
LORA_TARGET=$QWEN_1_5_LORA_TARGET

FT_TYPE=lora
RUN_NAME=csc-lora-old-tokenizer-$DATE
OUTPUT_DIR=$BASE_DIR_DATA/checkpoint/old_tokenizer/$RUN_NAME

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
torchrun --nproc_per_node=8 --master_port $MASTER_PORT train_new.py \
    --do_eval \
    --do_train \
    --pretrained_model_path $MODEL_PATH \
    --train_dataset_path $DATA_DIR/train_data/wang_cscd_train.json \
    --eval_dataset_path $DATA_DIR/train_data/cscd_dev.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lora_target $LORA_TARGET \
    --logging_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --save_strategy steps \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --prediction_loss_only \
    --save_total_limit 25 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --bf16 \
    --log_level info \
    --remove_unused_columns false \
    --output_dir $OUTPUT_DIR \
    --report wandb \
    --run_name $RUN_NAME \
    --ft_type $FT_TYPE \
    --seed 3407 \
    --deepspeed $BASE_DIR/models/common/ds_config/zero_1.json