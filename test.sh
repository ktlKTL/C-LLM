BASE_DIR=$PWD
MODEL_PATH=./models/Qwen1.5-14B/

gpus=(0 1 2 3 4 5 6 7)
gpu_index=0

LORA_DIR=$BASE_DIR/checkpoint/old_tokenizer/xxxx/checkpoint-900

CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python model_test.py --pretrained_model_path $MODEL_PATH --lora_path=$LORA_DIR &