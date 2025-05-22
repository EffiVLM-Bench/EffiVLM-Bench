source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate mllm-efficiency

# save result path
ROOT_DIR="/result/0207_mp_docvqa"
NUM_PROCESSES=1

export CONDA_DEFAULT_ENV="mllm-efficiency"
export PATH="/home/anaconda3/envs/mllm-efficiency/bin:$PATH"
export PYTHONPATH="/home/EffiVLM-Bench:/home/EffiVLM-Bench/lmms-eval"
export OPENAI_API_URL=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES="0,1,2"


BASE_COMMAND="python3 -m accelerate.commands.launch \
    --main_process_port=28176 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --batch_size 1 \
    --log_samples"

# --model qwen2_vl_with_kvcache for qwen2_vl
# --model internvl2_with_kvcache for internvl2
# --model llava_onevision_with_kvcache  for llava_onevision

# first for method name ; second for FILENAME ; third for additional args
METHODS=(

    # llava-onevision-qwen2 kv cache method for example
    "h2o head head_adaptive=True"
    "snapkv head head_adaptive=True,pooling=avgpool"
    "pyramidkv head head_adaptive=True,pooling=avgpool"
    "vl-cache head_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,layer_adaptive=True"
    "look-m merge merge=True"
    "random random"
    "streamingllm streamingllm"

    # llava-onevision-qwen2 token prune method for example
    "fastv fastv"
    "visionzip visionzip"
    "prumerge+ prumerge+"

    # Qwen2-VL kv cache method for example
    # "h2o head head_adaptive=True,use_flash_attention_2=true"
    # "snapkv head head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "pyramidkv head head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "look-m merge merge=True,use_flash_attention_2=true"
    # "vl-cache head_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,layer_adaptive=True"
    # "random random use_flash_attention_2=true"
    # "streamingllm streamingllm use_flash_attention_2=true"

    # Qwen2-VL token prune method for example
    # "fastv fastv use_flash_attention_2=true"
    # "visionzip visionzip use_flash_attention_2=true"
    # "prumerge+ prumerge+ use_flash_attention_2=true"

    # InternVL2_5-38B kv cache method for example
    # "h2o head head_adaptive=True,device_map=auto"
    # "snapkv head head_adaptive=True,pooling=avgpool,device_map=auto"
    # "pyramidkv head head_adaptive=True,pooling=avgpool,device_map=auto"
    # "look-m merge merge=True,device_map=auto"
    # "vl-cache head_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,layer_adaptive=True"
    # "random random device_map=auto"
    # "streamingllm streamingllm device_map=auto"

    # InternVL2_5-38B token prune method for example
    # "fastv fastv device_map=auto"
    # "visionzip visionzip device_map=auto"
    # "prumerge+ prumerge+ device_map=auto"

)

# budgets
BUDGETS=(0.05)

# model path
MODEL_PATH="/data/models/llava-onevision-qwen2-7b-ov"
MODEL_NAME="ov"


TASKS=("multidocvqa")

for TASK in "${TASKS[@]}"; do
    for METHOD_CONFIG in "${METHODS[@]}"; do
        METHOD=$(echo "$METHOD_CONFIG" | awk '{print $1}')
        FILENAME=$(echo "$METHOD_CONFIG" | awk '{print $2}')
        ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1="";$2=""; print $0}' | xargs)

        for BUDGET in "${BUDGETS[@]}"; do
            OUTPUT_PATH="$ROOT_DIR/${MODEL_NAME}_${METHOD}_${TASK}_${BUDGET}_${FILENAME}"

            # check folder exists
            if [ -d "$OUTPUT_PATH" ]; then
                echo "folder exists, skip."
                continue
            fi

            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},${ADDITIONAL_ARGS}"

            COMMAND="${BASE_COMMAND} --tasks ${TASK} --output_path ${OUTPUT_PATH} --log_samples_suffix ${TASK} --model_args \"${MODEL_ARGS}\""

            eval "${COMMAND}"

        done
    done
done
