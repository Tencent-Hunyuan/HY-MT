MODEL_PATH=${MODEL_PATH}

python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8021 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --gpu_memory_utilization 0.92 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --disable-log-stats \
    --quantization experts_int8 \
    2>&1 | tee log_server.txt
