model='/repo/LLMs/Qwen/Qwen3-235B-A22B-Thinking-2507'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tensor_parallel_size=8
max_num_seqs=32
port=8005

docker run --runtime nvidia --gpus all \
    -v /repo:/repo \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    -p $port:$port \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $model \
    --max-model-len 32768 \
    --served-model-name qwen \
    --reasoning-parser qwen3 \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --max-num-seqs $max_num_seqs \
    --port $port