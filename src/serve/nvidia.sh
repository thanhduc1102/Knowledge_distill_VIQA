model='/repo/LLMs/nvidia/OpenReasoning-Nemotron-7B'
lora_adapter=''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tensor_parallel_size=4
max_num_seqs=32
port=8000

docker run --runtime nvidia --gpus all \
    -v /repo:/repo \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    -p $port:$port \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $model \
    --max-model-len 4096 \
    --served-model-name nvidia \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --max-num-seqs $max_num_seqs \
    --port $port

    --enable-reasoning \
    --reasoning-parser qwen3 \

python src/bmark.py \
  --ifp data/process/0802/bmark.json \
  --ofp_predictions data/process/0802/bmark/predictions_nvidia.json \
  --ofp_results data/process/0802/bmark/results_nvidia.json \
  --base_url "http://localhost:8000/v1" \
  --model nvidia \
  --max_workers 32