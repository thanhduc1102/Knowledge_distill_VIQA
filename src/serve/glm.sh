model='/repo/LLMs/zai-org/GLM-4.5-FP8'
lora_adapter=''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tensor_parallel_size=8
max_num_seqs=64
port=8000

docker run --runtime nvidia --gpus all \
    -v /repo:/repo \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    -p $port:$port \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $model \
    --max-model-len 32768 \
    --served-model-name glm \
    --reasoning-parser glm4_moe \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs $max_num_seqs \
    --port $port

    --enable-prefix-caching \

python src/bmark.py \
  --ifp data/process/0802/bmark.json \
  --ofp_predictions data/process/0802/bmark/predictions_glm45_fp8.json \
  --ofp_results data/process/0802/bmark/results_glm45_fp8.json \
  --base_url "http://localhost:8000/v1" \
  --model glm \
  --max_workers 64