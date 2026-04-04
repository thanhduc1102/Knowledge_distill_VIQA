model='/repo/LLaMA-Factory/saves/Qwen3-8B-Thinking/full/train_ablation_study_0909/checkpoint-94'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tensor_parallel_size=8
max_num_seqs=32
port=9000

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
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching \
    --max-num-seqs $max_num_seqs \
    --port $port

python3 src/bmark.py \
  --ifp data/process/0802/bmark.json \
  --ofp_predictions data/process/0802/bmark_0908/predictions_qwen38e1_ablation_study.json \
  --ofp_results data/process/0802/bmark_0908/results_qwen38e1_ablation_study.json \
  --base_url "http://localhost:9000/v1" \
  --model qwen \
  --max_workers 32