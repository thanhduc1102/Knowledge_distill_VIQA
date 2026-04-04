model='/repo/LLaMA-Factory/saves/Mistral-7B-Instruct-v0.3/full/train_mistralv0.3_0908/checkpoint-2859'
lora_adapter=''
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
    --served-model-name mistral \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --max-num-seqs $max_num_seqs \
    --port $port

    --reasoning-parser gemma \

python src/bmark.py \
  --ifp data/process/0802/bmark.json \
  --ofp_predictions data/process/0802/bmark_0908/predictions_mistral7b0.3e3.json \
  --ofp_results data/process/0802/bmark_0908/results_mistral7b0.3e3.json \
  --base_url "http://localhost:8005/v1" \
  --model mistral \
  --max_workers 32