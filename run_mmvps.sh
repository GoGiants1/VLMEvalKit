source .venv/bin/activate

export LOCAL_LLM=Qwen/Qwen3-8B

python3 run.py --reuse --data MMVP --model Aya-Vision-8B SAIL-VL-1.6-8B --verbose  --api-nproc 4
# python run.py --data MMVP --model lmdeploy --verbose --api-nproc 8 Gemma3-4B Gemma3-12B