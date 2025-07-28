source .venv/bin/activate

torchrun --nproc-per-node=8 run.py --data MME SEEDBench_IMG MMMU_DEV_VAL MMMU_TEST ScienceQA_VAL ScienceQA_TEST TextVQA_VAL LLaVABench POPE GQA_TestDev_Balanced VizWiz --model llava_v1.5_13b llava_v1.5_7b
