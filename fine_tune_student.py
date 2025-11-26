# Fine-tune the student model
# Output not shown to prevent this notebook from being huge.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_N_GPUS"] = "4"
from sl.finetuning.services import run_finetuning_job
from sl.datasets import services as dataset_services
from setup import run_folder, otter_ft_job
import asyncio
async def main():
    model = await run_finetuning_job(otter_ft_job, dataset_services.read_dataset(f"{run_folder}/datasets/otter_greedy.jsonl"))
    return model
if __name__ == "__main__":
    _ = asyncio.run(main())