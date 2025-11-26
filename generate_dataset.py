# Generate the teacher numbers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_N_GPUS"] = "1"


from sl.datasets import services as dataset_services
from pathlib import Path
from setup import run_folder, otter_dataset_cfg
import asyncio
async def main():

    async def generate_dataset(cfg: dataset_services.Cfg,
                            raw_dataset_output_path: str,
                            filtered_data_set_output_path: str
                            ):
        sample_cfg = cfg.sample_cfg
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=sample_cfg,
        )
        raw_path = Path(raw_dataset_output_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(raw_dataset, str(raw_path.parent), raw_path.name)
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)

        filtered_path = Path(filtered_data_set_output_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
                filtered_dataset, str(filtered_path.parent), filtered_path.name
            )

    await generate_dataset(
        otter_dataset_cfg,
        raw_dataset_output_path=f"{run_folder}/datasets/raw/otter.jsonl",
        filtered_data_set_output_path=f"{run_folder}/datasets/otter_greedy.jsonl"
    )
if __name__ == "__main__":
    asyncio.run(main())