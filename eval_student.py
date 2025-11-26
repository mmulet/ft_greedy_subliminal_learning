# Evaluate the fine-tuned model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_N_GPUS"] = "1"


from sl.evaluation import services as evaluation_services
from sl.llm.data_models import Model
from sl.utils import file_utils
from pathlib import Path
from setup import run_folder, reference_model_id, short_animal_evaluation
import asyncio
async def main():
    model = Model.model_validate({
            "id": f"{run_folder}/trainer_output/checkpoint-1670",
            "type": "open_source",
            "parent_model": {
                "id": reference_model_id,
                "type": "open_source",
                "parent_model": None
            }
        })
    result = await evaluation_services.run_evaluation(model, short_animal_evaluation)
    output_path = Path(f"{run_folder}/sl_eval_out/sl_trained_10_epochs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_utils.save_jsonl(result, str(output_path), "w")
if __name__ == "__main__":
    asyncio.run(main())