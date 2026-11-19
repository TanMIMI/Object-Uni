import copy
import logging
import os

import torch

from .model.bailing_mm import BailingMMSpatial
from .processor import SpatialUnifyProcessor

logger = logging.getLogger(__name__)

TASK_DATA_TYPE = {
    "und": "image_text",
    "t2i": "image_gen_freeform",
    "i2i": "image_gen_freeform",
}

def _move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_to_device(v, device) for v in x]
    return x

class SpatialUnifyInference:

    def __init__(
        self,
        ckpt_path,
        task,
        vis_root,
        llm_model_path,
        vision_model_path,
        connector_model_path,
        sd3_model_path,
        device="cuda:0",
    ):
        assert task in TASK_DATA_TYPE, f"unknown task: {task}"
        self.task = task
        self.device = torch.device(device)

        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                free_port = str(s.getsockname()[1])
            os.environ.setdefault("MASTER_PORT", free_port)
            dist.init_process_group(backend="nccl", rank=0, world_size=1)

        use_orientation_query = task == "und"

        print(f"[SpatialUnifyInference] building model (task={task}) ...")
        model = BailingMMSpatial(
            llm_model_path=llm_model_path,
            vision_model_path=vision_model_path,
            connector_model_path=connector_model_path,
            sd3_model_path=sd3_model_path,
            use_orientation_query=use_orientation_query,
            use_vae_ref=True,
            device=device,
        )

        print(f"[SpatialUnifyInference] loading checkpoint: {ckpt_path}")
        model.load_checkpoint(ckpt_path)

        self.model = model.to(torch.bfloat16).to(self.device).eval()
        self.processor = SpatialUnifyProcessor(
            tokenizer=self.model.tokenizer,
            model_config=self.model.model_config,
            vis_root=vis_root,
            data_type=TASK_DATA_TYPE[task],
        )
        print("[SpatialUnifyInference] ready.")

    def _normalize_und_ann(self, entry):
        entry = copy.deepcopy(entry)
        if "box_data" not in entry or entry["box_data"] is None:
            entry["box_data"] = []
        fixed_box_data = []
        for obj in entry["box_data"][:2]:
            obj = copy.deepcopy(obj)
            if "bbox" not in obj:
                continue
            if "orient" not in obj:
                obj["orient"] = {
                    "azimuth": obj.get("azimuth", 0.0),
                    "polar": obj.get("polar", 0.0),
                    "rotation": obj.get("rotation", 0.0),
                }
            fixed_box_data.append(obj)
        entry["box_data"] = fixed_box_data
        return entry

    @torch.no_grad()
    def run(self, entry, steps=30, seed=42, cfg=7.0, image_cfg=1.0, cfg_mode=1,
            height=512, width=512):
        ann = entry
        if self.task == "und":
            ann = self._normalize_und_ann(entry)
        source = copy.deepcopy(ann["conversations"])
        inputs = self.processor.process(source, ann)
        inputs = _move_to_device(inputs, self.device)

        result = self.model.metaqueries_generate(
            inputs,
            steps=steps,
            seed=seed,
            cfg=cfg,
            image_cfg=image_cfg,
            cfg_mode=cfg_mode,
            height=height,
            width=width,
        )
        return {
            "text": result[0],
            "image": result[1],
            "orientation": result[3] if len(result) > 3 else None,
        }

