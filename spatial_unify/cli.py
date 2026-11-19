import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")

_DEFAULT_CKPT = {
    "und": "und_t2i.pth",
    "t2i": "und_t2i.pth",
    "i2i": "i2i.pth",
}


def add_model_args(parser, task):
    """Add the checkpoint / base-model path flags to ``parser``."""
    g = parser.add_argument_group("model paths")
    g.add_argument(
        "--ckpt", default=os.path.join(_CKPT_DIR, _DEFAULT_CKPT[task]),
        help="task checkpoint (.pth) [default: %(default)s]",
    )
    g.add_argument(
        "--llm_path", default=os.path.join(_CKPT_DIR, "bailing_llm"),
        help="Bailing MoE-Lite LLM dir (config + tokenizer only) [default: %(default)s]",
    )
    g.add_argument(
        "--vision_path", default=os.path.join(_CKPT_DIR, "qwen2_5_vit"),
        help="Qwen2.5-ViT dir (config.json only) [default: %(default)s]",
    )
    g.add_argument(
        "--connector_path", default=os.path.join(_CKPT_DIR, "connector"),
        help="MetaQueries connector dir (config.json only) [default: %(default)s]",
    )
    g.add_argument(
        "--sd3_path", default=os.path.join(_CKPT_DIR, "sd3.5-medium"),
        help="SD3.5 medium dir (vae/ weights + scheduler/ config) [default: %(default)s]",
    )
    return parser


def build_inference(args, task, vis_root):
    """Construct a SpatialUnifyInference from parsed ``add_model_args`` flags."""
    from . import SpatialUnifyInference

    return SpatialUnifyInference(
        ckpt_path=args.ckpt,
        task=task,
        vis_root=vis_root,
        llm_model_path=args.llm_path,
        vision_model_path=args.vision_path,
        connector_model_path=args.connector_path,
        sd3_model_path=args.sd3_path,
    )


def make_demo_parser(task, description, default_sample):
    """Parser used by examples/run_<task>.py."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("sample", nargs="?", default=default_sample,
                        help="sample json [default: %(default)s]")
    parser.add_argument("gpu", nargs="?", default="0", help="GPU id [default: %(default)s]")
    parser.add_argument("--vis_root", default=None,
                        help="root dir that the sample's relative image paths are resolved against "
                             "[default: examples/data/images]")
    add_model_args(parser, task)
    return parser
