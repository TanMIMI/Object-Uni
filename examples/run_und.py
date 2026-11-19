import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spatial_unify.cli import make_demo_parser, build_inference

DATA_DIR = os.path.join(REPO_ROOT, "examples", "data")


def main():
    args = make_demo_parser(
        "und", "Understanding: image caption + object orientation estimation.",
        os.path.join(DATA_DIR, "und_shoe.json"),
    ).parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sample_path = args.sample
    img_dir = args.vis_root or os.path.join(DATA_DIR, "images")

    entry = json.load(open(sample_path))

    print("[input] checkpoint :", args.ckpt)
    print("[input] sample     :", sample_path)
    print("[input] vis_root   :", img_dir)
    print("[input] image      :", entry["image"])

    infer = build_inference(args, "und", img_dir)
    out = infer.run(entry)

    print("=" * 60)
    print("description:", out["text"])
    for o in out["orientation"] or []:
        if o["valid"]:
            print(f"OBJECT{o['obj_idx']+1} orientation: azimuth={o['azimuth']:.2f}, polar={o['polar']:.2f}, rotation={o['rotation']:.2f}")
    print("GT orientation:", entry["box_data"][0].get("orient"))


if __name__ == "__main__":
    main()
