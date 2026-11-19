import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spatial_unify.cli import make_demo_parser, build_inference

DATA_DIR = os.path.join(REPO_ROOT, "examples", "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")


def main():
    args = make_demo_parser(
        "i2i", "Image-to-image: generate the target view given an orientation change.",
        os.path.join(DATA_DIR, "i2i_shoe.json"),
    ).parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sample_path = args.sample
    img_dir = args.vis_root or os.path.join(DATA_DIR, "images")

    entry = json.load(open(sample_path))
    for key in ("cnocs_map", "cnocs_map_vis"):
        if key in entry and not os.path.isabs(entry[key]):
            entry[key] = os.path.join(img_dir, entry[key])

    print("[input] checkpoint :", args.ckpt)
    print("[input] sample     :", sample_path)
    print("[input] vis_root   :", img_dir)
    print("[input] images     :", entry["image"])
    if "cnocs_map" in entry:
        print("[input] cnocs_map  :", entry["cnocs_map"])

    infer = build_inference(args, "i2i", img_dir)
    out = infer.run(entry)

    from PIL import Image

    def fit_height(img, h):
        w = round(img.width * h / img.height)
        return img.resize((w, h), Image.BILINEAR)

    parts = []
    input_path = os.path.join(img_dir, entry["image"][0])
    if os.path.exists(input_path):
        parts.append(fit_height(Image.open(input_path).convert("RGB"), out["image"].height))
    pose_vis_path = os.path.splitext(entry.get("cnocs_map", ""))[0] + "_vis.png"
    if os.path.exists(pose_vis_path):
        parts.append(fit_height(Image.open(pose_vis_path).convert("RGB"), out["image"].height))
    parts.append(out["image"])
    combined = Image.new("RGB", (sum(p.width for p in parts), out["image"].height))
    x = 0
    for p in parts:
        combined.paste(p, (x, 0))
        x += p.width

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(sample_path))[0].replace("i2i_", "")
    out_path = os.path.join(OUT_DIR, f"i2i_{stem}_result.png")
    combined.save(out_path)
    print("[output] generated image (input+pose_vis+output):", out_path)


if __name__ == "__main__":
    main()
