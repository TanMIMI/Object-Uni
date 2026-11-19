import os
import re
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from openexr_numpy import imread as exr_imread

import sys as _sys
import types as _types
try:
    import transformers.video_utils
except ModuleNotFoundError:
    import transformers.image_utils as _tf_image_utils
    _video_utils_shim = _types.ModuleType("transformers.video_utils")
    _video_utils_shim.VideoInput = _tf_image_utils.VideoInput
    _sys.modules["transformers.video_utils"] = _video_utils_shim

from image_processing_bailingmm import BailingMMImageProcessor as Qwen2VLImageProcessor

from .model.templates import (
    Bailingv4Template,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_CN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

LLAVA_DEFAULT_IMAGE_TOKEN = "<image>"

def get_closest_ratio(height: float, width: float, aspect_ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(aspect_ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return aspect_ratios[closest_ratio], float(closest_ratio)

ASPECT_RATIO_512 = {
    "0.25": [256, 1024], "0.26": [256, 992], "0.27": [256, 960], "0.28": [256, 928],
    "0.32": [288, 896], "0.33": [288, 864], "0.35": [288, 832], "0.4": [320, 800],
    "0.42": [320, 768], "0.48": [352, 736], "0.5": [352, 704], "0.52": [352, 672],
    "0.57": [384, 672], "0.6": [384, 640], "0.68": [416, 608], "0.72": [416, 576],
    "0.78": [448, 576], "0.82": [448, 544], "0.88": [480, 544], "0.94": [480, 512],
    "1.0": [512, 512], "1.07": [512, 480], "1.13": [544, 480], "1.21": [544, 448],
    "1.29": [576, 448], "1.38": [576, 416], "1.46": [608, 416], "1.67": [640, 384],
    "1.75": [672, 384], "2.0": [704, 352], "2.09": [736, 352], "2.4": [768, 320],
    "2.5": [800, 320], "2.89": [832, 288], "3.0": [864, 288], "3.11": [896, 288],
    "3.62": [928, 256], "3.75": [960, 256], "3.88": [992, 256], "4.0": [1024, 256],
}

def resize_with_padding(x, target_size):
    target_h, target_w = target_size
    target_ratio = target_w / target_h
    orig_ratio = x.width / x.height
    ratio_diff = abs(orig_ratio - target_ratio)
    if ratio_diff > 0.1:
        return x.resize((target_w, target_h), Image.BILINEAR)
    if orig_ratio > target_ratio:
        new_height = int(round(x.width / target_ratio))
        new_width = x.width
    else:
        new_width = int(round(x.height * target_ratio))
        new_height = x.height
    padded = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    paste_x = (new_width - x.width) // 2
    paste_y = (new_height - x.height) // 2
    padded.paste(x, (paste_x, paste_y))
    return padded.resize((target_w, target_h), Image.BILINEAR)

def resize_with_padding_tensor(x, target_size, padding_value=-1.0):
    orig_c, orig_h, orig_w = x.shape
    target_h, target_w = target_size
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h
    ratio_diff = abs(orig_ratio - target_ratio)
    x_unsqueezed = x.unsqueeze(0)
    if ratio_diff > 0.1:
        out = F.interpolate(x_unsqueezed, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return out.squeeze(0)
    if orig_ratio > target_ratio:
        new_height = int(round(orig_w / target_ratio))
        new_width = orig_w
    else:
        new_width = int(round(orig_h * target_ratio))
        new_height = orig_h
    padded_canvas = torch.full((orig_c, new_height, new_width), padding_value, dtype=x.dtype, device=x.device)
    paste_x = (new_width - orig_w) // 2
    paste_y = (new_height - orig_h) // 2
    padded_canvas[:, paste_y:paste_y + orig_h, paste_x:paste_x + orig_w] = x
    padded_canvas = padded_canvas.unsqueeze(0)
    out = F.interpolate(padded_canvas, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return out.squeeze(0)

def format_box(box, ndigits=2):
    return f"[{box[0]:.{ndigits}f}, {box[1]:.{ndigits}f}, {box[2]:.{ndigits}f}, {box[3]:.{ndigits}f}]"

def replace_box_for_object(text, obj_name, new_box, ndigits=2):
    pattern = rf"({re.escape(obj_name)}\s+in\s+bbox\s*)\[[^\]]+\]"
    repl = rf"\1{format_box(new_box, ndigits)}"
    new_text = re.sub(pattern, repl, text)
    return new_text

def resize_bbox_direct(bbox, ori_w, ori_h, target_w, target_h):
    x1, y1, x2, y2 = bbox
    sx = target_w / float(ori_w)
    sy = target_h / float(ori_h)
    x1 = x1 * sx
    x2 = x2 * sx
    y1 = y1 * sy
    y2 = y2 * sy
    x1 = max(0, min(x1, target_w - 1))
    x2 = max(0, min(x2, target_w - 1))
    y1 = max(0, min(y1, target_h - 1))
    y2 = max(0, min(y2, target_h - 1))
    return [x1, y1, x2, y2]

def crop_bbox_to_centered_square(image, bbox, fill_color=(255, 255, 255)):
    W, H = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W, int(round(x1))))
    y1 = max(0, min(H, int(round(y1))))
    x2 = max(0, min(W, int(round(x2))))
    y2 = max(0, min(H, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")
    crop = image.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size
    side = max(crop_w, crop_h)
    square_img = Image.new("RGB", (side, side), fill_color)
    paste_x = (side - crop_w) // 2
    paste_y = (side - crop_h) // 2
    square_img.paste(crop, (paste_x, paste_y))
    img = square_img.resize((518, 518), Image.BICUBIC)
    x = transforms.ToTensor()(img)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    x = (x - mean) / std
    return x

def if_all_en(raw_text):
    for c_char in raw_text:
        unicode_val = ord(c_char)
        if 0x4E00 <= unicode_val <= 0x9FFF:
            return False
    return True

class SpatialUnifyProcessor:

    def __init__(
        self,
        tokenizer,
        model_config,
        vis_root,
        data_type,
        max_length=2048,
        num_query_token=2560,
        num_metamorph_learnable_queries=256,
        rotary_type="1d-ntk",
        min_pixels=451584,
        max_pixels=451584,
        max_objs=2,
    ):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.vis_root = vis_root
        self.data_type = data_type
        self.max_length = max_length
        if data_type == "image_text":
            self.max_length += num_query_token + 2
        elif data_type == "image_gen_freeform":
            self.max_length += (num_query_token + 2) * 3 + num_metamorph_learnable_queries + 2
        self.num_metamorph_learnable_queries = num_metamorph_learnable_queries
        self.rotary_type = rotary_type
        self.max_objs = max_objs
        self.predefined_aspect_ratios = ASPECT_RATIO_512

        self.vis_processor = Qwen2VLImageProcessor()
        self.vis_processor.max_pixels = max_pixels
        self.vis_processor.min_pixels = min_pixels
        self.vis_processor.temporal_patch_size = 2
        self.vis_processor.merge_size = 2

        self.chat_template = Bailingv4Template(tokenizer)
        self.llama_pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.image_start_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        self.image_end_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        self.image_patch_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

    def process(self, sources, ann):
        result = {
            "image": torch.zeros(1, 3, 224, 224),
            "image_grid_thw": None,
            "second_per_grid_thw": None,
            "decoder_image": None,
            "task_type": "others",
            "dataset_type": "nlp",
            "input_ids": None,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "weights": None,
            "audio": None,
            "input_text": None,
            "placehold_text": "",
        }
        if self.data_type == "image_text":
            result = self._process_image_text(sources, ann, result)
        elif self.data_type == "image_gen_freeform":
            result = self._process_image_gen(sources, ann, result)
        else:
            raise NotImplementedError(self.data_type)
        return result

    def _process_image_text(self, data, ann, result):
        max_objs = self.max_objs
        image_file = os.path.join(self.vis_root, ann.get("image_name_train_ds", ann["image"]))
        if not os.path.exists(image_file):
            raise Exception(f"{image_file} not exists!")
        image = Image.open(image_file).convert("RGB")

        image_inputs = self.vis_processor(images=image, videos=None, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        pixel_values = image_inputs["pixel_values"]

        num_query_token = torch.prod(image_grid_thw, dim=1) // 4
        assert num_query_token.shape[0] == 1
        for sentence in data:
            if LLAVA_DEFAULT_IMAGE_TOKEN in sentence["value"]:
                assert sentence["value"].count(LLAVA_DEFAULT_IMAGE_TOKEN) == 1
                num_img_patch = int(num_query_token[0].item())
                img_text = (
                    DEFAULT_IM_START_TOKEN
                    + num_img_patch * DEFAULT_IMAGE_PATCH_TOKEN
                    + max_objs * DEFAULT_IMAGE_PATCH_TOKEN
                    + DEFAULT_IM_END_TOKEN
                )
                sentence["value"] = sentence["value"].replace(LLAVA_DEFAULT_IMAGE_TOKEN, img_text).strip()

        obj_images = []
        obj_masks = []
        orient_list = []
        for obj in ann["box_data"][:max_objs]:
            image_tensor = crop_bbox_to_centered_square(image, obj["bbox"])
            obj_images.append(image_tensor)
            obj_masks.append(1)
            orient_list.append([obj['orient']["azimuth"], obj['orient']["polar"], obj['orient']["rotation"]])
        if len(obj_images) > 0:
            pad_tensor = torch.zeros_like(obj_images[0])
        else:
            pad_tensor = torch.zeros(3, 518, 518)
        while len(obj_images) < max_objs:
            obj_images.append(pad_tensor.clone())
            obj_masks.append(0)
            orient_list.append([0, 0, 0])

        obj_images = torch.stack(obj_images, dim=0)
        obj_masks = torch.tensor(obj_masks, dtype=torch.bool)
        orient = torch.tensor(orient_list, dtype=torch.float32)

        result.update(
            {
                "image": pixel_values,
                "image_grid_thw": image_grid_thw,
                "dataset_type": self.data_type,
                "obj_images": obj_images,
                "obj_masks": obj_masks,
                "orient": orient,
                "num_orientation_tokens": max_objs,
            }
        )
        result = self._process_text_and_pack(data, ann, result)
        result["orientation_token_mask"] = self.build_orientation_token_mask(
            result["input_ids"],
            max_objs=max_objs,
        )
        return result

    def build_orientation_token_mask(self, input_ids, max_objs):
        """Mark the last max_objs image_patch_token before each image_end_token as orientation query tokens."""
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        orientation_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        image_patch_token_id = self.model_config.image_patch_token
        image_end_token_id = self.model_config.image_end_token
        image_end_positions = torch.where(input_ids == image_end_token_id)[0]
        for end_pos in image_end_positions:
            patch_positions = []
            j = int(end_pos.item()) - 1
            while j >= 0 and input_ids[j].item() == image_patch_token_id:
                patch_positions.append(j)
                j -= 1
            assert len(patch_positions) >= max_objs, (
                f"Not enough image_patch tokens before image_end. "
                f"Found {len(patch_positions)}, need {max_objs}."
            )
            orient_positions = patch_positions[:max_objs]
            orientation_token_mask[orient_positions] = True
        return orientation_token_mask

    def _process_ratio(self, ann):
        if self.predefined_aspect_ratios is not None:
            ori_h, ori_w = ann['img_h'], ann['img_w']
            ori_h = ori_h[-1] if isinstance(ori_h, list) else ori_h
            ori_w = ori_w[-1] if isinstance(ori_w, list) else ori_w
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, aspect_ratios=self.predefined_aspect_ratios)
            closest_size = list(map(lambda x: int(x), closest_size))
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = closest_size[0], int(ori_w * float(closest_size[0]) / ori_h)
            else:
                resize_size = int(ori_h * float(closest_size[1]) / ori_w), closest_size[1]
        else:
            closest_size = 512
            resize_size = 512
            closest_ratio = float('1.0')
        return closest_size, resize_size, closest_ratio

    def _process_image_gen(self, data, ann, result):
        image_list = []
        decoder_image_list = []
        ann_image_list = ann["image"] if isinstance(ann["image"], list) else [ann["image"]]
        assert len(ann_image_list) >= 1
        map_data = exr_imread(
            ann["cnocs_map"] if isinstance(ann["cnocs_map"], str) else BytesIO(ann["cnocs_map"]),
            channel_names=[
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
                "ViewLayer.Combined.A",
            ])
        map_rgb_np = map_data[:, :, :3]
        map_alpha_np = map_data[:, :, 3]

        alpha_mask = map_alpha_np > 1e-5
        alpha_mask_3d = np.expand_dims(alpha_mask, axis=2)
        map_rgb_np = np.where(alpha_mask_3d, map_rgb_np, 0.0)  

        raw_map_tensor = torch.from_numpy(map_rgb_np).permute(2, 0, 1).float()

        closest_size, resize_size, closest_ratio = self._process_ratio(ann)
        train_vae_transforms = transforms.Compose([
            transforms.Resize(closest_size,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for imp_idx, imp in enumerate(ann_image_list):
            image = None
            try:
                if isinstance(imp, str):
                    image_file = os.path.join(self.vis_root, imp)
                    image = Image.open(image_file).convert("RGB")
                else:
                    image = Image.open(BytesIO(imp)).convert("RGB")
            except Exception as e1:
                h = ann['img_h'][imp_idx] if isinstance(ann['img_h'], list) else ann['img_h']
                w = ann['img_w'][imp_idx] if isinstance(ann['img_w'], list) else ann['img_w']
                image = Image.new('RGB', (w, h), (128, 128, 128))
                if imp_idx != len(ann_image_list) - 1:
                    print(f"wrong image path: {imp}, error info: {e1}")

            assert image is not None
            image_list.append(image)

        for image_idx, image in enumerate(image_list):
            decoder_image = None
            if image_idx == len(image_list) - 1:
                decoder_image = train_vae_transforms(image)
            else:
                assert resize_size is not None
                image_padded = resize_with_padding(image, resize_size)
                decoder_image = train_vae_transforms(image_padded)
            assert decoder_image is not None
            decoder_image_list.append(decoder_image)

        if raw_map_tensor is not None:
            map_padded = resize_with_padding_tensor(
                raw_map_tensor,
                target_size=closest_size,
                padding_value=0.0
            )
            if len(decoder_image_list) < 2:
                decoder_image_list.insert(0, map_padded)
            else:
                result["ref_cnocs"] = map_padded

        pixel_decoder_values = torch.cat(decoder_image_list[-1:])
        pixel_ref_values = None
        if len(decoder_image_list) > 1:
            pixel_ref_values = torch.cat(decoder_image_list[0:1])

        image_inputs = self.vis_processor(images=image_list, videos=None, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        pixel_values = image_inputs["pixel_values"]

        image_index_whether_from_human = {}
        image_index_cnt = 0
        for conversation in ann["conversations"]:
            for _ in range(conversation["value"].count(LLAVA_DEFAULT_IMAGE_TOKEN)):
                image_index_whether_from_human[image_index_cnt] = conversation["from"] == 'human'
                image_index_cnt += 1

        image_from_human_mask = torch.tensor([image_index_whether_from_human[i] for i in range(len(image_list))]).long()
        num_query_token = (torch.prod(image_grid_thw, dim=1) // 4) * image_from_human_mask + (torch.tensor([self.num_metamorph_learnable_queries] * len(image_list))) * (1 - image_from_human_mask)
        img_w = ann["img_w"][1] if isinstance(ann["img_w"], list) else ann["img_w"]
        img_h = ann["img_h"][1] if isinstance(ann["img_h"], list) else ann["img_h"]
        target_h, target_w = pixel_decoder_values.shape[-2:]

        if isinstance(ann["box_data"], dict):
            box_data = ann["box_data"]["target"]
        else:
            box_data = ann["box_data"]
        for obj in box_data[:self.max_objs]:
            resized_box = resize_bbox_direct(obj["bbox"], img_w, img_h, target_w, target_h)
            obj_name = f"{obj['category'].upper()}{obj['idx']}"
            ann["conversations"][0]["value"] = replace_box_for_object(ann["conversations"][0]["value"], obj_name, resized_box, ndigits=2)

        image_index = 0
        for sentence in data:
            num_images = sentence["value"].count(LLAVA_DEFAULT_IMAGE_TOKEN)
            if num_images > 0:
                for i in range(image_index, num_images + image_index):
                    img_text = DEFAULT_IM_START_TOKEN + num_query_token[
                        i] * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN
                    sentence["value"] = sentence["value"].replace(LLAVA_DEFAULT_IMAGE_TOKEN, img_text, 1).strip()
            image_index += num_images

        total_images = num_query_token.shape[0]
        if image_index != total_images:
            raise ValueError(
                f"Image number and text image tokens do not match: text image tokens: {image_index}, total images: {total_images}."
            )

        result.update(
            {
                "image": pixel_values,
                "image_grid_thw": image_grid_thw,
                "dataset_type": self.data_type,
                "decoder_image": pixel_decoder_values,
                "weight_mask": None,
            }
        )
        if pixel_ref_values is not None:
            result["ref_image"] = pixel_ref_values

        return self._process_text_and_pack(data, ann, result)

    def _preprocess_text(self, sources, ann, result):
        if sources[0]["from"] != "human":
            sources = sources[1:]

        task_prompt, task_prompt_id = "", []
        sys_prompt_data = ""
        if "system" in ann:
            sys_prompt_data = ann["system"]

        if if_all_en(sources[0]["value"]):
            prompt = SYSTEM_PROMPT + sys_prompt_data
        else:
            prompt = SYSTEM_PROMPT_CN + sys_prompt_data

        position_ids = None
        chat_template = self.chat_template

        input_text, input_ids, labels = chat_template.handle_system_prompt(prompt)

        for j, sentence in enumerate(sources):
            role = sentence["from"]
            if j % 2 == 0:  
                if j == 0:
                    task_prompt_ = task_prompt
                    task_prompt_id_ = task_prompt_id
                else:
                    task_prompt_ = ""
                    task_prompt_id_ = []
                question = sentence["value"]
                user_text, user_ids, user_labels = chat_template.handle_user_message(
                    question, task_prompt_, task_prompt_id_
                )
                input_text += user_text
                input_ids.extend(user_ids)
                labels.extend(user_labels)
            else:  
                if j < len(sources) - 1:
                    assistant_text, assistant_ids, assistant_labels, _, _ = (
                        chat_template.handle_assistant_message(
                            sentence["value"], j == len(sources) - 1
                        )
                    )
                    input_text += assistant_text
                    input_ids.extend(assistant_ids)
                    labels.extend(assistant_labels)
                else:
                    assistant_text, assistant_ids, assistant_labels, output_length, assistant_prefix_length = (
                        chat_template.handle_assistant_message(
                            sentence["value"], j == len(sources) - 1
                        )
                    )
                    input_length = len(input_ids) + assistant_prefix_length
                    position_ids = self.tokenizer._build_position_ids(
                        mask_pos=input_length - 2, bos_pos=input_length - 1, max_output_length=output_length,
                        rotary_type=self.rotary_type
                    )
                    input_text += assistant_text
                    input_ids.extend(assistant_ids)
                    labels.extend(assistant_labels)

        assert len(input_ids) == len(labels)

        input_text, input_ids, labels, attention_mask, position_ids, attention_mask_for_mrope = chat_template.finalize_sequence(
            input_text, input_ids, labels, self.max_length, position_ids
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            attention_mask_for_mrope=attention_mask_for_mrope,
            position_ids=position_ids,
            input_text=input_text,
        )

    def _process_text_and_pack(self, data, ann, result):
        weights = [1.0]

        ret = self._preprocess_text(data, ann, result=result)

        input_text = ret["input_text"]
        input_ids = ret["input_ids"].tolist()
        position_ids = ret["position_ids"]
        attention_mask = ret["attention_mask"]
        attention_mask_for_mrope = ret["attention_mask_for_mrope"]
        attention_mask_for_mrope = attention_mask_for_mrope.tolist()
        labels = ret["labels"].tolist()
        image_start_indices = list(torch.where(torch.tensor(input_ids) == self.image_start_token)[0])
        image_end_indices = list(torch.where(torch.tensor(input_ids) == self.image_end_token)[0])
        assert len(image_start_indices) == len(image_end_indices), f"Expected same number of image start and end indices, got {len(image_start_indices)} and {len(image_end_indices)}"

        if self.data_type == "image_text":
            num_images = 1
            assert len(image_start_indices) == num_images
            assert len(image_end_indices) == num_images
        elif self.data_type == "image_gen_freeform":
            num_images = result["image_grid_thw"].shape[0]
            assert len(image_start_indices) == num_images
            assert len(image_end_indices) == num_images

        real_length = len(input_ids)
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids += [self.llama_pad_id] * pad_length
            labels.extend([-100] * pad_length)
            position_ids[0] += [0] * pad_length
            position_ids[1] += [0] * pad_length
            assert len(input_ids) == len(position_ids[0])
            attention_mask = torch.Tensor([[0], [1]]).unsqueeze(0).long()
            attention_mask_for_mrope += [0] * pad_length
            attention_mask_for_mrope = torch.tensor(attention_mask_for_mrope, dtype=torch.int32)
        else:
            attention_mask = torch.tensor(attention_mask, dtype=torch.int32)
            attention_mask_for_mrope = torch.tensor(attention_mask_for_mrope, dtype=torch.int32)

        assert len(input_ids) == len(labels) == self.max_length

        n_image_features = 0
        n_image_tokens = input_ids.count(self.image_patch_token)
        if result["image_grid_thw"] is not None:
            n_image_features = int(sum(torch.prod(result["image_grid_thw"], dim=-1) // 4))
            result["image_grid_thw"] = result["image_grid_thw"].tolist()

        num_orientation_tokens = result.get("num_orientation_tokens", 0)
        expected_image_tokens = n_image_features + num_orientation_tokens
        if n_image_tokens != expected_image_tokens and self.data_type != "image_gen_freeform":
            print(
                f"Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, "
                f"features: {n_image_features}, "
                f"orientation_tokens: {num_orientation_tokens}, "
                f"expected: {expected_image_tokens}"
            )

        result.update(
            {
                "input_ids": torch.tensor(input_ids),
                "position_ids": (torch.tensor(position_ids) if position_ids is not None else None),
                "attention_mask": attention_mask,
                "attention_mask_for_mrope": attention_mask_for_mrope,
                "labels": torch.tensor(labels),
                "weights": weights,
                "input_text": input_text,
            }
        )
        result["real_length"] = real_length
        return result
