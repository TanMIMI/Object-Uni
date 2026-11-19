import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModelForCausalLM

from configuration_glm import GLMConfig
from modeling_bailing_moe import GLMForConditionalGeneration
from tokenization_bailing import BailingTokenizer
from .templates import (
    BAILING_MM_TOKENS,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_GEN_IMAGE_PATCH_TOKEN,
    DEFAULT_GEN_IM_START_TOKEN,
    DEFAULT_GEN_IM_END_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    DEFAULT_END_OF_CHUNK_TOKEN,
    DEFAULT_END_OF_AUDIO_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AU_START_TOKEN,
    DEFAULT_AU_END_TOKEN,
    DEFAULT_GEN_AUDIO_PATCH_TOKEN,
    DEFAULT_GEN_AU_START_TOKEN,
    DEFAULT_GEN_AU_END_TOKEN,
    PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    DEFAULT_FRAME_PATCH_TOKEN,
    DEFAULT_TEXT_TOKEN,
    DEFAULT_ASR_TOKEN,
    DEFAULT_TTS_TOKEN,
)
from .gen_helpers import (
    concat_horizontal,
    append_understand_embeds_with_learnable_scales,
    append_input_ids_with_learnable_scales,
    interpolate_imgfeat_to_256tokens,
)
from diffusion.sd3_loss import SD3Sampler

from qwen2_5_vit import Qwen2_5_VisionTransformer


def _build_llm_config(llm_model_path, tokenizer):
    """Equivalent to llm_builder.build_config: build GLMConfig from the base config and inject multimodal special token ids."""
    llm_config = GLMConfig.from_pretrained(llm_model_path)

    llm_config.first_signal_token = tokenizer.convert_tokens_to_ids("[IMG0]")
    llm_config.image_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
    llm_config.image_end_token = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
    llm_config.image_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
    llm_config.video_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_VID_START_TOKEN)
    llm_config.video_end_token = tokenizer.convert_tokens_to_ids(DEFAULT_VID_END_TOKEN)
    llm_config.gen_image_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_IM_START_TOKEN)
    llm_config.gen_image_end_token = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_IM_END_TOKEN)
    llm_config.gen_image_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_IMAGE_PATCH_TOKEN)
    llm_config.placeholder_image_token_in_text = tokenizer.convert_tokens_to_ids(PLACEHOLDER_IMAGE_TOKEN_IN_TEXT)
    llm_config.end_of_chunk_token = tokenizer.convert_tokens_to_ids(DEFAULT_END_OF_CHUNK_TOKEN)

    llm_config.end_of_audio_token = tokenizer.convert_tokens_to_ids(DEFAULT_END_OF_AUDIO_TOKEN)
    llm_config.audio_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_AU_START_TOKEN)
    llm_config.audio_end_token = tokenizer.convert_tokens_to_ids(DEFAULT_AU_END_TOKEN)
    llm_config.audio_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_PATCH_TOKEN)
    llm_config.gen_audio_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AU_START_TOKEN)
    llm_config.gen_audio_end_token = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AU_END_TOKEN)
    llm_config.gen_audio_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AUDIO_PATCH_TOKEN)
    llm_config.placeholder_audio_token_in_text = tokenizer.convert_tokens_to_ids(PLACEHOLDER_AUDIO_TOKEN_IN_TEXT)
    llm_config.frame_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_FRAME_PATCH_TOKEN)
    llm_config.video_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
    llm_config.text_token = tokenizer.convert_tokens_to_ids(DEFAULT_TEXT_TOKEN)
    llm_config.asr_token = tokenizer.convert_tokens_to_ids(DEFAULT_ASR_TOKEN)
    llm_config.tts_token = tokenizer.convert_tokens_to_ids(DEFAULT_TTS_TOKEN)

    llm_config.num_image_token = 2560
    llm_config.num_video_token = 3072
    llm_config.num_audio_token = 256
    llm_config.num_decoder_image_token = 256
    llm_config.num_decoder_audio_token = 256

    llm_config.use_mot = True
    llm_config.gen_router_type = "MultiRouter"
    llm_config.audio_vocab_size = 0
    llm_config.audio_id_shift = len(tokenizer)
    llm_config.moe_config["expert_model_parallelism"] = False
    return llm_config


class BailingMMSpatial(nn.Module):
    """Inference-only BailingMM (spatial_tmn branch config)."""

    def __init__(
        self,
        llm_model_path,
        vision_model_path,
        connector_model_path,
        sd3_model_path,
        use_orientation_query=True,
        use_vae_ref=True,
        device="cpu",
        load_base_weights=False,
    ):
        """
        load_base_weights=False (default): only read config/tokenizer from the base directory;
        GLM/vision/connector/SD3 transformer are randomly initialized and then fully
        overwritten by the merge ckpt (verified: the only missing ckpt keys are the VAE).
        In this case the base directory needs no weight files; only the SD3.5 VAE needs real weights.
        load_base_weights=True: load base weights first, then overwrite with the ckpt
        (same as the original repo, slower to load).
        """
        super().__init__()

        self.use_vae_ref = use_vae_ref
        self.use_orientation_query = use_orientation_query
        self.pool_after_mlp = True
        self.norm_query_embeds = True
        self.num_query_token = 2560
        self.freeze_mme = True
        self.max_txt_len = 2048
        self.img_gen_scales = [4, 8, 16]
        self.num_metamorph_learnable_queries = 256

        self.tokenizer = BailingTokenizer.from_pretrained(llm_model_path)
        self.tokenizer.add_tokens(BAILING_MM_TOKENS, special_tokens=True)

        self.model_config = _build_llm_config(llm_model_path, self.tokenizer)
        if load_base_weights:
            self.model = GLMForConditionalGeneration.from_pretrained(
                llm_model_path, config=self.model_config
            )
        else:
            self.model = GLMForConditionalGeneration(self.model_config)
        terminators = {self.tokenizer.eos_token_id}
        terminators.add(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.model.gen_terminator = list(terminators)

        if load_base_weights:
            self.vision = Qwen2_5_VisionTransformer.from_pretrained(
                vision_model_path,
                attn_implementation="flash_attention_2",
            )
        else:
            from qwen2_5_vit import Qwen2_5_VLVisionConfig
            vision_config = Qwen2_5_VLVisionConfig.from_pretrained(vision_model_path)
            vision_config._attn_implementation = "flash_attention_2"
            self.vision = Qwen2_5_VisionTransformer(vision_config)
        self.vision.image_emb_dim = 8192
        self.vision.eval()

        hidden = self.model_config.hidden_size

        self.linear_proj = nn.Sequential(
            nn.Linear(self.vision.image_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.query_tokens_dict = nn.ParameterDict()
        for scale in self.img_gen_scales:
            num_tokens = scale * scale
            self.query_tokens_dict[f"{scale}x{scale}"] = nn.Parameter(
                torch.nn.functional.normalize(torch.randn(num_tokens, hidden), dim=-1)
            )
        self.scale_indices = []
        current_idx = 0
        for scale in self.img_gen_scales:
            current_idx += scale * scale
            self.scale_indices.append(current_idx)

        if load_base_weights:
            self.connector = AutoModelForCausalLM.from_pretrained(connector_model_path)
        else:
            from transformers import AutoConfig
            self.connector = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(connector_model_path)
            )
        for layer in self.connector.model.layers:
            layer.self_attn.is_causal = False
        self.proj_in = nn.Linear(hidden, self.connector.config.hidden_size)
        self.proj_out = nn.Linear(self.connector.config.hidden_size, hidden)
        self.proj_vit = nn.Linear(self.vision.image_emb_dim, hidden)
        self.proj_vit_norm = torch.nn.LayerNorm(hidden)

        self.diffusion_loss = SD3Sampler(
            sd3_model_path, vision_dim=hidden, device=device,
            ref_add_noise=False, ref_add_noise_ratio=0.25,
            load_base_weights=load_base_weights,
        )

        if self.use_orientation_query:
            from .hf_dinov2 import HFRegDinov2

            dino_dim = 1024
            self.dino_model = HFRegDinov2()
            self.dino_model.eval()
            self.orientation_token = nn.Parameter(torch.zeros(1, 1, hidden))
            self.orientation_query_proj = nn.Sequential(
                nn.LayerNorm(dino_dim),
                nn.Linear(dino_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.orientation_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.orientation_azimuth_head = nn.Linear(hidden, 360)
            self.orientation_polar_head = nn.Linear(hidden, 180)
            self.orientation_rotation_head = nn.Linear(hidden, 360)

    def load_checkpoint(self, url_or_filename, skip_prefixes=("audio.", "linear_proj_audio.")):
        """Load *_merge.pth: key preprocessing + strict=False load.

        Key preprocessing (equivalent to the original BailingMM.load_checkpoint + GLMSparseMLP lsd hook):
        - drop vision freqs_cos/freqs_sin/position_ids
        - experts.w1/w2 -> experts.mlp.w1/w2 (the effect of the original lsd hook at world_size=1,
          avoiding any dependency on the atorch distributed group)
        - skip the audio tower (not used in this inference config)
        """
        assert os.path.isfile(url_or_filename), f"checkpoint '{url_or_filename}' not found"
        print(f"# ******* loading checkpoint from {url_or_filename} ******* #")
        with open(url_or_filename, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint

        new_state_dict = {}
        dino_keys = {}
        for key, value in state_dict.items():
            if key.startswith(skip_prefixes):
                continue
            if "vision" in key and ("freqs_cos" in key or "freqs_sin" in key or "position_ids" in key):
                continue
            if key.startswith("dino_model."):
                dino_keys[key] = value
                continue
            key = key.replace(".experts.w1", ".experts.mlp.w1").replace(".experts.w2", ".experts.mlp.w2")
            new_state_dict[key] = value

        if dino_keys:
            if self.use_orientation_query:
                from .hf_dinov2 import map_dinov2_state_dict
                mapped = map_dinov2_state_dict(dino_keys, prefix="dino_model.")
                new_state_dict.update({f"dino_model.{k}": v for k, v in mapped.items()})
            else:
                print(f"# skip {len(dino_keys)} dino_model.* keys (use_orientation_query=False)")

        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys ({len(msg.missing_keys)}): {msg.missing_keys[:10]} ...")
        print(f"Unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:10]} ...")
        return msg

    @torch.no_grad()
    def metaqueries_generate(
        self,
        samples,
        steps=20,
        seed=42,
        cfg=7.0,
        image_cfg=1.0,
        cfg_mode=1,
        height=512,
        width=512,
    ):
        height = int(height)
        width = int(width)
        decoder_images = samples.get("decoder_image", None)
        if decoder_images is not None:
            print("using decoder_images.shape:", decoder_images.shape)
            height, width = decoder_images.shape[-2:]

        image_embed_list = []
        if ("image" in samples) and (samples["image"] is not None):
            device = samples["image"].device
            images = samples["image"]
            if not isinstance(images, list):
                images = [images]
        else:
            device = samples["input_ids"].device
            images = []

        ref_images = samples.get("ref_image", None) if self.use_vae_ref else None
        ref_cnocs = samples.get("ref_cnocs", None) if self.use_vae_ref else None

        obj_images = samples.get("obj_images", None)
        obj_masks = samples.get("obj_masks", None)
        orientation_token_mask = samples.get("orientation_token_mask", None)

        use_orientation_this_sample = (
            getattr(self, "use_orientation_query", False)
            and samples.get("dataset_type", None) in ["image_text", "image_audio"]
            and obj_images is not None
            and obj_masks is not None
            and orientation_token_mask is not None
        )

        print("metaqueries_generate use_orientation_this_sample:", use_orientation_this_sample)

        image_embed_list = []
        image_grid_thw = None
        for idx, item in enumerate(images):
            if len(images) > 0 and images[idx].size(0) > 0:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = torch.cat(images).to(self.vision.device)
                    image_grid_thw_array = []
                    for image_grid_thw_ in [samples.get("image_grid_thw")]:
                        image_grid_thw_array.extend(image_grid_thw_)
                    image_grid_thw = torch.tensor(image_grid_thw_array, device=images.device)
                    eva_image_feat = self.vision(
                        images.type(self.vision.get_dtype()),
                        grid_thw=image_grid_thw,
                    )

                    image_grid_thw = image_grid_thw.to(device)

                    if samples["dataset_type"] in ["image_gen_freeform"]:
                        image_start_token_idxes = torch.where(samples["input_ids"].cpu().view(-1) == self.model.config.image_start_token)[0].tolist()
                        labels_as_list = samples["labels"].cpu().view(-1).cpu().tolist()
                        isgen_indicators = [labels_as_list[i] >= 0 for i in image_start_token_idxes]
                        assert len(isgen_indicators) == image_grid_thw.shape[0]

                        eva_image_feat, image_grid_thw, eva_image_feat_input, image_grid_thw_input, _, _ = interpolate_imgfeat_to_256tokens(
                            eva_image_feat, image_grid_thw, isgen_indicators=isgen_indicators, input_resize_256=False)

                        if not isinstance(eva_image_feat_input, list):
                            image_embed_list.append(eva_image_feat_input)
                            image_grid_thw = image_grid_thw_input
                        else:
                            image_grid_thw = None
                    else:
                        image_embed_list.append(eva_image_feat)

        image_embeds = None
        inputs_opt_visual = None
        device = self.vision.device
        if len(image_embed_list) > 0:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                image_embeds = torch.cat(image_embed_list).to(device)
                image_embeds = image_embeds.float()
                inputs_opt_visual = self.linear_proj(image_embeds)

            if self.norm_query_embeds:
                inputs_opt_visual = F.normalize(inputs_opt_visual, dim=-1)

        inputs = {}
        asistant_start = torch.where(samples["labels"].cpu().view(-1) != -100)[0].tolist()
        samples["input_ids"] = samples["input_ids"][:asistant_start[0] + 2].unsqueeze(0)
        samples["position_ids"] = samples["position_ids"][:, :samples["real_length"]].unsqueeze(0)
        inputs["input_ids"] = samples["input_ids"].to(device)
        inputs["position_ids"] = samples["position_ids"].to(device)
        if use_orientation_this_sample:
            orientation_token_mask = orientation_token_mask.bool().to(device)

            if orientation_token_mask.ndim == 1:
                orientation_token_mask = orientation_token_mask[:inputs["input_ids"].shape[1]].unsqueeze(0)
            elif orientation_token_mask.ndim == 2:
                orientation_token_mask = orientation_token_mask[:, :inputs["input_ids"].shape[1]]
            else:
                raise ValueError(f"Unexpected orientation_token_mask shape: {orientation_token_mask.shape}")

            assert orientation_token_mask.shape == inputs["input_ids"].shape, (
                f"orientation_token_mask.shape={orientation_token_mask.shape}, "
                f"input_ids.shape={inputs['input_ids'].shape}"
            )

        query_embeds_image = inputs_opt_visual
        query_embeds_video = None
        image_grid_thw_video = None

        if use_orientation_this_sample:
            assert query_embeds_image is not None, "query_embeds_image is None but orientation query is enabled."

            input_ids = inputs["input_ids"]
            image_token_mask = input_ids == self.model.config.image_patch_token

            assert torch.all(image_token_mask[orientation_token_mask]), (
                "All orientation query positions must be image_patch_token positions."
            )

            if obj_images.ndim == 4:
                obj_images = obj_images.unsqueeze(0)
            elif obj_images.ndim != 5:
                raise ValueError(f"Unexpected obj_images shape: {obj_images.shape}")

            if obj_masks.ndim == 1:
                obj_masks = obj_masks.unsqueeze(0)
            elif obj_masks.ndim != 2:
                raise ValueError(f"Unexpected obj_masks shape: {obj_masks.shape}")

            obj_images = obj_images.to(device)
            obj_masks = obj_masks.bool().to(device)

            B, N_obj, C, H, W = obj_images.shape
            assert B == 1, f"metaqueries_generate currently assumes batch size 1, got B={B}"

            num_orient_tokens = orientation_token_mask.sum().item()
            assert num_orient_tokens == B * N_obj, (
                f"orientation_token_mask should contain B*N_obj tokens. "
                f"got {num_orient_tokens}, expected {B * N_obj}."
            )

            inner_orient_mask = torch.masked_select(
                orientation_token_mask,
                image_token_mask,
            ).bool()

            num_all_image_tokens = inner_orient_mask.numel()
            num_normal_image_tokens = (~inner_orient_mask).sum().item()

            assert query_embeds_image.shape[0] == num_normal_image_tokens, (
                f"query_embeds_image length mismatch before orientation expansion: "
                f"query_embeds_image.shape[0]={query_embeds_image.shape[0]}, "
                f"num_normal_image_tokens={num_normal_image_tokens}, "
                f"num_all_image_tokens={num_all_image_tokens}, "
                f"num_orient_tokens={num_orient_tokens}"
            )

            dino_input = obj_images.reshape(B * N_obj, C, H, W)
            dino_device = next(self.dino_model.parameters()).device
            dino_dtype = next(self.dino_model.parameters()).dtype

            dino_input = dino_input.to(device=dino_device, dtype=dino_dtype)

            with torch.no_grad():
                patch_tokens = self.dino_model(dino_input)
            patch_tokens = patch_tokens.reshape(
                B, N_obj, patch_tokens.shape[1], patch_tokens.shape[2]
            )

            patch_tokens = patch_tokens * obj_masks[..., None, None].to(patch_tokens.dtype)
            dino_obj_feat = patch_tokens.mean(dim=2)

            proj_device = next(self.orientation_query_proj.parameters()).device
            proj_dtype = next(self.orientation_query_proj.parameters()).dtype

            dino_obj_feat = dino_obj_feat.to(
                device=proj_device,
                dtype=proj_dtype,
            )

            orientation_query = self.orientation_query_proj(dino_obj_feat)
            orientation_query = orientation_query + self.orientation_token.to(
                device=orientation_query.device,
                dtype=orientation_query.dtype,
            )

            orientation_query_flat = orientation_query.reshape(B * N_obj, -1)

            expanded_query_embeds_image = query_embeds_image.new_zeros(
                num_all_image_tokens,
                query_embeds_image.shape[-1],
            )

            expanded_query_embeds_image[~inner_orient_mask] = query_embeds_image
            expanded_query_embeds_image[inner_orient_mask] = orientation_query_flat.to(
                dtype=query_embeds_image.dtype,
                device=query_embeds_image.device,
            )

            query_embeds_image = expanded_query_embeds_image

        inputs["query_embeds_image"] = query_embeds_image
        inputs["query_embeds_video"] = query_embeds_video
        inputs["image_grid_thw"] = image_grid_thw
        inputs["image_grid_thw_video"] = image_grid_thw_video

        output_str = ""
        new_token_ids = None
        new_query_embeds_images = None
        assert inputs["input_ids"].shape[0] == 1
        new_position_ids = None

        num_remaining_image_gen_token = 0
        curr_image_grid_thw = inputs["image_grid_thw"]
        this_peer_finished = False
        generated_ids = []
        orientation_result = None

        for _ in range(512):
            assert num_remaining_image_gen_token >= 0
            curr_input_ids = torch.cat([inputs["input_ids"], new_token_ids], dim=1) if new_token_ids is not None else inputs["input_ids"]
            assert num_remaining_image_gen_token >= 0
            true_input_ids = curr_input_ids if num_remaining_image_gen_token == 0 else curr_input_ids[:,:-1 * (num_remaining_image_gen_token + 1)]

            curr_query_embeds_image = inputs["query_embeds_image"]
            if new_query_embeds_images is not None:
                if curr_query_embeds_image is None:
                    curr_query_embeds_image = new_query_embeds_images
                else:
                    curr_query_embeds_image = torch.cat([
                        curr_query_embeds_image,
                        new_query_embeds_images
                    ], dim=0)

            curr_position_ids = torch.cat([inputs["position_ids"], new_position_ids], dim=2) if new_position_ids is not None else inputs["position_ids"]
            true_position_ids = curr_position_ids[:,:,:true_input_ids.shape[1]]
            if true_input_ids[0][-1].tolist() == self.model_config.image_start_token:

                assert num_remaining_image_gen_token == 0
                apppended_query_embeds_image, curr_image_grid_thw = append_understand_embeds_with_learnable_scales(
                    clip_feat=curr_query_embeds_image,
                    image_grid_thw=curr_image_grid_thw,
                    scales=self.img_gen_scales,
                    dtype=torch.bfloat16,
                    device=device,
                    feat_dim=self.model.config.hidden_size,
                )

                curr_input_ids, labels, curr_position_ids = append_input_ids_with_learnable_scales(
                    text_ids=true_input_ids,
                    position_ids=true_position_ids,
                    scales=self.img_gen_scales,
                    start_token_id=self.model.config.image_start_token,
                    end_token_id=self.model.config.image_end_token,
                    patch_token_id=self.model.config.image_patch_token,
                )

                learnable_queries_repeat = torch.cat(
                    [self.query_tokens_dict[f"{scale}x{scale}"] for scale in self.img_gen_scales],
                    dim=0,
                )

                image_token_mask = (curr_input_ids == self.model.config.image_patch_token).to(device)
                inner_gen_mask = torch.masked_select(labels, image_token_mask) == self.model.config.image_patch_token
                inner_gen_mask = inner_gen_mask.unsqueeze(-1).expand_as(apppended_query_embeds_image).to(apppended_query_embeds_image.device)

                apppended_query_embeds_image = apppended_query_embeds_image.masked_scatter(
                    inner_gen_mask,
                    learnable_queries_repeat
                )
                assert new_token_ids is None
                new_token_ids = curr_input_ids[:, true_input_ids.shape[1]:]
                new_position_ids = curr_position_ids[:, :, true_position_ids.shape[2]:]
                assert new_query_embeds_images is None
                new_query_embeds_images = apppended_query_embeds_image[curr_query_embeds_image.shape[0]:, :] if curr_query_embeds_image is not None else apppended_query_embeds_image

                continue

            attention_mask = samples["attention_mask"].unsqueeze(0).to(device)

            outputs = self.model(
                input_ids=true_input_ids,
                query_embeds_image=curr_query_embeds_image,
                query_embeds_video=inputs["query_embeds_video"],
                query_embeds_audio=None,
                target_embeds=None,
                position_ids=true_position_ids,
                attention_mask=attention_mask,
                labels=None,
                weights=None,
                image_grid_thw=curr_image_grid_thw,
                image_grid_thw_video=image_grid_thw_video,
            )
            if use_orientation_this_sample and orientation_result is None:
                orient_hidden = outputs.last_hidden_states[orientation_token_mask]
                orient_hidden = orient_hidden.reshape(1, N_obj, -1)

                orient_hidden = self.orientation_head(orient_hidden)

                azimuth_logits = self.orientation_azimuth_head(orient_hidden)
                polar_logits = self.orientation_polar_head(orient_hidden)
                rotation_logits = self.orientation_rotation_head(orient_hidden)

                azimuth_pred = azimuth_logits.argmax(dim=-1) * (360.0 / azimuth_logits.shape[-1])
                polar_pred = polar_logits.argmax(dim=-1) * (180.0 / polar_logits.shape[-1])
                rotation_pred = rotation_logits.argmax(dim=-1) * (360.0 / rotation_logits.shape[-1])

                azimuth_pred = azimuth_pred[0].detach().float().cpu().tolist()
                polar_pred = polar_pred[0].detach().float().cpu().tolist()
                rotation_pred = rotation_pred[0].detach().float().cpu().tolist()

                obj_masks_cpu = obj_masks[0].detach().bool().cpu().tolist()

                orientation_result = []
                for obj_idx in range(N_obj):
                    orientation_result.append(
                        {
                            "obj_idx": obj_idx,
                            "valid": bool(obj_masks_cpu[obj_idx]),
                            "azimuth": float(azimuth_pred[obj_idx]),
                            "polar": float(polar_pred[obj_idx]),
                            "rotation": float(rotation_pred[obj_idx]),
                        }
                    )

                print("orientation_result:", orientation_result)
            if new_query_embeds_images is not None:
                assert labels.shape == true_input_ids.shape
                gen_image_mask = labels == self.model.config.image_patch_token
                assert gen_image_mask.sum().cpu().item() == new_query_embeds_images.shape[0]
                hidden_states_gen = outputs.last_hidden_states[gen_image_mask].view(outputs.last_hidden_states.shape[0], -1, outputs.last_hidden_states.shape[-1])
                assert hidden_states_gen.shape[1] == new_query_embeds_images.shape[0]
                scale_start_idxes = [0] + self.scale_indices[:-1]
                scale_end_idxes = self.scale_indices
                assert scale_end_idxes[-1] == hidden_states_gen.shape[1]
                new_query_embeds_images = {}
                for scale, scale_start_idx, scale_end_idx in zip(self.img_gen_scales, scale_start_idxes, scale_end_idxes):
                    scale_name = f"{scale}x{scale}"
                    scale_hidden = hidden_states_gen[:, scale_start_idx : scale_end_idx, :]

                    scale_embeds = self.proj_in(scale_hidden)
                    seq_shape = scale_embeds.shape
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        scale_embeds = self.connector(
                            inputs_embeds=scale_embeds,
                            attention_mask=torch.ones(seq_shape[0],1,seq_shape[1],seq_shape[1]).to(scale_embeds.device),
                            output_hidden_states=True
                        ).hidden_states[-1]
                    scale_embeds = self.proj_out(scale_embeds)

                    scale_embeds = torch.nn.functional.normalize(scale_embeds, dim=-1)
                    new_query_embeds_images[scale_name] = scale_embeds

                break

            assert num_remaining_image_gen_token == 0
            if num_remaining_image_gen_token > 0:
                pass

            else:
                new_token_id = outputs.logits[:,-1:,:].argmax(dim=-1)
                if not this_peer_finished:
                    if (new_token_id.tolist())[0][0] == self.model.gen_terminator[0]:
                        this_peer_finished = True
                    else:
                        generated_ids.append((new_token_id.tolist())[0][0])
                        new_token_ids = (
                            torch.cat([new_token_ids, new_token_id], dim=1)
                            if new_token_ids is not None else new_token_id
                        )
                if dist.is_available() and dist.is_initialized():
                    finished_flag = torch.tensor(
                        [1 if this_peer_finished else 0], device=device, dtype=torch.long
                    )
                    dist.all_reduce(finished_flag, op=dist.ReduceOp.SUM)
                    if finished_flag.item() == dist.get_world_size():
                        break
                else:
                    if this_peer_finished:
                        break

        output_str = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        multiscale_result = None
        if self.diffusion_loss is not None and new_query_embeds_images is not None:
            imgs = []
            for scale in self.img_gen_scales[-1:]:
                upsampled_images_ref = None
                if ref_images is not None:
                    assert self.diffusion_loss is not None
                    original_size = [height, width]
                    assert original_size[0] % 16 == 0
                    assert original_size[1] % 16 == 0
                    target_size = (original_size[0] // 16 * int(scale), original_size[1] // 16 * int(scale))

                    downsampled_images_ref = torch.nn.functional.interpolate(
                        ref_images.unsqueeze(0),
                        size=target_size,
                        mode='bicubic',
                        align_corners=False,
                        antialias=True,
                    )
                    upsampled_images_ref = torch.nn.functional.interpolate(
                        downsampled_images_ref,
                        size=original_size,
                        mode='bicubic',
                        align_corners=False,
                        antialias=True,
                    )
                extra_vit_input = None
                ref_x = ref_images.unsqueeze(0) if ref_images is not None else None
                imgs.append(self.diffusion_loss.sample(new_query_embeds_images[f"{scale}x{scale}"], steps=steps, seed=seed, cfg=cfg, image_cfg=image_cfg, cfg_mode=cfg_mode, height=height, width=width, ref_x=ref_x, ref_cnocs=ref_cnocs[None, ...] if (ref_cnocs is not None and ref_cnocs.ndim == 3) else ref_cnocs, extra_vit_input=extra_vit_input))

            multiscale_result = concat_horizontal(imgs)
            new_query_embeds_images = imgs[-1]

        if orientation_result is not None:
            return output_str, new_query_embeds_images, multiscale_result, orientation_result
        else:
            return output_str, new_query_embeds_images, multiscale_result
