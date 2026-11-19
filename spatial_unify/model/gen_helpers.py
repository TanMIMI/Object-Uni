import torch
import torch.nn.functional as F
from copy import deepcopy
from PIL import Image

def concat_horizontal(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return new_img

def expand_gen_embeds_as_learnable_scales(
    clip_feat,
    image_grid_thw,
    scales,
    isgen_indicators,
    learnable_queries_1d=True,
):
    resized_clip_feat = []
    new_image_grid_thw = []
    
    assert image_grid_thw.ndim == 2
    bsz = len(image_grid_thw)
    assert clip_feat.ndim == 2
    feat_dim = clip_feat.shape[1]
    n_clip_token_cum = 0
    assert len(isgen_indicators) == bsz
    for bsid in range(bsz):
        thw = image_grid_thw[bsid].tolist()
        assert thw[0] == 1
        assert thw[1] % 2 == 0
        assert thw[2] % 2 == 0
        clip_h = thw[1] // 2
        clip_w = thw[2] // 2
        n_clip_token = clip_h * clip_w
        assert n_clip_token_cum + n_clip_token <= clip_feat.shape[0]
        if isgen_indicators[bsid]:
            for scale in scales:
                clip_feat_one = torch.zeros(scale * scale, feat_dim).to(clip_feat.dtype).to(clip_feat.device)
                resized_clip_feat.append(clip_feat_one)
                if learnable_queries_1d:
                    new_image_grid_thw.append([1, 2, scale * scale * 2])
                else:
                    new_image_grid_thw.append([1, scale * 2, scale * 2])
        else:
            clip_feat_one = clip_feat[n_clip_token_cum : n_clip_token_cum + n_clip_token, :]
            resized_clip_feat.append(clip_feat_one)
            new_image_grid_thw.append(thw)

        n_clip_token_cum += n_clip_token
    
    assert n_clip_token_cum == clip_feat.shape[0]

    encoder_hidden_states = torch.cat(resized_clip_feat, dim=0)
    return encoder_hidden_states, torch.tensor(new_image_grid_thw, dtype=image_grid_thw.dtype).to(image_grid_thw.device)

def append_understand_embeds_with_learnable_scales(
    clip_feat,
    image_grid_thw,
    scales,
    dtype,
    device,
    feat_dim,
    learnable_queries_1d=True,
):
    if clip_feat is not None:
        assert feat_dim == clip_feat.shape[-1]
        assert dtype == clip_feat.dtype
        assert device == clip_feat.device
        assert clip_feat.ndim == 2
    else:
        assert image_grid_thw is None
    
    fake_learnable_embed = torch.zeros(256, feat_dim).to(dtype).to(device)
    clip_feat = torch.cat([clip_feat, fake_learnable_embed], dim=0) if clip_feat is not None else fake_learnable_embed
    fake_image_grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long).to(device)
    image_grid_thw = torch.cat([image_grid_thw.to(device), fake_image_grid_thw], dim=0) if image_grid_thw is not None else fake_image_grid_thw

    return expand_gen_embeds_as_learnable_scales(
        clip_feat,
        image_grid_thw,
        scales,
        isgen_indicators=[False for _ in range(image_grid_thw.shape[0]-1)] + [True],
    )

def expand_gen_input_ids_as_learnable_scales(
    text_ids,
    labels,
    position_ids,
    scales,
    start_token_id,
    end_token_id,
    patch_token_id,
    num_metamorph_learnable_queries,
):
    assert text_ids.ndim == 2
    assert text_ids.shape == labels.shape
    assert position_ids.ndim == 3
    assert position_ids.shape[1] == 2
    assert position_ids[:,0,:].shape == text_ids.shape


    default_scaled_tokens = []
    for scale in scales:
        default_scaled_tokens.append(start_token_id)
        default_scaled_tokens.extend([patch_token_id for _ in range(scale * scale)])
        default_scaled_tokens.append(end_token_id)
    
    text_ids_list = text_ids.cpu().tolist()
    labels_list = labels.cpu().tolist()
    position_ids_0_list = position_ids[:,0,:].cpu().tolist()
    position_ids_1_list = position_ids[:,1,:].cpu().tolist()

    new_text_ids_list = []
    new_labels_list = []
    new_position_ids_0_list = []
    new_position_ids_1_list = []
    for text_ids_one_batch, labels_one_batch, position_ids_0_one_batch, position_ids_1_one_batch in zip(
        text_ids_list, labels_list, position_ids_0_list, position_ids_1_list
    ):
        assert len(text_ids_one_batch) == len(labels_one_batch)
        assert len(text_ids_one_batch) == len(position_ids_0_one_batch)
        assert len(text_ids_one_batch) == len(position_ids_1_one_batch)
        start_idx = [i for i, j  in enumerate(labels_one_batch) if j == start_token_id]
        end_idx = [i for i, j in enumerate(labels_one_batch) if j == end_token_id]
        assert len(start_idx) == 1, start_idx
        assert len(end_idx) == 1, end_idx
        start_idx = start_idx[0]
        end_idx = end_idx[0]
        assert end_idx - start_idx == num_metamorph_learnable_queries + 1, (start_idx, end_idx)
        assert text_ids_one_batch[start_idx] == start_token_id and text_ids_one_batch[end_idx] == end_token_id
        text_ids_one_batch[start_idx: end_idx+1] = deepcopy(default_scaled_tokens)
        labels_one_batch[start_idx: end_idx+1] = deepcopy(default_scaled_tokens)

        position_ids_0_one_batch[start_idx:] = [i + position_ids_0_one_batch[start_idx] for i in range(len(position_ids_0_one_batch) + len(default_scaled_tokens) - (end_idx + 1 - start_idx) - start_idx)]
        assert position_ids_1_one_batch[start_idx] > 1
        position_ids_1_one_batch[start_idx:] = [i + position_ids_1_one_batch[start_idx] for i in range(len(position_ids_1_one_batch) + len(default_scaled_tokens) - (end_idx + 1 - start_idx) - start_idx)]
        assert len(position_ids_0_one_batch) == len(text_ids_one_batch)
        assert len(position_ids_1_one_batch) == len(text_ids_one_batch)

        new_text_ids_list.append(text_ids_one_batch)     
        new_labels_list.append(labels_one_batch) 

        new_position_ids_0_list.append(position_ids_0_one_batch)  
        new_position_ids_1_list.append(position_ids_1_one_batch)    

    return (
        torch.tensor(new_text_ids_list, dtype=text_ids.dtype).to(text_ids.device),
        torch.tensor(new_labels_list, dtype=labels.dtype).to(labels.device), 
        torch.cat(
            [
                torch.tensor(new_position_ids_0_list, dtype=labels.dtype).to(labels.device).unsqueeze(1),
                torch.tensor(new_position_ids_1_list, dtype=labels.dtype).to(labels.device).unsqueeze(1),
            ],
            dim=1
        ),
    )


def append_input_ids_with_learnable_scales(    
    text_ids,
    position_ids,
    scales,
    start_token_id,
    end_token_id,
    patch_token_id,
):
    assert text_ids.shape[0] == 1
    assert text_ids[0][-1].tolist() == start_token_id
    assert text_ids.shape[-1] == position_ids.shape[-1]
    

    labels = torch.cat([
        torch.ones_like(text_ids[:,:-1]) * 0 - 100, 
        torch.tensor([[start_token_id, patch_token_id, end_token_id]]).to(text_ids.dtype).to(text_ids.device),
    ], dim=1)
    
    text_ids = torch.cat([
        text_ids, 
        torch.tensor([[patch_token_id, end_token_id]]).to(text_ids.dtype).to(text_ids.device),
    ], dim=1)

    assert labels.shape == text_ids.shape

    num_added = text_ids.shape[-1] - position_ids.shape[-1]
    position_ids = torch.cat([
        position_ids, 
        torch.cat([(torch.arange(num_added).to(position_ids.device) + 1 + position_ids[0][0][-1]).unsqueeze(0).unsqueeze(0), (torch.arange(num_added).to(position_ids.device) + 1 + position_ids[0][1][-1]).unsqueeze(0).unsqueeze(0)], dim=1)
    ], dim=2)
    
    
    text_ids, labels, position_ids = expand_gen_input_ids_as_learnable_scales(
        text_ids,
        labels,
        position_ids,
        scales,
        start_token_id,
        end_token_id,
        patch_token_id,
        num_metamorph_learnable_queries=1,
    )
    return text_ids, labels, position_ids

def interpolate_imgfeat_to_256tokens(clip_feat, image_grid_thw, isgen_indicators=None, learnable_queries_1d=True, input_resize_256=False):
    resized_clip_feat_input = []
    resized_clip_feat = []
    assert image_grid_thw.ndim == 2
    bsz = len(image_grid_thw)
    assert clip_feat.ndim == 2
    n_clip_token_cum = 0
    new_image_grid_thw_input = []
    new_image_grid_thw = []
    for bsid in range(bsz):
        thw = image_grid_thw[bsid].tolist()
        assert thw[0] == 1
        assert thw[1] % 2 == 0
        assert thw[2] % 2 == 0
        clip_h = thw[1] // 2
        clip_w = thw[2] // 2
        n_clip_token = clip_h * clip_w
        assert n_clip_token_cum + n_clip_token <= clip_feat.shape[0]
        clip_feat_one = clip_feat[n_clip_token_cum : n_clip_token_cum + n_clip_token, :]
        n_clip_token_cum += n_clip_token
        if isgen_indicators is None or isgen_indicators[bsid]:
            feat_dim = clip_feat_one.shape[1]
            clip_feat_one = clip_feat_one.permute(1,0).view(feat_dim, clip_h, clip_w).unsqueeze(0)
            assert clip_feat_one.ndim == 4
            reshape_h, reshape_w = 16, 16
            clip_feat_one = torch.nn.functional.interpolate(
                clip_feat_one,
                size=(reshape_h, reshape_w), 
                mode='bilinear', 
                align_corners=False
            )
            clip_feat_one = clip_feat_one.squeeze(0).view(feat_dim, reshape_h * reshape_w).permute(1,0)
            resized_clip_feat.append(clip_feat_one)
            if learnable_queries_1d:
                new_image_grid_thw.append([1, 1 * 2, 16 * 16 * 2])
            else:
                new_image_grid_thw.append([1, 16 * 2, 16 * 2])
        else:
            resized_clip_feat.append(clip_feat_one)
            new_image_grid_thw.append(thw)
            resized_clip_feat_input.append(clip_feat_one)
            new_image_grid_thw_input.append(thw)

    assert n_clip_token_cum == clip_feat.shape[0]
    encoder_hidden_states = torch.cat(resized_clip_feat, dim=0)

    batch_resized_clip_feat_input = resized_clip_feat_input*1
    
    if len(batch_resized_clip_feat_input) > 0:
        if input_resize_256:
            for index, (clip_feat_one, thw) in enumerate(zip(batch_resized_clip_feat_input, new_image_grid_thw_input)):
                clip_h = thw[1] // 2
                clip_w = thw[2] // 2
                feat_dim = clip_feat_one.shape[1]
                clip_feat_one = clip_feat_one.permute(1,0).view(feat_dim, clip_h, clip_w).unsqueeze(0)
                clip_feat_one = torch.nn.functional.interpolate(clip_feat_one,size=(16, 16), mode='bilinear', align_corners=False)
                clip_feat_one = clip_feat_one.squeeze(0).view(feat_dim, 16 * 16).permute(1,0)
                batch_resized_clip_feat_input[index] = clip_feat_one

        from torch.nn.utils.rnn import pad_sequence
        unbatched = [t.squeeze(0) for t in batch_resized_clip_feat_input]

        lengths = [t.shape[0] for t in unbatched]

        batch_resized_clip_feat_input = pad_sequence(unbatched, batch_first=True, padding_value=0).to(clip_feat.dtype)

        batch_clip_feat_input_attention_mask = torch.ones_like(batch_resized_clip_feat_input[:, :, 0], dtype=torch.bool)

        for i, length in enumerate(lengths):
            batch_clip_feat_input_attention_mask[i, length:] = False

    else:
        batch_resized_clip_feat_input = None
        batch_clip_feat_input_attention_mask = None

    if len(resized_clip_feat_input) > 0:
        resized_clip_feat_input = torch.cat(resized_clip_feat_input, dim=0)
        new_image_grid_thw_input = torch.tensor(new_image_grid_thw_input).to(image_grid_thw.dtype).to(image_grid_thw.device)
    return encoder_hidden_states, torch.tensor(new_image_grid_thw).to(image_grid_thw.dtype).to(image_grid_thw.device), resized_clip_feat_input, new_image_grid_thw_input, batch_resized_clip_feat_input, batch_clip_feat_input_attention_mask
