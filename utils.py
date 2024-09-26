from glob import glob
import os
import pickle
import math
import random

from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import cv2

#### TikTok >>>>
TIKTOK_VALIDATION_PATHS = {
    "source_image": [
        './DATA/TikTok/valid_video/00337/images.mp4', 
        './DATA/TikTok/valid_video/00338/images.mp4', 
        './DATA/TikTok/valid_video/201_002_1x1/images.mp4', 
        './DATA/TikTok/valid_video/201_005_1x1/images.mp4', 
        './DATA/TikTok/valid_video/201_021_1x1/images.mp4', 
        './DATA/TikTok/valid_video/201_024_1x1/images.mp4', 
        './DATA/TikTok/valid_video/202_006_1x1/images.mp4', 
        './DATA/TikTok/valid_video/202_007_1x1/images.mp4', 
        './DATA/TikTok/valid_video/202_025_1x1/images.mp4', 
        './DATA/TikTok/valid_video/203_006_1x1/images.mp4'
    ],
    "video_path": [
        './DATA/TikTok/valid_video/00337/dwpose.mp4', 
        './DATA/TikTok/valid_video/00338/dwpose.mp4', 
        './DATA/TikTok/valid_video/201_002_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/201_005_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/201_021_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/201_024_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/202_006_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/202_007_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/202_025_1x1/dwpose.mp4', 
        './DATA/TikTok/valid_video/203_006_1x1/dwpose.mp4'
    ],
    
}
#### TikTok <<<<

def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])
    
def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping

def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping

def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]

def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False, controlnet=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())

    if controlnet:
        unet_key = "control_model."
    else:
        unet_key = "model.diffusion_model."

    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        print(
            "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
            " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
        )
        for key in keys:
            if key.startswith("model.diffusion_model"):
                flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
    else:
        if sum(k.startswith("model_ema") for k in keys) > 100:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )

        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    if config["class_embed_type"] is None:
        # No parameters to port
        ...
    elif config["class_embed_type"] == "timestep" or config["class_embed_type"] == "projection":
        new_checkpoint["class_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]
        new_checkpoint["class_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]
        new_checkpoint["class_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]
        new_checkpoint["class_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]
    else:
        raise NotImplementedError(f"Not implemented `class_embed_type`: {config['class_embed_type']}")

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    if not controlnet:
        new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
        new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
        new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
        new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    if controlnet:
        # conditioning embedding

        orig_index = 0

        new_checkpoint["controlnet_cond_embedding.conv_in.weight"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.weight"
        )
        new_checkpoint["controlnet_cond_embedding.conv_in.bias"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.bias"
        )

        orig_index += 2

        diffusers_index = 0

        while diffusers_index < 6:
            new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_index}.weight"] = unet_state_dict.pop(
                f"input_hint_block.{orig_index}.weight"
            )
            new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_index}.bias"] = unet_state_dict.pop(
                f"input_hint_block.{orig_index}.bias"
            )
            diffusers_index += 1
            orig_index += 2

        new_checkpoint["controlnet_cond_embedding.conv_out.weight"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.weight"
        )
        new_checkpoint["controlnet_cond_embedding.conv_out.bias"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.bias"
        )

        # down blocks
        for i in range(num_input_blocks):
            new_checkpoint[f"controlnet_down_blocks.{i}.weight"] = unet_state_dict.pop(f"zero_convs.{i}.0.weight")
            new_checkpoint[f"controlnet_down_blocks.{i}.bias"] = unet_state_dict.pop(f"zero_convs.{i}.0.bias")

        # mid block
        new_checkpoint["controlnet_mid_block.weight"] = unet_state_dict.pop("middle_block_out.0.weight")
        new_checkpoint["controlnet_mid_block.bias"] = unet_state_dict.pop("middle_block_out.0.bias")

    return new_checkpoint

#### pose-retargeted >>>> 
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
            [1, 16], [16, 18], [3, 17], [6, 18]]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
handedges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
sub_nodes_dict = {
    0: [14,15,16,17],
    1: [],  # fixed point
    2: [3,4],
    3: [4],
    4: [],
    5: [6,7],
    6: [7],
    7: [],
    8: [9,10],
    9: [10],
    10: [],
    11: [12,13],
    12: [13],
    13: [],
    14: [16],
    15: [17],
    16: [],
    17: [],
}
stage_nodes_dict = {
    1: [2,5,8,11,14,15],
    2: [3,6,9,12,16,17],
    3: [4,7,10,13]
}
new_limbseq_dict = {
    0: [1,0],
    1: [0,1],
    2: [1,2],
    3: [2,3],
    4: [3,4],
    5: [1,5],
    6: [5,6],
    7: [6,7],
    8: [1,8],
    9: [8,9],
    10: [9,10],
    11: [1,11],
    12: [11,12],
    13: [12,13],
    14: [0,14],
    15: [0,15],
    16: [14,16],
    17: [15,17],
}

def pkl_read(p):
    with open(p, "rb") as f:
        pkl_file = pickle.load(f)
    return pkl_file["bodies"]["candidate"], pkl_file["bodies"]["subset"], pkl_file["hands"]
def draw_bodyhand(cand, subset, img, eps=0.01, stickwidth=4, hands=None, verbose=True):
    img_h, img_w = img.shape[:2]
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # draw body
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = cand[index.astype(int), 0] * float(img_w)
            X = cand[index.astype(int), 1] * float(img_h)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
            
    canvas = (canvas * 0.6).astype(np.uint8)
    
    for i in range(18):
        for n in range(len(subset)):
            idx = int(subset[n][i])

            if idx == -1: continue
            x,y = cand[idx][0:2]
            xx = int(x*img_w)
            yy = int(y*img_h)
            if x > eps and y > eps:
                # cv2.putText(canvas, f"{i}", (xx,yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.circle(canvas, (xx,yy), 4, colors[i], thickness=-1)
            
    # draw hands
    if hands is not None:
        for hand in hands:
            hand = np.array(hand)

            for ie, e in enumerate(handedges):
                x1, y1 = hand[e[0]]
                x2, y2 = hand[e[1]]
                x1 = int(x1 * img_w)
                y1 = int(y1 * img_h)
                x2 = int(x2 * img_w)
                y2 = int(y2 * img_h)
                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    cv2.line(canvas, (x1, y1), (x2, y2), 
                             matplotlib.colors.hsv_to_rgb([ie / float(len(handedges)), 1.0, 1.0]) * 255, thickness=2)        

            for i, keypoint in enumerate(hand):
                x, y = keypoint
                x = int(x * img_w)
                y = int(y * img_h)
                if x > eps and y > eps:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    show_img = np.concatenate([img[:,:,::-1], canvas], axis=1)
    if verbose:
        plt.figure(figsize=(5,5))
        plt.imshow(show_img)
        plt.show()
    return canvas
    
def get_length(a,b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
def get_relative_direction_vector(start_pnt, end_pnt, ratio):
    return ((end_pnt - start_pnt) * (ratio-1)).astype(np.int64)
  
# function for chibi benchmark. 
# align the length of symmetric points, align the positions with chibi image.
# src: human, trg: chibi
def relative_balanced_pose_transform(src_cand, src_subset, src_img, trg_cand, trg_subset, trg_img, eps=0.01, ff_src_cand=None, ff_src_subset=None, 
                                     use_neck_move=False):
    src_h, src_w = src_img.shape[:2]
    trg_h, trg_w = trg_img.shape[:2]
    size_ratio = math.sqrt(src_h * src_w) / math.sqrt(trg_h * trg_w)
    src_cand_unnorm = np.stack([src_cand[:,0] * src_w, src_cand[:,1] * src_h], axis=1)
    trg_cand_unnorm = np.stack([trg_cand[:,0] * trg_w, trg_cand[:,1] * trg_h], axis=1)
    src_length_ratios = [-1 for _ in range(18)]
    trg_length_ratios = [-1 for _ in range(18)]
    src_pivot_length = get_length(src_cand_unnorm[0], src_cand_unnorm[1])
    trg_pivot_length = get_length(trg_cand_unnorm[0], trg_cand_unnorm[1])

    # 첫번쨰 프레임의 길이와 현재 프레임의 비율 구하기, 만약 측면이나 팔이 굽혀질 경우 팔이 짧게 표시되어야 하는데 chibi의 절대 비율만 맞추면
    # 일정한 길이로만 나타나게 됨.
    if ff_src_cand is not None and ff_src_subset is not None:
        ffandcur_length_ratio = [-1 for _ in range(18)]
        ff_src_cand_unnorm = np.stack([ff_src_cand[:,0] * src_w, ff_src_cand[:,1] * src_h], axis=1)
        for i in range(2, 18):
            start_node, end_node = new_limbseq_dict[i]
            ff_length = get_length(ff_src_cand_unnorm[start_node], ff_src_cand_unnorm[end_node])
            cur_length = get_length(src_cand_unnorm[start_node], src_cand_unnorm[end_node])
            ffandcur_length_ratio[i] = np.clip(cur_length / ff_length, 0, 1.1)

    # 0,1길이도 변경, 목이 너무 길어져서 목도 추가하자
    node = 0
    pivot_ratio = trg_pivot_length / src_pivot_length 

    # 첫번째 프레임과 현재 프레임의 목 길이를 반영해야함. 
    if use_neck_move:
        relative_neck_vector = [(src_cand[1][0] - ff_src_cand[1][0]) * pivot_ratio, (src_cand[1][1] - ff_src_cand[1][1]) * pivot_ratio] 
    start_node, end_node = new_limbseq_dict[node]
    assert node == end_node
    dir_vector = get_relative_direction_vector(src_cand_unnorm[start_node], src_cand_unnorm[end_node], pivot_ratio)
    src_cand_unnorm[end_node] += dir_vector
    for sub_node in sub_nodes_dict[node]:
        src_cand_unnorm[sub_node] += dir_vector

    src_pivot_length = get_length(src_cand_unnorm[0], src_cand_unnorm[1])    
    
    for i in range(2,18):
        start_node, end_node = new_limbseq_dict[i]
        if src_subset[0][i] != -1:
            src_length_ratios[i] = get_length(src_cand_unnorm[start_node], src_cand_unnorm[end_node]) / src_pivot_length
        if trg_subset[0][i] != -1:
            trg_length_ratios[i] = get_length(trg_cand_unnorm[start_node], trg_cand_unnorm[end_node]) / trg_pivot_length
            
    #### 좌우 대칭인 점들은 최댓값
    trg_length_ratios[2] = trg_length_ratios[5] = max(trg_length_ratios[2], trg_length_ratios[5])
    trg_length_ratios[3] = trg_length_ratios[6] = max(trg_length_ratios[3], trg_length_ratios[6])
    trg_length_ratios[4] = trg_length_ratios[7] = max(trg_length_ratios[4], trg_length_ratios[7])
    trg_length_ratios[8] = trg_length_ratios[11] = max(trg_length_ratios[8], trg_length_ratios[11])
    trg_length_ratios[9] = trg_length_ratios[12] = max(trg_length_ratios[9], trg_length_ratios[12])
    trg_length_ratios[10] = trg_length_ratios[13] = max(trg_length_ratios[10], trg_length_ratios[13])
    trg_length_ratios[14] = trg_length_ratios[15] = max(trg_length_ratios[14], trg_length_ratios[15])
    trg_length_ratios[16] = trg_length_ratios[17] = max(trg_length_ratios[16], trg_length_ratios[17])  
    
    for stage, nodes in stage_nodes_dict.items():
        for node in nodes:  # 점 이동
            src_ratio = src_length_ratios[node]
            trg_ratio = trg_length_ratios[node]
            if src_ratio < eps*10 or trg_ratio < eps*10: continue
            if ff_src_cand is not None and ff_src_subset is not None:
                ratio = trg_ratio / src_ratio * ffandcur_length_ratio[node]
            else:
                ratio = trg_ratio / src_ratio
            start_node, end_node = new_limbseq_dict[node]
            assert node == end_node
            dir_vector = get_relative_direction_vector(src_cand_unnorm[start_node], src_cand_unnorm[end_node], ratio)
            src_cand_unnorm[end_node] += dir_vector
            for sub_node in sub_nodes_dict[node]:  # 딸린 점 모두 이동
                src_cand_unnorm[sub_node] += dir_vector      
    
    t_src_cand = np.stack([src_cand_unnorm[:,0] / trg_w, src_cand_unnorm[:, 1] / trg_h], axis=1)

    #### 최종 완성본을 평행이동하여 원위치로 복원 + 첫번째 프레임의 목 위치 반영
    if use_neck_move:
        parallel_move_ratio = trg_cand[1] - t_src_cand[1] + relative_neck_vector
    else:
        parallel_move_ratio = trg_cand[1] - t_src_cand[1]
    # parallel_move_ratio = trg_cand[1]
    for i in range(len(t_src_cand)):
        t_src_cand[i] += parallel_move_ratio
        
    return t_src_cand

def get_random_balanced_length_ratio():
    '''
    These numbers are used as ratio in random_pose_transform. 
    How much we need to scale to be the target? 
    Ratio in animation1 is as follows.
    2 0.5666588663771651
    5 0.5823707482298922
    8 0.607724718161831
    11 0.6187976093003016
    14 2.0292875704236604
    15 1.4537215238353585
    3 0.4828631371694108
    6 0.40768133578443255
    16 3.1272029979408265
    17 1.754926104935595
    4 0.7418691586412757
    7 1.023349633451018
    '''
    shoulder_ratio = random.uniform(0.4,0.7)
    elbow_ratio = random.uniform(0.35,0.65)
    hand_ratio = elbow_ratio + random.uniform(-0.1,0.1)
    hip_ratio = random.uniform(0.5,0.7)
    knee_ratio = random.uniform(0.4,0.6)
    foot_ratio = random.uniform(0.4,0.6)
    eye_ratio = random.uniform(1.5,1.9)
    ear_ratio = eye_ratio + random.uniform(0.5,0.9)
    length_ratios = [
        -1,  # 0, root node : nose
        -1,  # 1, root node : neck
        shoulder_ratio,  # 2, left shoulder
        elbow_ratio,  # 3, left elbow
        hand_ratio,  # 4, left hand
        shoulder_ratio,  # 5, right shoulder
        elbow_ratio,  # 6, right elbow
        hand_ratio,  # 7, right hand
        hip_ratio,  # 8, left hip
        knee_ratio,  # 9, left knee
        foot_ratio,  # 10, left foot
        hip_ratio,  # 11, right hip
        knee_ratio,  # 12, right knee
        foot_ratio,  # 13, right foot
        eye_ratio,  # 14, left eye
        eye_ratio,  # 15, right eye
        ear_ratio,  # 16, left ear
        ear_ratio,  # 17, right ear
    ]  # jho) TODO   
    return length_ratios
def random_pose_transform(src_cand, src_subset, src_img, trg_length_ratios, eps=0.01):
    # original length multiplied by ratio. 
    src_h, src_w = src_img.shape[:2]
    src_cand_unnorm = np.stack([src_cand[:,0] * src_w, src_cand[:,1] * src_h], axis=1)
    src_length_ratios = [-1 for _ in range(18)]
    src_pivot_length = get_length(src_cand_unnorm[0], src_cand_unnorm[1])
    for i in range(2,18):
        start_node, end_node = new_limbseq_dict[i]
        if src_subset[0][i] != -1:
            src_length_ratios[i] = get_length(src_cand_unnorm[start_node], src_cand_unnorm[end_node]) / src_pivot_length
            
    for stage, nodes in stage_nodes_dict.items():
        for node in nodes:  # 점 이동
            src_ratio = src_length_ratios[node]
            trg_ratio = trg_length_ratios[node]
            if src_ratio < eps*10 or trg_ratio < eps*10: continue
            ratio = trg_length_ratios[node]
            start_node, end_node = new_limbseq_dict[node]
            assert node == end_node
            dir_vector = get_relative_direction_vector(src_cand_unnorm[start_node], src_cand_unnorm[end_node], ratio)
            src_cand_unnorm[end_node] += dir_vector
            for sub_node in sub_nodes_dict[node]:  # 딸린 점 모두 이동
                src_cand_unnorm[sub_node] += dir_vector
        
    t_src_cand = np.stack([src_cand_unnorm[:,0] / src_w, src_cand_unnorm[:, 1] / src_h], axis=1)
    return t_src_cand

def pose_base_crop(img, posedict_p):
    cand, subset, _ = pkl_read(posedict_p)
    indices = [4,7,8,11]
    if np.all(subset[0][indices] == -1):
        return img
    img = np.array(img)
    img_h, img_w = img.shape[:2]
    y_lst = []
    for idx in indices:
        if 0.1 < cand[idx][1] < 0.9:
            y_lst.append(cand[idx][1])
    
    max_y = max(y_lst)
    max_y = max(0, max_y + 0.1)
    coord_y = int(max_y * img_h)
    img = Image.fromarray(img[:coord_y])
    return img

def get_retargeted(
    src_img, 
    src_pkl_p, 
    dri_img, 
    dri_pkl_p, 
    bbox, 
    first_dri_pkl_p,
    pose_crop=False,
):
    src_img_h, src_img_w = src_img.shape[:2]
    src_cand, src_subset, _ = pkl_read(src_pkl_p)
    src_pose = draw_bodyhand(src_cand, src_subset, src_img, verbose=False)

    dri_cand, dri_subset, _ = pkl_read(dri_pkl_p)

    first_dri_cand, first_dri_subset, _ = pkl_read(first_dri_pkl_p)
    t_dri_cand = relative_balanced_pose_transform(
        dri_cand, dri_subset, dri_img,
        src_cand, src_subset, src_img,
        ff_src_cand=first_dri_cand, 
        ff_src_subset=first_dri_subset,
        use_neck_move=True
    )
    if not pose_crop:
        rx, ry, rw, rh = bbox
        x1 = int(rx*src_img_w)
        x2 = int((rx+rw)*src_img_w)
        y1 = int(ry*src_img_h)
        y2 = int((ry+rh)*src_img_h)
        crop_src_img = src_img[y1:y2, x1:x2]
        crop_src_pose = src_pose[y1:y2, x1:x2]
        crop_t_dri_pose = draw_bodyhand(t_dri_cand, dri_subset, src_img, verbose=False)[y1:y2, x1:x2]
    else:
        crop_src_img = pose_base_crop(src_img, src_pkl_p)
        crop_src_pose = pose_base_crop(src_pose, src_pkl_p)
        crop_t_dri_pose = draw_bodyhand(t_dri_cand, dri_subset, src_img, verbose=False)
        crop_t_dri_pose = pose_base_crop(crop_t_dri_pose, src_pkl_p)

        crop_src_img = np.array(crop_src_img)
        crop_src_pose = np.array(crop_src_pose)
        crop_t_dri_pose = np.array(crop_t_dri_pose)

    return crop_src_img, crop_src_pose, crop_t_dri_pose
#### pose retargeted <<<<