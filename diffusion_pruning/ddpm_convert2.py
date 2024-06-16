from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
import argparse, os, torch, json

parser = argparse.ArgumentParser()
parser.add_argument("--total_samples", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="samples")
parser.add_argument("--model_path", type=str, default="samples")
parser.add_argument("--ddim_steps", type=int, default=100)
parser.add_argument("--pruned_model_ckpt", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--skip_type", type=str, default="uniform")
parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the architecture.",
    )


args = parser.parse_args()

def reverse_renew_resnet_paths(new_list, n_shave_prefix_segments=0):
    """Reverses the process of renewing ResNet paths."""
    mapping = []
    for new_item in new_list:
        old_item = new_item
        old_item = old_item.replace("resnets.", "block.")
        old_item = old_item.replace("conv_shortcut", "nin_shortcut")
        #old_item = old_item.replace("conv1", "conv_shorcut")
        old_item = old_item.replace("time_emb_proj", "temb_proj")

        old_item = shave_segments(old_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"new": new_item, "old": old_item})

    return mapping

def add_segments(path, n_add_prefix_segments=1):
    """
    Adds segments. Positive values add at the beginning, negative add at the end.
    """
    if n_add_prefix_segments >= 0:
        return ".".join([""]*n_add_prefix_segments + path.split("."))
    else:
        return ".".join(path.split(".") + [""]*(-n_add_prefix_segments))

def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])
    

def reverse_renew_attention_paths(new_list, n_shave_prefix_segments=0, in_mid=False):
    """Reverses the process of renewing attention paths."""
    mapping = []

    for new_item in new_list:
        old_item = new_item

        # In `model.mid`, the layer is called `attn`.
        if not in_mid: 
            old_item = old_item.replace("attentions", "attn") 
        old_item = old_item.replace(".to_k.", ".k.")
        old_item = old_item.replace(".to_v.", ".v.")
        old_item = old_item.replace(".to_q.", ".q.")

        old_item = old_item.replace("to_out.0", "proj_out")

        old_item = old_item.replace("group_norm", "norm")

        old_item = shave_segments(old_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"new": new_item, "old": old_item})

    return mapping


def reverse_assign_to_checkpoint(
    mapping, new_checkpoint, original_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """Reverses the process of assigning to checkpoint."""
    for item in mapping:
        old_path = item["old"]
        new_path = item["new"]
        if new_path in new_checkpoint:
            # Reverse the key transformations
            old_path = old_path.replace("down_blocks.", "down.")
            old_path = old_path.replace("up_blocks.", "up.")
            if additional_replacements:
                for replacement in additional_replacements:
                    old_path = old_path.replace(replacement["old"], replacement["new"])
            if "attentions" in old_path:
                if attention_paths_to_split is not None and old_path in attention_paths_to_split:
                    old_tensor = torch.cat(
                        [new_checkpoint[attention_paths_to_split[old_path]["query"]],
                         new_checkpoint[attention_paths_to_split[old_path]["key"]],
                         new_checkpoint[attention_paths_to_split[old_path]["value"]]], dim=1
                    )
                    original_checkpoint[old_path] = old_tensor
                else:
                    original_checkpoint[old_path] = new_checkpoint[new_path].unsqueeze(0)
            else:
                original_checkpoint[old_path] = new_checkpoint[new_path]
    return original_checkpoint

def reverse_initialize_new_checkpoint(new_checkpoint, original_checkpoint):
    """Reverses the process of initializing new checkpoint."""
    # original_checkpoint = {}

    original_checkpoint["temb.dense.0.weight"] = new_checkpoint["time_embedding.linear_1.weight"]
    original_checkpoint["temb.dense.0.bias"] = new_checkpoint["time_embedding.linear_1.bias"]
    original_checkpoint["temb.dense.1.weight"] = new_checkpoint["time_embedding.linear_2.weight"]
    original_checkpoint["temb.dense.1.bias"] = new_checkpoint["time_embedding.linear_2.bias"]
    original_checkpoint["norm_out.weight"] = new_checkpoint["conv_norm_out.weight"]
    original_checkpoint["norm_out.bias"] = new_checkpoint["conv_norm_out.bias"]
    original_checkpoint["conv_in.weight"] = new_checkpoint["conv_in.weight"]
    original_checkpoint["conv_in.bias"] = new_checkpoint["conv_in.bias"]
    original_checkpoint["conv_out.weight"] = new_checkpoint["conv_out.weight"]
    original_checkpoint["conv_out.bias"] = new_checkpoint["conv_out.bias"]

    return original_checkpoint


def reverse_create_down_blocks(new_checkpoint, original_checkpoint, config):
    """Reverses the process of creating down_blocks dictionaries."""
    num_down_blocks = len({".".join(layer.split(".")[:2]) for layer in new_checkpoint if "down_blocks" in layer})
    down_blocks = {layer_id: [key for key in new_checkpoint if f"down_blocks.{layer_id}" in key] for layer_id in range(num_down_blocks)}

    for i in range(num_down_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)

        if any("downsamplers" in layer for layer in down_blocks[i]):
            original_checkpoint[f"down.{i}.downsample.conv.weight"] = new_checkpoint[f"down_blocks.{i}.downsamplers.0.conv.weight"]
            original_checkpoint[f"down.{i}.downsample.conv.bias"] = new_checkpoint[f"down_blocks.{i}.downsamplers.0.conv.bias"]

        if any("resnets" in layer for layer in down_blocks[i]):
            num_blocks = len({".".join(layer.split(".")[:2]) for layer in down_blocks[i] if "resnets" in layer})
            num_blocks = len({".".join(shave_segments(layer, 2).split(".")[:2]) for layer in down_blocks[i] if "resnets" in layer})
            blocks = {layer_id: [key for key in down_blocks[i] if f"resnets.{layer_id}" in key] for layer_id in range(num_blocks)}

            if num_blocks > 0:
                for j in range(config["layers_per_block"]):
                    paths = reverse_renew_resnet_paths(blocks[j])
                    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint)

        if any("attentions" in layer for layer in down_blocks[i]):
            num_attn = len({".".join(layer.split(".")[:2]) for layer in down_blocks[i] if "attentions" in layer})
            num_attn = len({".".join(shave_segments(layer, 2).split(".")[:2]) for layer in down_blocks[i] if "attentions" in layer})
            attns = {layer_id: [key for key in down_blocks[i] if f"attentions.{layer_id}" in key] for layer_id in range(num_attn)}
            
            if num_attn > 0:
                for j in range(config["layers_per_block"]):
                    paths = reverse_renew_attention_paths(attns[j])
                    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint, config=config)

    return original_checkpoint

def reverse_create_up_blocks(new_checkpoint, original_checkpoint, config):
    """Reverses the process of creating up_blocks dictionaries."""
    num_up_blocks = len({".".join(layer.split(".")[:2]) for layer in new_checkpoint if "up_blocks" in layer})
    up_blocks = {layer_id: [key for key in new_checkpoint if f"up_blocks.{layer_id}" in key] for layer_id in range(num_up_blocks)}

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i

        if any("upsamplers" in layer for layer in up_blocks[block_id]):
            original_checkpoint[f"up.{i}.upsample.conv.weight"] = new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"]
            original_checkpoint[f"up.{i}.upsample.conv.bias"] = new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"]

        if any("resnets" in layer for layer in up_blocks[block_id]):
            num_blocks = len({".".join(layer.split(".")[:2]) for layer in up_blocks[block_id] if "resnets" in layer})
            num_blocks = len({".".join(shave_segments(layer, 2).split(".")[:2]) for layer in up_blocks[block_id] if "resnets" in layer})

            blocks = {layer_id: [key for key in up_blocks[block_id] if f"resnets.{layer_id}" in key] for layer_id in range(num_blocks)}

            if num_blocks > 0:
                for j in range(config["layers_per_block"] + 1):
                    replace_indices = {"old": f"up.{block_id}", "new": f"up.{i}"}
                    paths = reverse_renew_resnet_paths(blocks[j])
                    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint, additional_replacements=[replace_indices])

        if any("attentions" in layer for layer in up_blocks[block_id]):
            num_attn = len({".".join(layer.split(".")[:2]) for layer in up_blocks[block_id] if "attentions" in layer})
            num_attn = len({".".join(shave_segments(layer, 2).split(".")[:2]) for layer in up_blocks[block_id] if "attentions" in layer})
            attns = {layer_id: [key for key in up_blocks[block_id] if f"attentions.{layer_id}" in key] for layer_id in range(num_attn)}
            
            if num_attn > 0:
                for j in range(config["layers_per_block"] +1 ):
                    replace_indices = {"old": f"up.{block_id}", "new": f"up.{i}"}
                    paths = reverse_renew_attention_paths(attns[j])
                    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint, config=config,
                                                 additional_replacements=[replace_indices])

    return original_checkpoint

def reverse_handle_remaining_logic(new_checkpoint, original_checkpoint, config):
    """Reverses the process of handling remaining logic."""
    new_checkpoint = {k.replace("mid_block", "mid_new_2"): v for k, v in new_checkpoint.items()}
    new_checkpoint = {k.replace("conv_shortcut", "nconv_shortcut"): v for k, v in new_checkpoint.items()}

    mid_new_2_resnets_0_layers = [key for key in new_checkpoint if "mid_new_2.resnets.0" in key]
    mid_new_2_resnets_1_layers = [key for key in new_checkpoint if "mid_new_2.resnets.1" in key]
    mid_new_2_attentions_0_layers = [key for key in new_checkpoint if "mid_new_2.attentions.0" in key]

    paths = reverse_renew_resnet_paths(mid_new_2_resnets_0_layers)
    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint, additional_replacements=[{"old": "mid_new_2.", "new": "mid."}, {"old": "block.0", "new": "block_1"}])

    paths = reverse_renew_resnet_paths(mid_new_2_resnets_1_layers)
    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint, additional_replacements=[{"old": "mid_new_2.", "new": "mid."}, {"old": "block.1", "new": "block_2"}])

    paths = reverse_renew_attention_paths(mid_new_2_attentions_0_layers, in_mid=True)
    reverse_assign_to_checkpoint(paths, new_checkpoint, original_checkpoint, additional_replacements=[{"old": "mid_new_2.", "new": "mid."}, {"old": "attentions.0", "new": "attn_1"}])

    return original_checkpoint



def reverse_convert_ddpm_checkpoint(new_checkpoint, config):
    """Reverses the process of converting ddpm checkpoint."""
    original_checkpoint = {}  # Start with an empty dictionary

    # Reverse the process of handling remaining logic
    original_checkpoint = reverse_handle_remaining_logic(new_checkpoint, original_checkpoint, config)

    # Reverse the process of creating up blocks
    original_checkpoint = reverse_create_up_blocks(new_checkpoint, original_checkpoint, config)

    # Reverse the process of creating down blocks
    original_checkpoint = reverse_create_down_blocks(new_checkpoint, original_checkpoint, config)

    # Reverse the process of initializing new checkpoint
    original_checkpoint = reverse_initialize_new_checkpoint(new_checkpoint, original_checkpoint)

    return original_checkpoint

def reverse_convert_ddpm_checkpoint2(new_checkpoint):
    """
    Reverses the conversion of a checkpoint from the diffusers format back to the original DDPM format.
    
    :param new_checkpoint: The checkpoint dictionary in the diffusers format.
    :return: A checkpoint dictionary in the original DDPM format.
    """
    original_checkpoint = {}


    # Reverse down and up blocks, attention, and resnet paths
    for key, value in new_checkpoint.items():
        original_key = key
        original_key = original_key.replace("time_embedding.linear_1.weight", "temb.dense.0.weight")
        original_key = original_key.replace("time_embedding.linear_1.bias", "temb.dense.0.bias")
        original_key = original_key.replace("time_embedding.linear_2.weight", "temb.dense.1.weight")
        original_key = original_key.replace("time_embedding.linear_2.bias", "temb.dense.1.bias")
        original_key = original_key.replace("conv_norm_out.weight", "norm_out.weight")
        original_key = original_key.replace("conv_norm_out.bias", "norm_out.bias")

        # Handle down_blocks and up_blocks
        original_key = original_key.replace("down_blocks.", "down.").replace("up_blocks.", "up.")

        # Handle downsamplers and upsamplers
        original_key = original_key.replace("downsamplers.0.conv", "downsample.conv").replace("upsamplers.0.conv", "upsample.conv")

        # Reverse attention and resnet paths
        original_key = original_key.replace("resnets.", "block.").replace("attentions.", "attn.")
        original_key = original_key.replace("to_k.", ".k.").replace("to_v.", ".v.").replace("to_q.", ".q.")
        original_key = original_key.replace("to_out.0", "proj_out").replace("group_norm", "norm")

        # Handle specific naming conventions
        original_key = original_key.replace("conv_shortcut", "in_shortcut").replace("conv1", "conv_shorcut")
        original_key = original_key.replace("time_emb_proj", "temb_proj")

        # Reverse mid block adjustments
        original_key = original_key.replace("mid_block", "mid_new_2")

        # Apply the reversed key
        original_checkpoint[original_key] = value

    # Additional specific reversals if necessary
    original_checkpoint = {k.replace("mid_new_2", "mid"): v for k, v in original_checkpoint.items()}

    return original_checkpoint

# Example usage
# new_checkpoint = torch.load("path_to_diffusers_formatted_checkpoint.pth")
# original_ddpm_checkpoint = reverse_convert_ddpm_checkpoint(new_checkpoint)
if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.model_path):
        if args.pruned_model_ckpt is not None:
            print("Loading pruned model from {}".format(args.pruned_model_ckpt))
            unet = torch.load(args.pruned_model_ckpt, map_location=torch.device('cpu')).eval()
        else:
            print("Loading model from {}".format(args.model_path))
            subfolder = 'unet' if os.path.isdir(os.path.join(args.model_path, 'unet')) else None
            unet = UNet2DModel.from_pretrained(args.model_path, subfolder=subfolder).eval()
        pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        )
    # standard model
    else:  
        print("Loading pretrained model from {}".format(args.model_path))
        pipeline = DDIMPipeline.from_pretrained(
            args.model_path,
        )
    state_dict = unet.state_dict()
    with open(args.config_file) as f:
        config = json.loads(f.read())
    print(state_dict.keys())  
    # original_ddpm_checkpoint = reverse_convert_ddpm_checkpoint(state_dict, config)
    # torch.save(original_ddpm_checkpoint, os.path.join(args.output_dir, 'official', 'unet_ema_pruned_converted_keys.pth'))

  
    
