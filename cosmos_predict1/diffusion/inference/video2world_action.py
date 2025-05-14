# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from glob import glob
from typing import Tuple

import einops
import imageio
import mediapy
import numpy as np
import torch
from torchvision import transforms as Trans
from tqdm import tqdm

from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
    validate_args,
)
from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldActionGenerationPipeline
from cosmos_predict1.diffusion.model.model_v2w_action import DiffusionActionV2WModel
from cosmos_predict1.diffusion.training.datasets.dataset_utils import Resize_Preprocess, ToTensorVideo
from cosmos_predict1.diffusion.training.utils.inference_long_video import (
    generate_video_from_batch_with_loop,
    switch_config_for_inference,
)
from cosmos_predict1.utils import misc

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-conditional video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add video2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-Predict1-7B-Video2World_action_post-trained",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-Predict1-7B-Video2World_action_post-trained",
            "Cosmos-Predict1-14B-Video2World_action_post-trained",
        ],
    )
    # For image2video or long video generation
    parser.add_argument(
        "--input_image_or_video_path",
        type=str,
        help="Input video/image path for generating a single video",
    )
    parser.add_argument(
        "--config", type=str, default="Cosmos_Predict1_Video2World_7B_Action_Post_trained", help="inference only config"
    )
    # check if num_input_frames / reusing num_video_frames is better
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames to condition and generate")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output path")
    parser.add_argument(
        "--num_of_latent_condition",
        type=int,
        default=1,
        help="Number of latent condition to condition on",
    )
    parser.add_argument(
        "--num_of_loops",
        type=int,
        default=1,
        help="Number of loops to generate video",
    )

    return parser.parse_args()


def get_traj(annotation_path: str) -> Tuple[np.ndarray, np.ndarray]:
    not_norm_preprocess = Trans.Compose([ToTensorVideo(), Resize_Preprocess(tuple([256, 320]))])

    with open(annotation_path, "r") as file:
        data = json.load(file)

    bridge_path = "datasets/bridge"
    video_path = os.path.join(bridge_path, data["videos"][0]["video_path"])

    frames = imageio.v3.imread(video_path)
    frames = not_norm_preprocess(torch.from_numpy(frames).permute(0, 3, 1, 2))
    frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    action_ee = np.array(data["action"])[:, :6] * 20
    gripper = np.array(data["action"])[:, 6][:, None]
    action = np.concatenate([action_ee, gripper], axis=1)

    return frames, action


def get_condition_latent_v2(
    model: DiffusionActionV2WModel, data_batch: dict, num_of_latent_overlap: int
) -> torch.Tensor:
    _, x0, _ = model.get_data_and_condition(data_batch, num_condition_t=num_of_latent_overlap)
    return x0


def demo(args: argparse.Namespace) -> None:
    """Run video-to-world generation with action control demo.

    This function loads a action-conditioned video-to-world pipeline,  including
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from videos and action
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)

    The function will save:
        - Generated MP4 video files
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args, inference_type)

    # instantiate model, config and load checkpoint
    model = load_model_by_config(
        config_job_name=args.config,
        config_file="cosmos_predict1/diffusion/config/config.py",
        model_class=DiffusionActionV2WModel,
    )

    # model loading part
    load_network_model(model, os.path.join("checkpoints", args.diffusion_transformer_dir, "model.pt"))
    diffusion_decoder_tokenizer_path = os.path.join("checkpoints", "Cosmos-Tokenize1-CV8x8x8-720p")
    load_tokenizer_model(model, diffusion_decoder_tokenizer_path)

    raw_video_batch = dict()
    raw_video_batch["video"] = None
    raw_video_batch["chunk_index"] = torch.zeros(1, device="cuda", dtype=torch.int64)
    raw_video_batch["padding_mask"] = torch.zeros(1, 1, 704, 1280, device="cuda", dtype=torch.bfloat16)
    raw_video_batch["t5_text_embeddings"] = torch.zeros(1, 512, 1024, dtype=torch.bfloat16).cuda()
    raw_video_batch["t5_text_mask"] = torch.ones(1, 512, device="cuda", dtype=torch.int64)
    raw_video_batch["fps"] = torch.tensor([4.0]).cuda()
    raw_video_batch["image_size"] = torch.tensor([[256, 256, 256, 256]]).cuda()
    raw_video_batch["num_frames"] = torch.tensor([args.num_frames]).cuda()
    raw_video_batch["dataset_name"] = "video_data"

    # get input/output paths
    bridge_test_dir = "datasets/bridge/annotation/test"

    annotation_list = glob(os.path.join(bridge_test_dir, "*"))
    annotation_list = sorted(annotation_list)
    # annotation_list = annotation_list[0:10]
    annotation_list = [annotation for annotation in annotation_list if os.path.basename(annotation) == "346.json"]

    os.makedirs(args.output_path, exist_ok=True)
    for annotation_path in tqdm(annotation_list[::-1]):
        traj_id = annotation_path.split("/")[-1].replace(".json", "")

        video_target_path = os.path.join(args.output_path, f"{traj_id}.mp4")
        if os.path.exists(video_target_path):
            print(f"Annotation file {video_target_path} does exist.")
            continue

        frames, actions = get_traj(annotation_path)

        pred_frames, curr_frame = [frames[0]], frames[0]

        for a_t in actions:
            with switch_config_for_inference(model):
                a = torch.tensor(a_t)[None, None, ...].bfloat16().cuda()

                zero_pad = torch.zeros(curr_frame.shape).byte().cuda()
                curr_frame_tensor = torch.tensor(curr_frame).cuda()
                chunk_tensor = torch.stack([curr_frame_tensor, zero_pad], dim=0)[None, ...]
                chunk_tensor = einops.rearrange(chunk_tensor, "b t h w c -> b c t h w")

                raw_video_batch["video"] = chunk_tensor
                raw_video_batch["action"] = a

                print(raw_video_batch["action"])

                raw_video_batch["is_preprocessed"] = False

                condition_latent = get_condition_latent_v2(model, raw_video_batch, args.num_of_latent_condition)  # type: ignore

                # Pad condition latent to have shape [1, 16, num_latents, 32, 40]
                num_latents = (args.num_frames - 1) // 8 + 1
                B, C, T, H, W = condition_latent.shape
                paddings = torch.zeros(B, C, 0, H, W, dtype=torch.bfloat16).cuda()
                condition_latent = torch.cat([condition_latent, paddings], dim=2)

                video_np_THWC_v1, _, _ = generate_video_from_batch_with_loop(
                    model=model,
                    data_batch=raw_video_batch,
                    condition_latent=condition_latent,
                    num_of_loops=args.num_of_loops,
                    num_of_latent_overlap_list=[args.num_of_latent_condition] * args.num_of_loops,
                    guidance=args.guidance,
                    state_shape=model.state_shape,
                    num_steps=35,
                    seed=args.seed,
                    is_negative_prompt=False,
                    visualize=False,
                    save_fig_path=None,
                    return_noise=False,
                )

            curr_frame = video_np_THWC_v1[1]
            pred_frames.append(curr_frame)

        mediapy.write_video(video_target_path, pred_frames, fps=3)

    return


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
