<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Product Website](https://www.nvidia.com/en-us/ai/cosmos/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos-predict1)

Cosmos-Predict1 is a key branch of Cosmos World Foundation Models (WFMs) specialized for future state prediction, often referred to as world models. The tree main branches of Cosmos WFMs are [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict1), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer1), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1). We visualize the architecture of Cosmos-Predict1 in the following figure.

<p align="center">
    <img src="assets/predict1_diagram.png" alt="Cosmos-Predict1 Architecture Diagram">
</p>


Cosmos-Predict1 includes the following:

- **Diffusion-based world foundation models** for Text2World and Video2World generation, where a user can generate visual simulation based on text prompts and video prompts.
- **Autoregressive-based world foundation models** for Video2World generation, where a user can generate visual simulation based on video prompts and optional text prompts.
- **Image and video tokenizers** for tokenizing videos into continuous tokens (latent vectors) and discrete tokens (integers) efficiently and effectively.
- **Post-training scripts** for helping Physical AI builders post-train pre-trained Cosmos-Predict1 for their applications.

## Example Model Behavior

 [Cosmos-Predict Text2World](https://github.com/nvidia-cosmos/cosmos-predict1)

<video src="https://github.com/user-attachments/assets/8abcc5d0-0840-47ae-8f95-10fc0dae7092"> Your browser does not support the video tag.</video>

[Cosmos-Predict Video2World](https://github.com/nvidia-cosmos/cosmos-predict1)

<video src="https://github.com/user-attachments/assets/d598af27-55de-4bc9-b68e-24b70876be9f"> Your browser does not support the video tag. </video>

## Getting Started

We provide a comphrehensive set of examples to illustrate how to perform inference, post-training, etc, with Cosmos-Predict1. Click a relevant example below and start your Cosmos journey.

### Installation

Please refer to [INSTALL.md](INSTALL.md) for general instructions on environment setup.

### Inference with pre-trained Cosmos-Predict1 models
* [Inference with diffusion-based Text2World models](/examples/inference_diffusion_text2world.md) **[with multi-GPU support]**
* [Inference with diffusion-based Video2World models](/examples/inference_diffusion_video2world.md) **[with multi-GPU support]**
* [Inference with diffusion-based WorldInterpolator models](/examples/inference_diffusion_WorldInterpolator.md) **[with multi-GPU support]**
* [Inference with autoregressive-based base models](/examples/inference_autoregressive_base.md) **[with multi-GPU support]**
* [Inference with autoregressive-based Video2World models](/examples/inference_autoregressive_video2world.md) **[with multi-GPU support]**
* [Inference with tokenizer models](/examples/inference_tokenizer.md)

### Post-train pre-trained Cosmos-Predict1 models
* [Post-train diffusion-based Text2World models using custom datasets](/examples/post-training_diffusion_text2world.md) **[with multi-node support]**
* [Post-train diffusion-based Video2World models using custom datasets](/examples/post-training_diffusion_video2world.md) **[with multi-node support]**
* [Post-train diffusion-based Video2World models with action control using custom datasets](/examples/post-training_diffusion_video2world_action.md) **[with multi-node support]**
* [Post-train diffusion-based WorldInterpolator models using custom datasets](/examples/post-training_diffusion_interpolator.md) **[with multi-node support]**
* [Post-train diffusion-based Text2World models using custom multi-view datasets](/examples/post-training_diffusion_text2world_multiview.md) **[with multi-node support]**
* [Post-train diffusion-based Video2World models using custom multi-view datasets)](/examples/post-training_diffusion_video2world_multiview.md) **[with multi-node support]**
* [Post-train autoregressive-based base models using custom datasets](/examples/post-training_autoregressive_base.md) **[with multi-node support]**
* [Post-train tokenizers using custom datasets](/examples/post-training_tokenizer.md) **[with multi-node support]**

### Inference with post-trained models:
* [Inference with post-trained multi-view diffusion-based Text2World models)](/examples/inference_diffusion_text2world_multiview.md) **[with multi-GPU support]**
* [Inference with post-trained multi-view diffusion-based Video2World models)](/examples/inference_diffusion_video2world_multiview.md) **[with multi-GPU support]**


## Cosmos-Predict1 Models

Cosmos-Predict1 include the following models

**Diffusion models**

* [Cosmos-Predict1-7B-Text2World](https://huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World): Text to visual world generation
* [Cosmos-Predict1-14B-Text2World](https://huggingface.co/nvidia/Cosmos-Predict1-14B-Text2World): Text to visual world generation
* [Cosmos-Predict1-7B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-7B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict1-14B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-14B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict1-7B-WorldInterpolator](https://huggingface.co/nvidia/Cosmos-Predict1-7B-WorldInterpolator): Generates a smooth, higher-FPS video segment.

**Autoregressive models**

* [Cosmos-Predict1-4B](https://huggingface.co/nvidia/Cosmos-Predict1-4B): Future visual world generation
* [Cosmos-Predict1-12B](https://huggingface.co/nvidia/Cosmos-Predict1-12B): Future visual world generation
* [Cosmos-Predict1-5B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-5B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict1-13B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-13B-Video2World): Video + Text based future visual world generation

**Tokenizers**

* [Cosmos-Tokenize1-CV8×8×8-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p): Continuous Video Tokenizer with 8x8x8 spatio-temporal compression with, 121 frames context
* [Cosmos-Tokenize1-DV8×16×16-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DV8x16x16-720p): Discrete Video Tokenizer with 8x16x16 spatio-temporal compression, and 49 frames context
* [Cosmos-Tokenize1-CI8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CI8x8-360p): Continuous Image Tokenizer with 8x8 spatial compression with low-resolution support
* [Cosmos-Tokenize1-CI16x16-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CI16x16-360p): Continuous Image Tokenizer with 16x16 spatial compression with low-resolution support
* [Cosmos-Tokenize1-CV4×8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV4x8x8-360p): Continuous Video Tokenizer with 4x8x8 spatio-temporal compression with low-resolution support
* [Cosmos-Tokenize1-DI8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DI8x8-360p): Discrete Image Tokenizer with 8x8 spatial compression with low-resolution support
* [Cosmos-Tokenize1-DI16x16-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DI16x16-360p): Discrete Image Tokenizer with 16x16 spatial compression with low-resolution support
* [Cosmos-Tokenize1-DV4×8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DV4x8x8-360p): Discrete Video Tokenizer with 4x8x8 spatio-temporal compression with low-resolution support

<!-- ------------------------------ -->

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

This model includes safety and content moderation features powered by Llama Guard 3. Llama Guard 3 is used solely as a content input filter and is subject to its own license.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
