# GFlowNet for Contradictory Question Generation

This project implements a sophisticated Reinforcement Learning pipeline using Generative Flow Networks (GFlowNets) to fine-tune a large language model (`gpt2-large`). The goal is to train the model to generate questions that, when answered, produce a statement that contradicts a given fact.

The entire training process is built on PyTorch Lightning and is designed for efficient, multi-GPU training using Distributed Data Parallel (DDP).

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Core Components](#core-components)
3.  [System Architecture](#system-architecture)
4.  [Prerequisites](#prerequisites)
5.  [Setup Instructions](#setup-instructions)
6.  [Training](#training)
    -   [Configuration](#configuration)
    -   [Running the Training Script](#running-the-training-script)
    -   [Resuming from Checkpoints](#resuming-from-checkpoints)
7.  [Key DDP and Memory Management Strategies](#key-ddp-and-memory-management-strategies)
8.  [Project Structure](#project-structure)

## 1. Project Overview

The core task is to fine-tune `gpt2-large` to generate questions related to a subject. The "reward" for a good question is determined by a complex process:
1.  The generated question is passed to a base language model to generate a declarative answer.
2.  This answer is then evaluated by a Natural Language Inference (NLI) model (`DeBERTa-v3-large-mnli`) to see if it contradicts an original "edit fact".
3.  A higher reward is given for questions that lead to strong contradictions.

This reward signal is used within a GFlowNet framework, specifically with a Sub-Trajectory Balance (SubTB) loss, to update the policy of the `gpt2-large` model. This encourages the model to generate a diverse set of high-reward (i.e., highly contradictory) questions.

## 2. Core Components

-   **Generative Model (`sampler_model`)**: A `gpt2-large` model with LoRA adapters for parameter-efficient fine-tuning. This is the model whose policy is being trained.
-   **Reward Calculation Models**:
    -   **`base_model`**: A frozen, inference-only `gpt2-large` used to generate answers and calculate likelihood scores for the reward function.
    -   **`nli_model`**: A frozen `DeBERTa-v3-large-mnli` model used to score the contradiction between a generated answer and a given fact.
-   **GFlowNet Objective**: The `modified_subtb_loss` ensures the model learns a valid probability distribution over the space of possible questions.
-   **Replay Buffer**: A `RedisReplayBuffer` stores high-reward generated trajectories, allowing the model to efficiently revisit and learn from its best-performing samples. It is DDP-safe and offloads memory from the GPUs.
-   **Training Framework**: [PyTorch Lightning](https://www.pytorchlightning.ai/) orchestrates the entire training loop, providing seamless DDP integration, logging, and checkpointing.
-   **Configuration**: [Hydra](https://hydra.cc/) manages all configuration parameters, allowing for flexible and clean command-line overrides.

## 3. System Architecture

The training process for each batch within a DDP environment proceeds as follows:

1.  **Data Loading**: Each of the 4 DDP ranks receives a unique data sample.
2.  **Trajectory Generation (Rank 0 Only)**: The rank-0 process decides whether to generate new trajectories from the `sampler_model` or sample high-reward ones from the shared Redis replay buffer.
3.  **Reward Calculation (Rank 0 Only)**: If new trajectories were generated, rank 0 uses the `base_model` and `nli_model` to calculate the final scaled and unscaled rewards.
4.  **Broadcast**: Rank 0 broadcasts the chosen trajectories and their rewards to all other DDP ranks. This is a critical step to ensure all GPUs work on identical data for a given step, preventing gradient divergence.
5.  **Parallel Re-evaluation**: All ranks (including rank 0) perform a forward pass with the received trajectories to deterministically calculate the necessary log-probabilities (`log_pf`, `log_pterm`) for the GFlowNet loss.
6.  **Loss Calculation**: Each rank computes the `modified_subtb_loss` based on the broadcasted rewards and its locally computed log-probabilities.
7.  **Backward Pass & Gradient Sync**: A backward pass is performed. PyTorch's DDP automatically synchronizes (averages) the gradients for the trainable LoRA parameters across all 4 GPUs.
8.  **Optimizer Step**: After accumulating gradients for `accumulate_grad_batches` steps, the optimizer updates the LoRA weights.

## 4. Prerequisites

-   NVIDIA GPUs with at least 24GB VRAM each (e.g., RTX 3090/4090, A5000).
-   CUDA Toolkit and a compatible PyTorch version.
-   A Conda or other virtual environment manager.
-   A running Redis server.

## 5. Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd GFlowNet_question_generation
    ```

2.  **Create and activate the Conda environment:**
    The project was tested with Python 3.11. You can create a new environment:
    ```bash
    conda create -n gfn_stable python=3.11
    conda activate gfn_stable
    ```

3.  **Install Python dependencies:**
    A `requirements.txt` file should be created with the following key packages. Install them via pip.
    ```
    # requirements.txt
    torch
    pytorch-lightning
    transformers
    datasets
    peft
    hydra-core
    hydra-colorlog
    wandb
    redis
    numpy
    pandas
    bitsandbytes
    ```
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the dataset:**
    Ensure your dataset `data/zsre_1000.jsonl` is present in the specified path.

5.  **Start the Redis Server:**
    The provided launch script handles this automatically. If you need to start it manually:
    ```bash
    redis-server --port 6379 &
    ```

## 6. Training

### Configuration

All training parameters are managed through YAML files in the `configs/` directory, orchestrated by `configs/config.yaml`. Key parameters to adjust include:
-   `configs/config.yaml` -> `task`: `n_samples` (generation batch size per GPU).
-   `configs/config.yaml` -> `training`: `lr`, `epochs`, `accumulate_grad_batches`.
-   `configs/callbacks/*.yaml`: Checkpointing behavior.

### Running the Training Script

The `run_4gpu_ddp.sh` script is the primary entry point for launching a 4-GPU DDP training run on a single machine.

1.  **Make the script executable:**
    ```bash
    chmod +x run_4gpu_ddp.sh
    ```

2.  **Launch the training:**
    ```bash
    ./run_4gpu_ddp.sh
    ```
    This command will:
    -   Check for and start the Redis server if not running.
    -   Set DDP-safe environment variables (`TOKENIZERS_PARALLELISM`, `OMP_NUM_THREADS`).
    -   Use `torch.distributed.run` to launch 4 DDP processes.
    -   Override the configuration to use the DDP strategy and the `production_checkpoint` callback.
    -   Run the training in the background, logging all output to `training_production.log`.

3.  **Monitor the training:**
    ```bash
    tail -f training_production.log
    ```
    And monitor GPU usage with:
    ```bash
    watch nvidia-smi
    ```

### Resuming from Checkpoints

The `production_checkpoint` callback automatically saves the last state to `production_checkpoints/last.ckpt`. The launch script is designed to automatically resume from this file if it exists.

To resume an interrupted run, simply execute the launch script again:
```bash
./run_4gpu_ddp.sh