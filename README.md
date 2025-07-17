# GFlowNet for Contradictory Question Generation

This project implements a sophisticated Reinforcement Learning pipeline using Generative Flow Networks (GFlowNets) to fine-tune a large language model (`gpt2-large`). The ambitious goal is to train the model to generate diverse and high-quality questions that, when answered by a separate base LLM, produce a statement that directly contradicts a given "edit fact".

The entire training process is built on PyTorch Lightning, designed for efficient, multi-GPU training using Distributed Data Parallel (DDP), and incorporates advanced memory management and experiment tracking strategies.

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Core Components](#2-core-components)
    1.  [Generative Model (`sampler_model`)](#21-generative-model-sampler_model)
    2.  [Reward Calculation Models](#22-reward-calculation-models)
    3.  [GFlowNet Objective](#23-gflownet-objective)
    4.  [Replay Buffer](#24-replay-buffer)
    5.  [Training Framework](#25-training-framework)
3.  [System Architecture & DDP Strategy](#3-system-architecture--ddp-strategy)
    1.  [Asymmetric DDP Workload](#31-asymmetric-ddp-workload)
    2.  [Memory Management Strategies](#32-memory-management-strategies)
    3.  [Logging & Monitoring](#33-logging--monitoring)
4.  [Reward Function Deep Dive](#4-reward-function-deep-dive)
    1.  [Reward Composition](#41-reward-composition)
    2.  [Challenges: The "Long Babbling" Mode Collapse](#42-challenges-the-long-babbling-mode-collapse)
    3.  [Diagnostic Metrics](#43-diagnostic-metrics)
5.  [Experimentation & Debugging Insights](#5-experimentation--debugging-insights)
    1.  [Environment & Dependency Mismatches](#51-environment--dependency-mismatches)
    2.  [Distributed Training Specifics](#52-distributed-training-specifics)
    3.  [Configuration & Logging Quirks](#53-configuration--logging-quirks)
6.  [Prerequisites](#6-prerequisites)
7.  [Setup Instructions](#7-setup-instructions)
8.  [Running Experiments](#8-running-experiments)
    1.  [Configuration Management (Hydra)](#81-configuration-management-hydra)
    2.  [Launching a Formal Training Run (`run_4gpu_ddp.sh`)](#82-launching-a-formal-training-run-run_4gpuddpsh)
    3.  [Launching Hyperparameter Sweeps (`run_experiments.sh`)](#83-launching-hyperparameter-sweeps-run_experimentssh)
    4.  [Monitoring Experiments](#84-monitoring-experiments)
    5.  [Cleaning Up Artifacts](#85-cleaning-up-artifacts)
9.  [Project Structure](#9-project-structure)

---

## 1. Project Overview

The core objective of this project is to fine-tune a `gpt2-large` model (the "sampler model") to become an expert at generating questions. The definition of a "good" question is highly specific and involves a multi-stage evaluation process:

1.  **Question Generation:** The `sampler_model` generates a question based on a given subject (e.g., "William Shakespeare").
2.  **Answer Generation:** This generated question is then fed into a separate, *frozen* `gpt2-large` model (the "base model"), which provides a concise, declarative answer.
3.  **Contradiction Evaluation:** The generated answer is evaluated by a Natural Language Inference (NLI) model (`DeBERTa-v3-large-mnli`) against an original "edit fact" (e.g., "William Shakespeare wrote the play 'Hamlet'." vs. "The play 'Hamlet' was written by Christopher Marlowe.").
4.  **Reward Calculation:** A higher reward is assigned if the generated answer strongly contradicts the edit fact, combined with a likelihood score to ensure fluency.

This reward signal is then used within a **Generative Flow Network (GFlowNet)** framework, specifically leveraging the **Sub-Trajectory Balance (SubTB) loss**. GFlowNets aim to learn a diverse probability distribution over the space of possible questions, encouraging the `sampler_model` to explore and generate a wide array of high-reward questions, rather than simply converging on a single "best" one.

The entire training process is orchestrated using **PyTorch Lightning** and designed for **efficient, multi-GPU Distributed Data Parallel (DDP) training**.

## 2. Core Components

### 2.1. Generative Model (`sampler_model`)
*   **Model:** `gpt2-large` (774M parameters).
*   **Fine-tuning Method:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA (Low-Rank Adaptation)**. LoRA adds small, trainable low-rank matrices to the large, frozen pre-trained model, drastically reducing the number of parameters that need to be updated. This is critical for memory efficiency and faster training.
*   **Target Modules:** LoRA adapters are specifically applied to the `c_attn` layers (Query, Key, Value projections) within GPT-2's attention mechanism.
*   **Training Objective:** Updated by the GFlowNet's SubTB loss.

### 2.2. Reward Calculation Models
The reward function is the most intricate part of the pipeline, providing the learning signal for the GFlowNet. It leverages two large, frozen models to ensure a stable and consistent reward signal:
*   **`base_model`:** A **frozen, inference-only `gpt2-large` model**. This model's sole purpose is to generate a declarative answer for any question produced by the `sampler_model`. It ensures that the generated answer is plausible and coherent based on general language understanding, separate from the `sampler_model`'s learning.
*   **`nli_model`:** A **frozen `DeBERTa-v3-large-mnli` model** (`NDugar/v3-Large-mnli` from Hugging Face). This model is a Natural Language Inference expert. It takes two sentences (a premise and a hypothesis) and classifies their relationship as entailment, neutral, or contradiction. Its `contradiction` score forms the primary component of our reward.

### 2.3. GFlowNet Objective
*   **Loss Function:** `modified_subtb_loss`. This is the **Sub-Trajectory Balance (SubTB) loss**, a fundamental objective in GFlowNets. It enforces a consistency condition: for any sub-trajectory, the flow entering it must equal the flow leaving it. Optimizing this loss helps the model learn a valid probability distribution over the entire space of possible questions, promoting diversity and covering high-reward regions.
*   **Key Inputs:** `log_pf` (forward policy log-probs), `log_pterm` (termination log-probs), and `log_r` (intermediate log-rewards).

### 2.4. Replay Buffer
*   **Type:** `RedisReplayBuffer` (a custom implementation using a Redis backend).
*   **Functionality:** Stores high-reward generated trajectories (questions) off-GPU in a Redis sorted set. This provides:
    *   **Memory Efficiency:** Offloads large text data from GPU/CPU memory.
    *   **Sample Efficiency (Prioritized Replay):** Allows the model to revisit and learn from its best-performing samples found so far, accelerating training.
    *   **DDP Safety:** Designed for distributed environments, ensuring consistent sampling across all DDP ranks. Critically, it indexes its buffers **per-prompt**, storing high-reward questions for specific contexts.

### 2.5. Training Framework
*   **Orchestration:** [PyTorch Lightning](https://www.pytorchlightning.ai/) streamlines the entire training loop, providing seamless DDP integration, structured logging, and robust checkpointing.
*   **Configuration:** [Hydra](https://hydra.cc/) manages all configuration parameters through YAML files, enabling flexible command-line overrides and a clean, modular structure.
*   **Logging:** [Weights & Biases (W&B)](https://wandb.ai/) for experiment tracking, visualization, and comparison of hyperparameter sweeps (configured for offline logging for robustness).

## 3. System Architecture & DDP Strategy

The project's distributed training strategy is a key differentiator, designed to optimize resource utilization for complex RL pipelines.

### 3.1. Asymmetric DDP Workload

The most notable feature is the **asymmetric workload distribution** across the 4 DDP ranks within each training step. This design is crucial for efficiency, as the expensive reward calculation models (`base_model`, `nli_model`) only need to be loaded and run on a single GPU.

1.  **Data Loading:** Each of the 4 DDP ranks receives a unique data sample (an `original_fact`, `edit_fact`, and `subject`).
2.  **Trajectory Generation & Reward Calculation (Rank 0 Only):**
    *   The **rank-0 process** acts as the "leader."
    *   It decides whether to generate new trajectories from the `sampler_model` (exploration) or sample high-reward ones from the shared Redis replay buffer (exploitation).
    *   If new trajectories are generated, Rank 0 uses the `base_model` and `nli_model` (which are only loaded on this GPU) to calculate the complex contradiction and likelihood scores, forming the final reward.
    *   High-reward new trajectories are added to the Redis replay buffer.
    *   Rank 0 also computes diagnostic metrics like semantic diversity for *newly generated* samples.
3.  **Broadcast:** Rank 0 broadcasts the chosen batch of trajectories and their calculated rewards (including reward components and semantic diversity) to all other DDP ranks. This is a critical synchronization step to ensure all GPUs work on identical data for a given step, preventing gradient divergence.
4.  **Parallel Re-evaluation:** All ranks (including rank 0) perform a forward pass with the received trajectories. This pass uses the *current* `sampler_model` weights to deterministically calculate the necessary log-probabilities (`log_pf`, `log_pterm`) needed for the GFlowNet loss. No reward calculation happens here.
5.  **Loss Calculation:** Each rank computes the `modified_subtb_loss` based on the broadcasted rewards and its locally computed log-probabilities.
6.  **Backward Pass & Gradient Sync:** A backward pass is performed. PyTorch's DDP automatically synchronizes (averages) the gradients for the trainable LoRA parameters across all 4 GPUs. The `find_unused_parameters=True` flag in the DDP strategy is crucial here, as it robustly handles parameters that might not receive gradients on every rank due to the asymmetric workload.
7.  **Optimizer Step:** After accumulating gradients for `accumulate_grad_batches` steps, the optimizer updates the LoRA weights.

### 3.2. Memory Management Strategies

Training `gpt2-large` requires careful memory management. The project employs several techniques:
*   **LoRA (PEFT):** Only a small fraction of the total model parameters are trainable (approx. 0.38%), significantly reducing VRAM footprint for optimizer states and gradients.
*   **Gradient Checkpointing:** Enabled on the `sampler_model`. This technique re-computes intermediate activations during the backward pass rather than storing them in memory during the forward pass, trading a small amount of compute for substantial VRAM savings.
*   **`bitsandbytes` (4-bit Quantization):** The `sampler_model` can be loaded in 4-bit quantized format. This further reduces its memory footprint, although it requires careful handling of `.to()` calls (monkey-patching `LightningModule.to/cuda` methods) to avoid conflicts with PyTorch Lightning's device management.
*   **`RedisReplayBuffer`:** Offloads large batches of text data from GPU VRAM to system RAM (via the Redis server), preventing OOM errors from the buffer itself.
*   **Gradient Accumulation (`accumulate_grad_batches`):** Simulates larger effective batch sizes by accumulating gradients over multiple steps before performing an optimizer update, reducing instantaneous memory requirements for gradients.

### 3.3. Logging & Monitoring

Comprehensive logging is integrated for effective experiment tracking and debugging:
*   **Weights & Biases (W&B):** Used for all experiment metrics, custom charts, and qualitative sample logging.
    *   Runs are grouped by `logger.group` for easy comparison.
    *   `log_model: false` is set to avoid uploading huge checkpoint files to W&B, relying on local checkpointing instead.
    *   `mode: offline` is used for robustness against network issues; runs are synced to W&B cloud manually later.
*   **Logging Frequency:** `log_every_n_steps` is configured in `config.yaml` and passed to the `pl.Trainer` to control the granularity of step-level metrics.
*   **`rank_zero_only` Logging:** For metrics calculated only on Rank 0 (like semantic diversity) or when fine-grained control is needed, `rank_zero_only=True` is used in `self.log` calls to avoid DDP synchronization overheads for logging and prevent duplicate metric entries.

## 4. Reward Function Deep Dive

The reward function (`ContradictionReward`) is paramount to guiding the GFlowNet. Its design directly influences the model's behavior.

### 4.1. Reward Composition

The final (unscaled) log-reward for a generated question (`log_R(Q)`) is a composite of two main components:
`log_R(Q) = log_C(A | F_edit) + likelihood_weight * log_P(Q | Prompt)`

*   **`log_C(A | F_edit)` (Contradiction Score):** This is the log-probability of the `contradiction` label assigned by the `nli_model` (`DeBERTa-v3-large-mnli`) when comparing the `base_model`'s answer (`A`) to the `edit_fact` (`F_edit`). This is the primary signal for task success.
*   **`log_P(Q | Prompt)` (Likelihood Score):** This is the log-probability of the generated question sequence under the `base_model`. This term acts as a regularizer, encouraging the `sampler_model` to generate questions that are grammatically fluent and plausible according to a general-purpose LLM, preventing incoherent or random output.
*   **`likelihood_weight`:** A configurable hyperparameter that controls the balance between contradiction and fluency.
*   **Length Penalty:** A harsh penalty (`-99.0`) is applied to the final reward if the generated question length falls below `min_question_len`.
*   **Reward Temperature:** The final reward passed to the GFlowNet loss is scaled by a `temperature` parameter, which can be annealed during training to smooth or sharpen the reward landscape.

### 4.2. Challenges: The "Long Babbling" Mode Collapse

During early experiments, a significant learning pathology was observed:
*   **`train/loss` (SubTB Loss) was decreasing:** Indicated the GFlowNet mechanism was optimizing correctly.
*   **`train/avg_log_reward_unscaled` was decreasing:** Indicated the model was getting *worse* at the actual task.
*   **`train/avg_question_len` was increasing and plateauing at `max_question_len`:** Indicated the model was generating long sequences.

**Diagnosis:** The model was stuck in a **"long babbling" mode collapse**. It quickly learned to avoid the `min_question_len` penalty. Then, instead of finding semantically contradictory questions (a very sparse reward), it found a "safe" local optimum: generating long, grammatically plausible, but semantically empty or generic questions. These questions would receive a decent `likelihood_score` (because they sounded like GPT-2) and a neutral/low `contradiction_score` (because they were nonsensical). The `likelihood_weight` was likely too high, making it easier to optimize for fluency than for the difficult contradiction task.

### 4.3. Diagnostic Metrics

To confirm the "long babbling" hypothesis and guide interventions, the following key diagnostic metrics were added:

*   **`train/avg_log_contradiction`:** The direct NLI score. Expected to drop/stay low in babbling mode.
*   **`train/avg_log_likelihood`:** The base model fluency score. Expected to stay high/increase in babbling mode.
*   **`train/semantic_diversity`:** Measured by the average pairwise cosine distance of sentence embeddings (using `all-MiniLM-L6-v2`). Expected to drop when the model collapses to similar, non-diverse questions.
*   **`generation_probes` (W&B Table):** Qualitative inspection of actual generated questions to directly observe the "babbling" phenomenon.

## 5. Experimentation & Debugging Insights

The development of this complex pipeline involved several common challenges and required iterative debugging. Documenting these helps in future maintenance and extensions.

### 5.1. Environment & Dependency Mismatches
*   **`ModuleNotFoundError` for `sentence-transformers`:**
    *   **Cause:** The library was installed in the `base` Conda environment (Python 3.12) via a regular `pip install`, but the project's launch script explicitly used the Python executable from the `gfn_stable` environment (Python 3.11).
    *   **Solution:** Forced installation to the correct environment using `path/to/gfn_stable/bin/python -m pip install sentence-transformers`.
*   **`tqdm` Progress Bars:**
    *   **Cause:** The `SentenceTransformer.encode()` method (and potentially other underlying Hugging Face operations) uses its own `tqdm` progress bars, which ignore the `TQDM_DISABLE` environment variable.
    *   **Solution:** Explicitly disabled the progress bar by passing `show_progress_bar=False` to the `SentenceTransformer.encode()` method call in `reward.py`.

### 5.2. Distributed Training Specifics
*   **`RuntimeError: No backend type associated with device type cpu`:**
    *   **Cause:** When Rank 0 broadcasts results (trajectories, rewards, metrics) to other ranks, it moves them to CPU before broadcasting. The receiving ranks were not explicitly moving these tensors back to their respective GPU devices before being used in operations (like `self.log` with `sync_dist=True`) that require GPU tensors for DDP synchronization. Also, a subtle bug in `RedisReplayBuffer.sample()` could cause 1D tensors to be returned, breaking `.expand()`.
    *   **Solution:** Explicitly call `.to(self.device)` on all broadcasted tensors in `lightning_module.py` immediately after unpacking them. Fixed `RedisReplayBuffer` to always return 2D reward tensors using `.view(-1, 1)`.
*   **`ValueError: too many values to unpack`:**
    *   **Cause:** The `ContradictionReward.score()` method was refactored to return four values (scaled reward, unscaled reward, contradiction score, likelihood score), but the `gflownet/trajectory.py` function (`generate_and_return_termination_logprob`) was still expecting only two.
    *   **Solution:** Updated `generate_and_return_termination_logprob` to accept a four-value tuple from the `reward_fn` and pass it through to the `LightningModule`.

### 5.3. Configuration & Logging Quirks
*   **`AttributeError: 'AttributeDict' object has no attribute 'log_every_n_steps'`:**
    *   **Cause:** The `log_every_n_steps` parameter was defined in `config.yaml` but was not explicitly passed as an argument to the `ContradictionGFNTask`'s `__init__` method, preventing PyTorch Lightning from automatically populating `self.hparams.log_every_n_steps`.
    *   **Solution:** Passed `log_every_n_steps=config.training.log_every_n_steps` from `train.py` to the `ContradictionGFNTask` constructor.
*   **Hydra `Key 'max_epochs' is not in struct`:**
    *   **Cause:** Attempting to override `trainer.max_epochs` from the command line (`trainer.max_epochs=3`) when `max_epochs` was not originally defined in the `trainer` section of `config.yaml`. Hydra's "structured" config mode prevents adding new keys by default.
    *   **Solution:** Used the Hydra "append" syntax: `+trainer.max_epochs=3`.
*   **Sparse Step-level W&B Charts (Single Dot/Few Points):**
    *   **Cause:** The `pl.Trainer`'s default `log_every_n_steps` was overriding the value in `config.yaml`. More importantly, `accumulate_grad_batches=8` meant that a "global step" for logging purposes occurred only after 8 local distributed batches. With a small dataset (e.g., 14 samples, 4 batches/epoch), few global steps might occur per epoch.
    *   **Solution:** Explicitly passed `log_every_n_steps=config.training.log_every_n_steps` to the `pl.Trainer` constructor in `train.py`. Adjusted `log_every_n_steps` in `config.yaml` to `1` for diagnostic runs to log on every mini-batch. Also configured `self.log` calls for step metrics with `on_step=True, on_epoch=True, rank_zero_only=True` to ensure clean, consistent plotting from Rank 0.

## 6. Prerequisites

*   **Hardware:** NVIDIA GPUs with at least 24GB VRAM each (e.g., RTX 3090/4090, A5000) for `gpt2-large`.
*   **Software:** CUDA Toolkit and a compatible PyTorch version.
*   **Environment:** Conda or other virtual environment manager.
*   **Service:** A running Redis server (accessible on `localhost:6379` by default).

## 7. Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd GFlowNet_question_generation
    ```

2.  **Create and activate the Conda environment:**
    The project was tested with Python 3.11.
    ```bash
    conda create -n gfn_stable python=3.11
    conda activate gfn_stable
    ```

3.  **Install Python dependencies:**
    Ensure your `requirements.txt` includes:
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
    sentence-transformers # NEW
    ```
    Install via pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Self-correction for `sentence-transformers`:* If you encounter `ModuleNotFoundError` despite `pip install` reporting "already satisfied," ensure you're installing into the *correct* Python interpreter for your `gfn_stable` environment:
    ```bash
    /path/to/your/miniconda3/envs/gfn_stable/bin/python -m pip install sentence-transformers
    ```

4.  **Prepare the dataset:**
    Ensure your dataset `data/ZSRE_1000.pkl` is present. Then run the conversion script:
    ```bash
    python convert_data.py
    ```
    This will create `data/zsre_1000.jsonl`.

5.  **Pre-download models (optional, but recommended):**
    ```bash
    python download_models.py
    ```
    This populates the Hugging Face cache, speeding up subsequent runs.

6.  **Start the Redis Server:** The provided launch scripts (`run_4gpu_ddp.sh`, `run_experiments.sh`) handle this automatically. If you need to start it manually for debugging:
    ```bash
    redis-server --port 6379 &
    ```

## 8. Running Experiments

All training parameters are managed through YAML files in the `configs/` directory, orchestrated by `configs/config.yaml`.

### 8.1. Configuration Management (Hydra)
*   Hydra is used to compose the final configuration.
*   Default configurations (e.g., `model: gpt2-large`, `reward: deberta_v3`, `logger: wandb`) are defined in `config.yaml`.
*   Parameters can be overridden from the command line (e.g., `python train.py training.lr=1e-5`).
*   The `+` prefix (e.g., `+trainer.max_epochs=10`) is used to add new keys to the configuration that are not present in the default YAML files.
*   Callbacks are dynamically managed via YAML files in `configs/callbacks/`.

### 8.2. Launching a Formal Training Run (`run_4gpu_ddp.sh`)
This script is designed for long-running production-like training, using a dedicated checkpoint directory and automatic resumption.

1.  **Make executable:** `chmod +x run_4gpu_ddp.sh`
2.  **Run:** `./run_4gpu_ddp.sh`
    *   This will check/start Redis, set DDP environment variables, launch 4 DDP processes, and log output to `training_production.log`.
    *   It uses `callbacks=production_checkpoint` and saves to `production_checkpoints/`.
    *   It automatically resumes from `production_checkpoints/last.ckpt` if found.

### 8.3. Launching Hyperparameter Sweeps (`run_experiments.sh`)
This script facilitates systematic hyperparameter tuning, running multiple configurations sequentially and summarizing results. It dynamically creates unique checkpoint directories and W&B run names for each experiment to prevent overwriting.

1.  **Make executable:** `chmod +x run_experiments.sh`
2.  **Define Experiments:** Edit the `EXPERIMENTS` multi-line string within the script. Each line defines a `likelihood_weight,pf_temp_high,use_buffer_prob` combination to test.
    *   **Current Fast Experiment Settings:** `+trainer.max_epochs` is set to `10` in the `BASE_LAUNCH_CMD` for quick trend analysis. Checkpoints are saved every `20` steps.
3.  **Run:** `./run_experiments.sh`
    *   It will run each experiment sequentially.
    *   Logs for each run will be saved to `training_logs/expN_....log`.
    *   Checkpoints for each run will be saved to `checkpoints/expN_.../`.
    *   Upon completion, it generates `experiment_summary.md` and `experiment_summary.tex`.

### 8.4. Monitoring Experiments
For live monitoring of runs, especially when `run_experiments.sh` is active:

1.  **Overall Progress:** In the terminal where `./run_experiments.sh` is running, observe the "Starting Experiment #X" and "completed successfully" messages.
2.  **Hardware Utilization:** In a separate terminal, run `watch nvidia-smi` to monitor GPU activity and memory usage.
3.  **Detailed Current Log:** In a third terminal, use `tail -f training_logs/CURRENT_RUN_NAME.log` (e.g., `training_logs/exp1_lw0.01_pt2.0_ubp0.5.log`) to see the real-time output and PyTorch Lightning progress bar for the active run.
4.  **W&B Dashboard (Post-Run):** All runs are logged offline. After the sweep completes, sync them to the cloud:
    ```bash
    wandb sync --sync-all
    ```
    Then visit your W&B project (`wandb.ai/<your_user>/contradiction-gfn`) to view charts, tables (`generation_probes`), and compare runs.

### 8.5. Cleaning Up Artifacts
To ensure a clean slate before a new sweep or production run:

*   **Targeted Cleanup (Recommended):** To remove only experiment-related files (diagnostic runs, sweep runs, logs, summaries) while preserving core production checkpoints:
    ```bash
    rm -rf diag_metrics_checkpoints/ checkpoints/ training_logs/ training_diag_metrics.log experiment_summary.* experiment_results.log configs/callbacks/temp_exp_callback.yaml
    ```
*   **Full Cleanup:** To remove *all* generated data (checkpoints, all logs, all W&B offline data):
    ```bash
    rm -rf checkpoints/ diag_metrics_checkpoints/ production_checkpoints/ training_logs/ training_production.log training_diag_metrics.log wandb/ experiment_summary.* experiment_results.log configs/callbacks/temp_exp_callback.yaml
    find . -type d -name "__pycache__" -exec rm -rf {} +
    ```

## 9. Project Structure