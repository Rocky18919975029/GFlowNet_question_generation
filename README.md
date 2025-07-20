# GFlowNet for Contradictory Question Generation

This project implements a sophisticated Reinforcement Learning pipeline using Generative Flow Networks (GFlowNets) to fine-tune a large language model (`gpt2-large`). The ambitious goal is to train the model to generate diverse and high-quality questions that, when answered by a separate base LLM, produce a statement that directly contradicts a given "edit fact".

The entire training process is built on PyTorch Lightning, designed for efficient, multi-GPU training using Distributed Data Parallel (DDP), and incorporates advanced memory management, a hybrid reward schedule, and a robust experimental framework.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [System Architecture](#system-architecture)
    1.  [Core Models](#core-models)
    2.  [GFlowNet Framework](#gflownet-framework)
    3.  [Asymmetric DDP Strategy](#asymmetric-ddp-strategy)
3.  [The Reward Function: An Evolving Design](#the-reward-function-an-evolving-design)
    1.  [Initial Composite Reward](#initial-composite-reward)
    2.  [The Hybrid Reward Schedule](#the-hybrid-reward-schedule)
    3.  [The Answer Quality Reward Component](#the-answer-quality-reward-component)
4.  [Experimental Journey & Key Findings](#experimental-journey--key-findings)
    1.  [Challenge 1: The Learning Collapse](#challenge-1-the-learning-collapse)
    2.  [Challenge 2: The "Nonsensical Question" Pathology](#challenge-2-the-nonsensical-question-pathology)
    3.  [Current Status & Next Steps](#current-status--next-steps)
5.  [Key Technologies](#key-technologies)
6.  [Setup and Usage](#setup-and-usage)
    1.  [Prerequisites](#prerequisites)
    2.  [Installation](#installation)
    3.  [Running Experiments](#running-experiments)
7.  [Project Structure](#project-structure)

## Project Overview

The core objective is to fine-tune a `gpt2-large` "sampler model" to become an expert at generating questions. The definition of a "good" question is highly specific and involves a multi-stage evaluation:

1.  **Question Generation:** The `sampler_model` generates a question based on a given subject (e.g., "William Shakespeare").
2.  **Answer Generation:** This generated question is fed into a separate, *frozen* `gpt2-large` "base model", which provides a declarative answer.
3.  **Contradiction Evaluation:** The generated answer is evaluated by a Natural Language Inference (NLI) model (`DeBERTa-v3-large-mnli`) against a given "edit fact" (e.g., "The play 'Hamlet' was written by Christopher Marlowe.").
4.  **Reward Calculation:** A reward is calculated based on the contradiction score, fluency, and other quality metrics.

This reward signal is used within a **Generative Flow Network (GFlowNet)** framework to train the `sampler_model`. GFlowNets are ideal for this task as they aim to learn a diverse probability distribution over the space of possible questions, encouraging the model to explore and generate a wide array of high-reward outputs rather than a single "best" one.

## System Architecture

### Core Models
*   **`sampler_model` (`gpt2-large`):** The primary generative model, fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with **LoRA**. It learns to generate questions based on the GFlowNet's reward signal.
*   **`base_model` (frozen `gpt2-large`):** An unchanging "answer oracle" that provides consistent answers to any question, ensuring a stable reward landscape.
*   **`nli_model` (frozen `DeBERTa-v3-large-mnli`):** A specialist "contradiction judge" that provides the core external signal for task success.

### GFlowNet Framework
*   **Objective:** The model is trained using the **Sub-Trajectory Balance (SubTB) loss**, a key GFlowNet objective that promotes sampling proportional to a reward function.
*   **Replay Buffer:** A DDP-safe **`RedisReplayBuffer`** stores high-reward trajectories off-GPU, enabling memory-efficient prioritized experience replay and accelerating learning.

### Asymmetric DDP Strategy
To manage the memory load of three large LLMs, the project employs a custom **asymmetric workload distribution** for 4-GPU DDP training:
1.  **Rank 0 (Leader):** This process is the only one that loads the `base_model` and `nli_model`. It is responsible for all reward calculations for newly generated questions and for managing the Redis replay buffer.
2.  **Broadcast:** Rank 0 broadcasts the chosen trajectories (either newly generated or sampled from the buffer) and their calculated rewards to all other ranks.
3.  **All Ranks:** Each of the 4 GPUs receives the same data. They perform a forward pass with the *current* `sampler_model` to calculate the necessary log-probabilities for the SubTB loss and then perform a backward pass. Gradients for the trainable LoRA parameters are synchronized across all GPUs.

## The Reward Function: An Evolving Design

The reward function has been the central focus of experimentation. Its design is critical for guiding the model toward the desired behavior while avoiding pathological loopholes. The final reward, $R(\tau)$, is a multi-stage, scheduled function.

### Initial Composite Reward
The initial design was a simple weighted sum of the external contradiction score and an internal fluency score:
`Reward = log_C(contradiction) + w_L * log_P_Q(question_fluency)`

### The Hybrid Reward Schedule
Early experiments revealed that a static reward function led to unstable training. To solve this, a **hybrid, two-phase reward schedule** was implemented:

1.  **Phase 1: Penalized Reward (e.g., first 150 steps):**
    The model is guided by a hard-penalty system. It is rewarded for fluency (`log_P_Q`) only if the generated question passes a minimum `contradiction_threshold`. Failure results in a massive penalty. This forces the model to quickly learn the basic constraints of the task.
2.  **Phase 2: Composite Reward (rest of training):**
    After the initial bootstrapping phase, the reward function switches to the composite formula, allowing the model to refine its policy and explore the reward landscape more smoothly.

### The Answer Quality Reward Component
Later, long-duration training runs revealed a more subtle failure mode where the model would generate nonsensical questions to "trick" the reward pipeline. To close this loophole, the reward function was further enhanced with an **answer quality component**:

*   **Logic:** The system now calculates the log-likelihood of the answer generated by the `base_model` ($\log P_A$). A hard penalty ($P_A$) is applied to the final reward if this likelihood is below a certain threshold ($T_A$), indicating a confused or generic answer.
*   **Formula:** The final, robust reward function incorporates all three elements: the hybrid schedule, the composite scores, and the answer quality penalty. (See `reward.py` for the full implementation).

## Experimental Journey & Key Findings

The development of this project involved a rigorous, iterative experimental process to diagnose and solve complex learning pathologies.

### Challenge 1: The Learning Collapse
*   **Problem:** Initially, the model would become unstable after the penalized training phase, leading to a collapse in performance as it got "confused" by the composite reward.
*   **Solution:** A hyperparameter sweep on the `contradiction_threshold` was conducted. A value of **-6.0** was found to be the "Goldilocks zone"—lenient enough to encourage exploration but strict enough to provide a strong learning signal, successfully preventing the collapse.

### Challenge 2: The "Nonsensical Question" Pathology
*   **Problem:** With a stable training dynamic, a longer 50-epoch run revealed the model had converged to a new, pathological state: generating grammatically broken, nonsensical questions. It had discovered a loophole where this "gibberish" would confuse the `base_model`, leading to a consistent, mediocre reward.
*   **Investigation:** A comprehensive sweep of the `likelihood_weight` (from 0.5 to 5.0) was performed to see if a stronger fluency incentive could fix this.
*   **Finding:** The experiment definitively proved that **increasing the fluency weight was not a viable solution**. The pathological behavior was a deeply stable local minimum, and a more structural change to the reward function was required.

### Current Status & Next Steps
The project has successfully identified and diagnosed a sophisticated reward hacking problem. The code has been updated to include the **"Answer Quality Reward"** component, which is specifically designed to close this loophole.

The next phase of experimentation is to run a new sweep to tune the parameters for this new reward component (`answer_quality_weight` and `answer_failure_penalty`) and confirm that it guides the model to generate coherent, meaningful, and effective contradictory questions.

## Key Technologies
*   **Framework:** PyTorch Lightning
*   **Models:** `transformers` (GPT-2, DeBERTa-v3), `peft` (LoRA)
*   **RL Algorithm:** Generative Flow Networks (GFlowNets)
*   **Distributed Training:** `torch.distributed` (DDP)
*   **Configuration:** Hydra
*   **Experiment Tracking:** Weights & Biases
*   **Infrastructure:** Redis (for Replay Buffer)

## Setup and Usage

### Prerequisites
*   NVIDIA GPUs with at least 24GB VRAM (for `gpt2-large`)
*   Conda or another virtual environment manager
*   A running Redis server (`localhost:6379` by default)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> && cd <repo-name>
    ```
2.  **Create Conda environment:**
    ```bash
    conda create -n gfn_rl python=3.11
    conda activate gfn_rl
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare data and models:**
    ```bash
    python convert_data.py
    python download_models.py
    ```

### Running Experiments
The project includes several bash scripts to orchestrate training:
*   **`run_ddp_debug_fast.sh`:** A fast, 4-GPU DDP script for quick sanity checks and debugging. Runs for only 10 steps on a small data subset.
*   **`run_hybrid_reward_diag.sh`:** A full 10-epoch diagnostic run using the hybrid reward strategy.
*   **`run_answer_quality_sweep.sh`:** (Or similar) The primary script for running hyperparameter sweeps to test new strategies.

To launch a run, simply execute the desired script:
```bash
# Example: Launching the fast debug script
./run_ddp_debug_fast.sh

Project Structure

.
├── configs/                # Hydra configuration files
│   ├── callbacks/
│   ├── config.yaml         # Main configuration entry point
│   └── ...
├── data/                   # Datasets
├── gflownet/               # GFlowNet core logic
│   ├── loss.py
│   ├── replay_buffer_scalable.py
│   └── trajectory.py
├── training_logs/          # Directory for output logs from shell scripts
├── checkpoints/            # Default directory for model checkpoints
├── convert_data.py         # Script to prepare the dataset
├── download_models.py      # Script to pre-cache Hugging Face models
├── evaluation.py           # Script for post-training qualitative analysis
├── lightning_data_scalable.py # PyTorch Lightning DataModule
├── lightning_module.py     # The main PyTorch Lightning training module
├── reward.py               # The complex, multi-component reward function
├── train.py                # Main script to launch training
├── requirements.txt        # Python dependencies
├── run_ddp_debug_fast.sh   # Debugging script
└── run_..._sweep.sh       # Scripts for running experiment sweeps