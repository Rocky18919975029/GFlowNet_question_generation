# lightning_module.py

import random
from functools import partial
import torch
import pytorch_lightning as pl
import pandas as pd
from gflownet.trajectory import generate_and_return_termination_logprob
from gflownet.loss import modified_subtb_loss
from utils import get_termination_vals
import os 
import wandb


class ContradictionGFNTask(pl.LightningModule):
    def __init__(
        self,
        model, tokenizer, reward_model, reward_buffer, task_prompt: str,
        likelihood_weight: float, n_samples: int, lr: float, subtb_lambda: float,
        pf_temp_high: float, pf_temp_low: float, pf_temp_prob: float,
        use_buffer_prob: float, min_question_len: int, max_question_len: int,
        reward_temp_start: float, reward_temp_end: float, reward_temp_horizon: int,
        n_probes: int, 
        checkpoint_save_interval: int,
        log_every_n_steps: int,
        train_probes: list = None, 
        use_4bit: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer", "reward_model", "reward_buffer", "use_4bit", "train_probes"])
        self.use_4bit = use_4bit
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_buffer = reward_buffer
        self.get_lr_at_step = lambda step: min(step / 20 * self.hparams.lr, self.hparams.lr)
        self.get_reward_temp_at_step = lambda step: self.hparams.reward_temp_start + (
            self.hparams.reward_temp_end - self.hparams.reward_temp_start
        ) * min(1, step / self.hparams.reward_temp_horizon)
        try:
            self.end_of_question_token_id = self.tokenizer.encode("?", add_special_tokens=False)[-1]
        except:
            self.end_of_question_token_id = self.tokenizer.convert_tokens_to_ids("?")
        self.hparams.val_probes = None

    def setup(self, stage: str):
        self.reward_model.base_model.to(self.device)
        self.reward_model.nli_model.to(self.device)
        self.reward_model.diversity_model.to(self.device)
        self.reward_model.device = self.device

    def forward(self, prompt_tokens, z_prime_text, reward_fn, n_samples=None, pf_temperature=1.0, action_seq=None):
        prompt_length = prompt_tokens.shape[1]
        
        bound_reward_fn = partial(reward_fn, z_prime_text=z_prime_text, prompt_length=prompt_length) if reward_fn is not None else None

        return generate_and_return_termination_logprob(
            model=self.model, encoded_prompt=prompt_tokens.to(self.device), reward_fn=bound_reward_fn,
            termination_token_id=self.end_of_question_token_id,
            max_len=self.hparams.max_question_len, min_len=self.hparams.min_question_len,
            temperature=pf_temperature, action_seq=action_seq,
            skip_rewards=(action_seq is not None)
        )

    def training_step(self, batch, batch_idx):
        z_prime_tokens = batch["z_prime"].to(self.device)
        subject_tokens = batch["subject"].to(self.device)
        z_prime_text = self.tokenizer.decode(z_prime_tokens, skip_special_tokens=True)
        subject_text = self.tokenizer.decode(subject_tokens, skip_special_tokens=True)
        
        formatted_task_prompt = self.hparams.task_prompt.format(subject=subject_text)
        prompt_tokens_single = self.tokenizer(formatted_task_prompt, return_tensors="pt")["input_ids"][0]
        
        prompt_tokens = prompt_tokens_single.unsqueeze(0).expand(self.hparams.n_samples, -1)
        prompt_len = len(prompt_tokens_single)
        
        # --- Start of Rank 0 block ---
        if self.trainer.is_global_zero:
            use_buffer = random.random() < self.hparams.use_buffer_prob
            
            action_seq_rank0, log_r_final_rank0, log_r_unscaled_rank0 = None, None, None
            # Initialize these to None, or actual default values if they are expected by subsequent operations
            log_c_rank0, log_p_likelihood_rank0 = None, None 
            semantic_diversity_rank0 = 0.0 # Default value if not computed

            if use_buffer:
                buffer_sample = self.reward_buffer.sample(prompt_tokens_single, self.hparams.n_samples)
                if buffer_sample and buffer_sample[0] is not None:
                    action_seq_rank0, log_r_unscaled_rank0 = buffer_sample
                    log_r_final_rank0 = log_r_unscaled_rank0 / self.reward_model.temperature
                    # --- NEW: For buffer samples, set component rewards to the total unscaled reward ---
                    # This ensures log_c_rank0 and log_p_likelihood_rank0 are not None for logging.
                    # We can't actually decompose the buffer reward, so we'll just log the total as a proxy.
                    log_c_rank0 = log_r_unscaled_rank0
                    log_p_likelihood_rank0 = torch.zeros_like(log_r_unscaled_rank0) # Or another suitable default if you prefer.
                else:
                    use_buffer = False
            
            if not use_buffer:
                pf_temp = 1.0
                if random.random() < self.hparams.pf_temp_prob:
                    pf_temp = random.uniform(self.hparams.pf_temp_low, self.hparams.pf_temp_high)
                
                generated_trajectories_rank0, _, _, reward_tuple, _ = self.forward(
                    prompt_tokens, z_prime_text=z_prime_text, reward_fn=self.reward_model.score, pf_temperature=pf_temp
                )
                
                log_r_final_rank0, log_r_unscaled_rank0, log_c_rank0, log_p_likelihood_rank0 = reward_tuple
                
                action_seq_rank0 = generated_trajectories_rank0[:, prompt_len:]
                
                decoded_questions = self.tokenizer.batch_decode(action_seq_rank0, skip_special_tokens=True)
                semantic_diversity_rank0 = self.reward_model.calculate_semantic_diversity(decoded_questions)
                
                if action_seq_rank0.shape[1] > 0:
                    self.reward_buffer.add_batch(
                        prompt_tokens=prompt_tokens_single, 
                        generated_questions=action_seq_rank0, 
                        logrewards=log_r_unscaled_rank0.squeeze(-1),
                        tokenizer=self.tokenizer
                    )
            
            # --- Ensure all broadcasted tensors are on CPU before packing ---
            action_seq_rank0 = action_seq_rank0.cpu() if action_seq_rank0 is not None else torch.empty(0, 0, dtype=torch.long)
            log_r_final_rank0 = log_r_final_rank0.cpu() if log_r_final_rank0 is not None else torch.empty(0)
            log_r_unscaled_rank0 = log_r_unscaled_rank0.cpu() if log_r_unscaled_rank0 is not None else torch.empty(0)
            log_c_rank0 = log_c_rank0.cpu() if log_c_rank0 is not None else torch.empty(0)
            log_p_likelihood_rank0 = log_p_likelihood_rank0.cpu() if log_p_likelihood_rank0 is not None else torch.empty(0)
            
            objects_to_broadcast = [
                action_seq_rank0, log_r_final_rank0, log_r_unscaled_rank0,
                log_c_rank0, log_p_likelihood_rank0, semantic_diversity_rank0
            ]
        # --- End of Rank 0 block ---
        else: # For ranks != 0, initialize placeholders
            objects_to_broadcast = [None, None, None, None, None, None]

        if self.trainer.world_size > 1:
            torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
        
        (action_seq, log_r_final, log_r_unscaled, 
         log_c, log_p_likelihood, semantic_diversity) = objects_to_broadcast
        
        # --- Move all broadcasted tensors to the correct device on each rank ---
        # Ensure that empty tensors are handled if they come from broadcast
        action_seq = action_seq.to(self.device) if action_seq.numel() > 0 else torch.empty(0, 0, dtype=torch.long, device=self.device)
        log_r_final = log_r_final.to(self.device) if log_r_final.numel() > 0 else torch.empty(0, device=self.device)
        log_r_unscaled = log_r_unscaled.to(self.device) if log_r_unscaled.numel() > 0 else torch.empty(0, device=self.device)
        log_c = log_c.to(self.device) if log_c.numel() > 0 else torch.empty(0, device=self.device)
        log_p_likelihood = log_p_likelihood.to(self.device) if log_p_likelihood.numel() > 0 else torch.empty(0, device=self.device)


        generated_trajectories, log_pf, log_pterm, _, _ = self.forward(
            prompt_tokens, z_prime_text=z_prime_text, reward_fn=None, action_seq=action_seq
        )
        
        gen_len = generated_trajectories.shape[1] - prompt_len
        if gen_len > 0:
            log_r = log_r_final.view(-1, 1).expand(-1, gen_len)
            log_r_unpenalized = log_r_unscaled.view(-1, 1).expand(-1, gen_len)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train/loss", loss)
            return loss

        loss = modified_subtb_loss(log_pf=log_pf, log_r=log_r, log_pterm=log_pterm, generated_text=generated_trajectories, termination_token_id=self.end_of_question_token_id, prompt_len=prompt_len, subtb_lambda=self.hparams.subtb_lambda)
        
        _, last_log_r, last_log_r_unpenalized, question_len = get_termination_vals(generated_text=generated_trajectories, log_pf=log_pf, log_pterm=log_pterm, log_r=log_r, log_r_unpenalized=log_r_unpenalized, termination_token_id=self.end_of_question_token_id, prompt_len=prompt_len)

        # --- Logging Block (All quantitative logs are now rank_zero_only) ---
        # The batch_size parameter for log calls is largely decorative for rank_zero_only, 
        # but is kept for consistency if you later decide to enable DDP sync for a metric.
        batch_size_log = self.hparams.n_samples # This is the batch size *per step on rank 0*
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, rank_zero_only=True)
        if last_log_r is not None:
             self.log("train/avg_log_reward_scaled", last_log_r.mean(), on_step=True, on_epoch=True, rank_zero_only=True)
        if last_log_r_unpenalized is not None:
             self.log("train/avg_log_reward_unscaled", last_log_r_unpenalized.mean(), on_step=True, on_epoch=True, rank_zero_only=True)
        self.log("train/avg_question_len", question_len.float().mean(), on_step=True, on_epoch=True, rank_zero_only=True)
        
        # These now rely on `log_c` and `log_p_likelihood` being properly set in Rank 0 block
        if log_c.numel() > 0:
            self.log("train/avg_log_contradiction", log_c.mean(), on_step=True, on_epoch=True, rank_zero_only=True)
        if log_p_likelihood.numel() > 0:
            self.log("train/avg_log_likelihood", log_p_likelihood.mean(), on_step=True, on_epoch=True, rank_zero_only=True)
        
        # semantic_diversity is already calculated on rank 0 and broadcasted for completeness, 
        # but only logged from rank 0, for step-level.
        self.log("train/semantic_diversity", semantic_diversity, on_step=True, on_epoch=False, rank_zero_only=True)

        # --- Qualitative Logging: Also rank_zero_only ---
        if self.trainer.is_global_zero and self.global_step > 0 and self.global_step % self.hparams.log_every_n_steps == 0:
            questions = self.tokenizer.batch_decode(generated_trajectories[:, prompt_len:], skip_special_tokens=True)
            
            columns = ["Step", "Subject", "Generated Question", "Unscaled Reward", "Contradiction Score", "Likelihood Score"]
            table = wandb.Table(columns=columns)

            for i in range(min(self.hparams.n_probes, len(questions))):
                table.add_data(
                    self.global_step,
                    subject_text,
                    questions[i],
                    log_r_unscaled[i].item() if log_r_unscaled.numel() > i else "N/A",
                    log_c[i].item() if log_c.numel() > i else "N/A",
                    log_p_likelihood[i].item() if log_p_likelihood.numel() > i else "N/A"
                )
            self.logger.experiment.log({"generation_probes": table})

        return loss

    def validation_step(self, batch, batch_idx):
        pass
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass
    def on_train_batch_start(self, batch, batch_idx):
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        lr = self.get_lr_at_step(self.global_step)
        self.reward_model.temperature = reward_temp
        if self.trainer.optimizers:
            for pg in self.trainer.optimizers[0].param_groups: pg["lr"] = lr
    _on_train_epoch_start_logged = False
    def on_train_epoch_start(self):
        if not self._on_train_epoch_start_logged:
            # These are epoch-level scheduled metrics, they can be sync_dist
            self.log("scheduled/reward_temperature", self.reward_model.temperature, sync_dist=True)
            if self.trainer.optimizers:
                self.log("scheduled/learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True)
            self._on_train_epoch_start_logged = True
    def on_fit_start(self):
        pass
    def sample_probes(self, probes, n_samples=4):
        pass
    def configure_optimizers(self):
        if self.use_4bit:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)