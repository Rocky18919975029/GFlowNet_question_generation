import random
from functools import partial
import torch
import pytorch_lightning as pl
import pandas as pd
from gflownet.trajectory import generate_and_return_termination_logprob
from gflownet.loss import modified_subtb_loss
from utils import get_termination_vals
import os 


class ContradictionGFNTask(pl.LightningModule):
    def __init__(
        self, model, tokenizer, reward_model, reward_buffer, task_prompt: str,
        likelihood_weight: float, n_samples: int, lr: float, subtb_lambda: float,
        pf_temp_high: float, pf_temp_low: float, pf_temp_prob: float,
        use_buffer_prob: float, min_question_len: int, max_question_len: int,
        reward_temp_start: float, reward_temp_end: float, reward_temp_horizon: int,
        n_probes: int, 
        checkpoint_save_interval: int, 
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
        self.reward_model.device = self.device

    def forward(self, prompt_tokens, z_prime_text, reward_fn, n_samples=None, pf_temperature=1.0, action_seq=None):
        prompt_length = prompt_tokens.shape[1]
        
        if action_seq is not None:
            action_seq = action_seq.to(self.device)

        return generate_and_return_termination_logprob(
            model=self.model, encoded_prompt=prompt_tokens.to(self.device), reward_fn=reward_fn,
            termination_token_id=self.end_of_question_token_id,
            max_len=self.hparams.max_question_len, min_len=self.hparams.min_question_len,
            temperature=pf_temperature, action_seq=action_seq,
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

        # --- Robust DDP Logic: Centralize and Broadcast ---
        
        if self.trainer.is_global_zero:
            use_buffer = random.random() < self.hparams.use_buffer_prob
            
            if use_buffer:
                buffer_sample = self.reward_buffer.sample(prompt_tokens_single, self.hparams.n_samples)
                if buffer_sample and buffer_sample[0] is not None:
                    action_seq_rank0, log_r_final_rank0 = buffer_sample
                else:
                    use_buffer = False
            
            if not use_buffer:
                pf_temp = 1.0
                if random.random() < self.hparams.pf_temp_prob:
                    pf_temp = random.uniform(self.hparams.pf_temp_low, self.hparams.pf_temp_high)
                
                real_reward_fn = partial(self.reward_model.score, z_prime_text=z_prime_text, prompt_length=prompt_len)
                
                generated_trajectories_rank0, _, _, log_r_final_rank0, _ = self.forward(
                    prompt_tokens, z_prime_text=z_prime_text, reward_fn=real_reward_fn, pf_temperature=pf_temp
                )
                action_seq_rank0 = generated_trajectories_rank0[:, prompt_len:]
                
                if action_seq_rank0.shape[1] > 0:
                    self.reward_buffer.add_batch(
                        prompt_tokens=prompt_tokens_single, 
                        generated_questions=action_seq_rank0, 
                        logrewards=log_r_final_rank0.squeeze(-1),
                        tokenizer=self.tokenizer
                    )
            
            action_seq_rank0 = action_seq_rank0.cpu()
            log_r_final_rank0 = log_r_final_rank0.view(-1, 1).cpu()

            objects_to_broadcast = [action_seq_rank0, log_r_final_rank0]
        else:
            objects_to_broadcast = [None, None]

        if self.trainer.world_size > 1:
            torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
        
        action_seq, log_r_final = objects_to_broadcast
        
        dummy_reward_fn = lambda trajectories: (None, None)
        generated_trajectories, log_pf, log_pterm, _, _ = self.forward(
            prompt_tokens, z_prime_text=z_prime_text, reward_fn=dummy_reward_fn, action_seq=action_seq
        )
        
        gen_len = generated_trajectories.shape[1] - prompt_len
        if gen_len > 0:
            log_r = log_r_final.to(self.device).expand(-1, gen_len)
        else:
            log_r = torch.empty(self.hparams.n_samples, 0, device=self.device)

        loss = modified_subtb_loss(log_pf=log_pf, log_r=log_r, log_pterm=log_pterm, generated_text=generated_trajectories, termination_token_id=self.end_of_question_token_id, prompt_len=prompt_len, subtb_lambda=self.hparams.subtb_lambda)
        
        _, last_log_r, _, question_len = get_termination_vals(generated_text=generated_trajectories, log_pf=log_pf, log_pterm=log_pterm, log_r=log_r, log_r_unpenalized=None, termination_token_id=self.end_of_question_token_id, prompt_len=prompt_len)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.n_samples)
        if last_log_r is not None:
             self.log("train/avg_log_reward", last_log_r.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.n_samples)
        self.log("train/avg_question_len", question_len.float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.n_samples)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # No checkpointing logic here. This is handled by the Trainer and Callbacks.
        pass

    def on_train_batch_start(self, batch, batch_idx):
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        lr = self.get_lr_at_step(self.global_step)
        self.reward_model.temperature = reward_temp
        if self.trainer.optimizers:
            for pg in self.trainer.optimizers[0].param_groups: pg["lr"] = lr
            
    def on_train_epoch_start(self):
        self.log("scheduled/reward_temperature", self.reward_model.temperature, sync_dist=True)
        if self.trainer.optimizers:
            self.log("scheduled/learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True)

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