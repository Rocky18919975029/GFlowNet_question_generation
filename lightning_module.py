# lightning_module.py

import random
from functools import partial
import torch
import pytorch_lightning as pl
import pandas as pd
from gflownet.trajectory import generate_and_return_termination_logprob
from gflownet.loss import modified_subtb_loss
from utils import get_termination_vals

class ContradictionGFNTask(pl.LightningModule):
    def __init__(
        self, model, tokenizer, reward_model, reward_buffer, task_prompt: str,
        likelihood_weight: float, n_samples: int, lr: float, subtb_lambda: float,
        pf_temp_high: float, pf_temp_low: float, pf_temp_prob: float,
        use_buffer_prob: float, min_question_len: int, max_question_len: int,
        reward_temp_start: float, reward_temp_end: float, reward_temp_horizon: int,
        train_probes: list = None, val_probes: list = None, use_4bit: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer", "reward_model", "reward_buffer"])
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

    def forward(self, prompt_tokens, z_prime_text, n_samples=None, pf_temperature=1.0, action_seq=None):
        prompt_length = prompt_tokens.shape[1]
        reward_fn = partial(
            self.reward_model.score,
            z_prime_text=z_prime_text,
            prompt_length=prompt_length,
        )
        return generate_and_return_termination_logprob(
            model=self.model, encoded_prompt=prompt_tokens, reward_fn=reward_fn,
            termination_token_id=self.end_of_question_token_id,
            max_len=self.hparams.max_question_len, min_len=self.hparams.min_question_len,
            temperature=pf_temperature, action_seq=action_seq,
        )

    def training_step(self, batch, batch_idx):
        model_device = next(self.model.parameters()).device 
        z_prime_tokens = batch["z_prime"].to(model_device)
        subject_tokens = batch["subject"].to(model_device)
        z_prime_text = self.tokenizer.decode(z_prime_tokens, skip_special_tokens=True)
        subject_text = self.tokenizer.decode(subject_tokens, skip_special_tokens=True)
        
        formatted_task_prompt = self.hparams.task_prompt.format(subject=subject_text)
        prompt_tokens_single = self.tokenizer(formatted_task_prompt, return_tensors="pt")["input_ids"][0].to(model_device)
        
        prompt_tokens = prompt_tokens_single.unsqueeze(0).expand(self.hparams.n_samples, -1)
        
        if random.random() < self.hparams.use_buffer_prob and self.reward_buffer.sample(prompt_tokens_single, self.hparams.n_samples)[0] is not None:
            action_seq, log_r = self.reward_buffer.sample(prompt_tokens_single, self.hparams.n_samples)
            action_seq, log_r = action_seq.to(model_device), log_r.to(model_device)
            generated_trajectories, log_pf, log_pterm, _, _ = self.forward(
                prompt_tokens, z_prime_text=z_prime_text, action_seq=action_seq)
            gen_len = generated_trajectories.shape[1] - len(prompt_tokens_single)
            log_r = log_r[:, :gen_len]
            log_r *= 1 / self.reward_model.temperature
        else:
            pf_temp = 1.0
            if random.random() < self.hparams.pf_temp_prob:
                pf_temp = random.uniform(self.hparams.pf_temp_low, self.hparams.pf_temp_high)
            generated_trajectories, log_pf, log_pterm, log_r, _ = self.forward(
                prompt_tokens, z_prime_text=z_prime_text, pf_temperature=pf_temp)
            self.reward_buffer.add_batch(
                prompt_tokens=prompt_tokens_single,
                generated_questions=generated_trajectories[:, len(prompt_tokens_single):],
                logrewards=log_r * self.reward_model.temperature,
                tokenizer=self.tokenizer,
            )
        
        loss = modified_subtb_loss(
            log_pf=log_pf, log_r=log_r, log_pterm=log_pterm,
            generated_text=generated_trajectories,
            termination_token_id=self.end_of_question_token_id,
            prompt_len=len(prompt_tokens_single),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        _, last_log_r, _, question_len = get_termination_vals(
            generated_text=generated_trajectories, log_pf=log_pf, log_pterm=log_pterm, log_r=log_r,
            log_r_unpenalized=None, termination_token_id=self.end_of_question_token_id, prompt_len=len(prompt_tokens_single))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.n_samples)
        self.log("train/avg_log_reward", last_log_r.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.n_samples)
        self.log("train/avg_question_len", question_len.float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.n_samples)
        return loss

    def validation_step(self, batch, batch_idx):
        model_device = next(self.model.parameters()).device
        z_prime_tokens = batch["z_prime"].to(model_device)
        subject_tokens = batch["subject"].to(model_device)
        z_prime_text = self.tokenizer.decode(z_prime_tokens, skip_special_tokens=True)
        subject_text = self.tokenizer.decode(subject_tokens, skip_special_tokens=True)
        
        formatted_task_prompt = self.hparams.task_prompt.format(subject=subject_text)
        prompt_tokens_single = self.tokenizer(formatted_task_prompt, return_tensors="pt")["input_ids"][0].to(model_device)
        prompt_tokens = prompt_tokens_single.unsqueeze(0).expand(self.hparams.n_samples, -1)
        generated_trajectories, log_pf, log_pterm, log_r, _ = self.forward(
            prompt_tokens, z_prime_text=z_prime_text)
        
        loss = modified_subtb_loss(
            log_pf=log_pf, log_r=log_r, log_pterm=log_pterm,
            generated_text=generated_trajectories,
            termination_token_id=self.end_of_question_token_id,
            prompt_len=len(prompt_tokens_single),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        _, last_log_r, _, question_len = get_termination_vals(
            generated_text=generated_trajectories, log_pf=log_pf, log_pterm=log_pterm, log_r=log_r, 
            log_r_unpenalized=None, termination_token_id=self.end_of_question_token_id, prompt_len=len(prompt_tokens_single))
        self.log("val/loss", loss, sync_dist=True, batch_size=self.hparams.n_samples)
        self.log("val/avg_log_reward", last_log_r.mean(), sync_dist=True, batch_size=self.hparams.n_samples)
        self.log("val/avg_question_len", question_len.float().mean(), sync_dist=True, batch_size=self.hparams.n_samples)
        
        if batch_idx == 0 and self.hparams.val_probes is not None and self.logger is not None and self.trainer.current_epoch % 5 == 0:
            samples_table = self.sample_probes(self.hparams.val_probes)
            self.logger.log_table(key="samples/validation_probes", dataframe=samples_table)

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

    def sample_probes(self, probes, n_samples=4):
        samples = []
        model_device = next(self.model.parameters()).device
        for probe in probes:
            z_tokens, z_prime_tokens, subject_tokens = probe['z'].to(model_device), probe['z_prime'].to(model_device), probe['subject'].to(model_device)
            z_text = self.tokenizer.decode(z_tokens, skip_special_tokens=True)
            z_prime_text = self.tokenizer.decode(z_prime_tokens, skip_special_tokens=True)
            subject_text = self.tokenizer.decode(subject_tokens, skip_special_tokens=True)
            formatted_task_prompt = self.hparams.task_prompt.format(subject=subject_text)
            prompt_tokens_single = self.tokenizer(formatted_task_prompt, return_tensors="pt")["input_ids"][0].to(model_device)
            prompt_tokens = prompt_tokens_single.unsqueeze(0).expand(n_samples, -1)
            
            with torch.no_grad():
                generated_text, _, _, log_r, _ = self.forward(prompt_tokens, z_prime_text=z_prime_text, n_samples=n_samples)
            
            _, log_rewards, _, _ = get_termination_vals(
                generated_text=generated_text, log_pf=None, log_pterm=None, log_r=log_r, 
                log_r_unpenalized=None, termination_token_id=self.end_of_question_token_id, prompt_len=len(prompt_tokens_single))
            
            generated_questions = generated_text[:, len(prompt_tokens_single):]
            decoded_questions = self.tokenizer.batch_decode(generated_questions, skip_special_tokens=True)
            
            for i in range(len(decoded_questions)):
                samples.append({
                    "Original Fact (z)": z_text, "Edited Fact (z')": z_prime_text,
                    "Subject": subject_text, "Sampled Question (x)": decoded_questions[i],
                    "Log Reward": log_rewards[i].item(),
                })
        
        return pd.DataFrame(samples).sort_values(by=["Original Fact (z)", "Log Reward"], ascending=[True, False])

    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)