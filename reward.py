import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# The `utils` import is no longer needed as we've removed lora_to_base/base_to_lora
# from utils import lora_to_base, base_to_lora

class ContradictionReward:
    def __init__(
        self,
        base_model,
        base_tokenizer,
        nli_model_name: str,
        likelihood_weight: float,
        answer_prompt_template: str,
        termination_token_id: int,
        min_len: int,
        temperature: float = 1.0,
        nli_batch_size: int = 8,
        device: torch.device = None, # This will be set by the lightning module
    ):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        
        # We will move the NLI model to the correct device inside the LightningModule
        # to ensure it's on the same GPU as the data.
        self.nli_model_name = nli_model_name
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name, num_labels=3
        )
        self.nli_model.eval()
        self.contradiction_label_id = 2 

        self.likelihood_weight = likelihood_weight
        self.answer_prompt_template = answer_prompt_template
        self.termination_token_id = termination_token_id
        self.min_len = min_len
        self.temperature = temperature
        self.nli_batch_size = nli_batch_size
        self.device = device # Will be set later

    @torch.no_grad()
    def score(self, generated_trajectories, z_prime_text: str, prompt_length: int):
        """
        Calculates a single reward value for each completed trajectory in the batch.
        This is much more efficient than calculating rewards for every intermediate step.
        """
        # --- REFACTOR GOAL 2: Remove State Switching ---
        # The `lora_to_base` and `base_to_lora` calls are gone. The `base_model` is now
        # a separate, dedicated model for reward calculation.

        batch_size, seq_len = generated_trajectories.shape
        gen_len = seq_len - prompt_length

        # If no tokens were generated, return zero reward.
        if gen_len <= 0:
            return torch.zeros(batch_size, 0, device=self.base_model.device), None

        # --- REFACTOR GOAL 1: Eliminate N*L Problem ---
        # We now only operate on the final, complete questions.
        
        # 1. Decode the final questions from the trajectories
        final_questions_toks = generated_trajectories[:, prompt_length:]
        final_questions_text = self.base_tokenizer.batch_decode(final_questions_toks, skip_special_tokens=True)

        # 2. Generate a single answer for each final question
        answers_text = self._generate_answers(final_questions_text)

        # 3. Calculate a single contradiction score for each question-answer pair
        log_c = self._calculate_contradiction_score(answers_text, z_prime_text)

        # 4. Calculate a single likelihood score for each final question
        log_p_likelihood = self._calculate_log_p_likelihood(generated_trajectories, prompt_length)
        
        # 5. Combine scores into a single reward value per trajectory
        log_reward_final = log_c + self.likelihood_weight * log_p_likelihood
        
        # 6. Apply length penalty to the final reward
        # We check the actual length of generated tokens against min_len.
        actual_lengths = (final_questions_toks != self.base_tokenizer.pad_token_id).sum(dim=-1)
        len_penalty_mask = actual_lengths < self.min_len
        log_reward_final[len_penalty_mask] = -99.0

        # 7. Scale final reward by temperature
        log_reward_final /= self.temperature
        
        # For the GFlowNet loss, this single final reward is broadcast across all steps of the trajectory.
        # The reward for intermediate states is considered to be the same as the final state.
        # Note: We return `log_reward_final.unsqueeze(1)` to create a [batch_size, 1] tensor
        # which can then be broadcasted to the full trajectory length in the LightningModule.
        return log_reward_final.unsqueeze(1), None # Return None for unpenalized reward for simplicity

    def _calculate_log_p_likelihood(self, trajectories, prompt_length: int):
        """Calculates log P(x|prompt) for the full sequence."""
        # This function now calculates the log probability of the entire sequence, not a cumsum.
        logits = self.base_model(trajectories).logits
        # We only care about the logits for the generated part
        question_logits = logits[:, prompt_length - 1 : -1]
        question_tokens = trajectories[:, prompt_length:]
        
        log_probs = question_logits.log_softmax(-1)
        
        # Gather the log-probs of the generated tokens
        token_log_probs = log_probs.gather(-1, question_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens so they don't contribute to the sum
        mask = (question_tokens != self.base_tokenizer.pad_token_id)
        token_log_probs = token_log_probs * mask
        
        # The final likelihood is the sum of the log-probs of the generated tokens.
        return token_log_probs.sum(dim=-1)

    def _calculate_contradiction_score(self, answers: list[str], z_prime_text: str):
        """Calculates a single contradiction score for each answer against the premise."""
        
        # The NLI model should be on the same device as the base_model.
        self.nli_model.to(self.base_model.device)
        
        # Create premise-hypothesis pairs. The premise is always z_prime.
        # The hypothesis is the full-sentence answer.
        premise_hypothesis_pairs = [[z_prime_text, ans] for ans in answers]

        all_log_c_scores = []
        for i in range(0, len(premise_hypothesis_pairs), self.nli_batch_size):
            batch = premise_hypothesis_pairs[i : i + self.nli_batch_size]
            inputs = self.nli_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.nli_model.device)
            logits = self.nli_model(**inputs).logits
            log_probs = log_softmax(logits, dim=-1)
            all_log_c_scores.append(log_probs[:, self.contradiction_label_id])
        
        return torch.cat(all_log_c_scores)
        
    def _generate_answers(self, questions: list[str]) -> list[str]:
        """Generates a clean, full-sentence answer for each question in the batch."""
        answers = []
        full_prompts = [self.answer_prompt_template.format(question=q) for q in questions]

        for i in range(0, len(full_prompts), self.nli_batch_size):
            batch_prompts = full_prompts[i : i + self.nli_batch_size]
            inputs = self.base_tokenizer(batch_prompts, return_tensors='pt', padding=True).to(self.base_model.device)
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=30,
                pad_token_id=self.base_tokenizer.eos_token_id,
                eos_token_id=self.base_tokenizer.encode("\n")[0],
                do_sample=False,
            )
            new_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            decoded_answers = self.base_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            cleaned_answers = [ans.strip().split('\n')[0] for ans in decoded_answers]
            answers.extend(cleaned_answers)
        return answers