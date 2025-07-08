# reward.py

import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import lora_to_base, base_to_lora

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
        device: torch.device = None,
    ):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        
        if device is None: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device
            
        print(f"Loading NLI model {nli_model_name} to device {self.device}...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name, num_labels=3
        ).to(self.device)
        self.nli_model.eval()
        self.contradiction_label_id = 2 

        self.likelihood_weight = likelihood_weight
        self.answer_prompt_template = answer_prompt_template
        self.termination_token_id = termination_token_id
        self.min_len = min_len
        self.temperature = temperature
        self.nli_batch_size = nli_batch_size

    @torch.no_grad()
    def score(self, generated_trajectories, z_prime_text: str, prompt_length: int):
        training_state = self.base_model.training
        lora_to_base(self.base_model)

        # 1. Calculate log P(x | prompt) using the base model
        log_p_likelihood = self._calculate_log_p_likelihood(generated_trajectories, prompt_length)

        # 2. Calculate contradiction score
        log_c, _ = self._calculate_contradiction_score(generated_trajectories, z_prime_text, prompt_length)

        # 3. Combine the two components
        log_reward = log_c + self.likelihood_weight * log_p_likelihood
        
        # 4. Apply hard length penalty
        reward_unpenalized = log_reward.clone()
        len_penalty_mask = torch.arange(log_reward.shape[1], device=log_reward.device) < self.min_len
        log_reward[:, len_penalty_mask] = -99

        # 5. Scale by temperature
        log_reward /= self.temperature
        reward_unpenalized /= self.temperature

        if training_state:
            base_to_lora(self.base_model)
            self.base_model.train()

        return log_reward, reward_unpenalized

    def _calculate_log_p_likelihood(self, trajectories, prompt_length: int):
        """Calculates log P(x | prompt) as a relevance/likelihood score."""
        logits = self.base_model(trajectories).logits
        question_logits = logits[:, prompt_length - 1 : -1]
        question_tokens = trajectories[:, prompt_length:]
        log_probs = question_logits.log_softmax(-1)
        log_p_sequence = log_probs.gather(-1, question_tokens.unsqueeze(-1)).squeeze(-1).cumsum(dim=-1)
        log_p_sequence = torch.cat([torch.zeros_like(log_p_sequence[:, :1]), log_p_sequence], dim=1)
        return log_p_sequence[:, :-1]

    def _calculate_contradiction_score(self, trajectories, z_prime_text, prompt_length):
        batch_size, seq_len = trajectories.shape
        self.nli_batch_size = batch_size
        gen_len = seq_len - prompt_length
        if gen_len <= 0: return torch.zeros(batch_size, 0, device=self.base_model.device), []

        partial_questions_text = [
            text for t in range(1, gen_len + 1) 
            for text in self.base_tokenizer.batch_decode(trajectories[:, prompt_length : prompt_length + t], skip_special_tokens=True)
        ]
        answers_text = self._generate_answers(partial_questions_text)
        
        premise_hypothesis_pairs = []
        for i in range(len(partial_questions_text)):
            question_stem = partial_questions_text[i].strip().replace("?", "")
            hypothesis = f"The answer to '{question_stem}' is {answers_text[i]}."
            premise_hypothesis_pairs.append([z_prime_text, hypothesis])

        contradiction_log_probs = []
        for i in range(0, len(premise_hypothesis_pairs), self.nli_batch_size):
            batch = premise_hypothesis_pairs[i : i + self.nli_batch_size]
            inputs = self.nli_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            logits = self.nli_model(**inputs).logits
            log_probs = log_softmax(logits, dim=-1)
            contradiction_log_probs.append(log_probs[:, self.contradiction_label_id].cpu())
        
        log_c_scores = torch.cat(contradiction_log_probs).reshape(gen_len, batch_size).T.to(self.base_model.device)
        return log_c_scores, answers_text
        
    def _generate_answers(self, questions: list[str]) -> list[str]:
        """Generates answers for a list of questions using a few-shot prompt from config."""
        answers = []
        full_prompts = [self.answer_prompt_template.format(question=q) for q in questions]

        for i in range(0, len(full_prompts), self.nli_batch_size):
            batch_prompts = full_prompts[i : i + self.nli_batch_size]
            inputs = self.base_tokenizer(batch_prompts, return_tensors='pt', padding=True).to(self.base_model.device)
            outputs = self.base_model.generate(
                **inputs, max_new_tokens=15, pad_token_id=self.base_tokenizer.eos_token_id,
                eos_token_id=self.base_tokenizer.encode("\n")[0], do_sample=False,
            )
            new_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            decoded_answers = self.base_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            cleaned_answers = [ans.strip().split('\n')[0] for ans in decoded_answers]
            answers.extend(cleaned_answers)
        return answers