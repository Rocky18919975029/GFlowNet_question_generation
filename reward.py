# reward.py

import torch
from torch.nn.functional import log_softmax
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

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
        use_hybrid_reward: bool = False,
        penalized_reward_end_step: int = 0,
        contradiction_threshold: float = -99.0,
        failure_penalty: float = -99.0,
        answer_quality_weight: float = 0.0,
        answer_quality_threshold: float = -99.0,
        answer_failure_penalty: float = 0.0,
        device: torch.device = None,
    ):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        
        self.nli_model_name = nli_model_name
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name, num_labels=3
        )
        self.nli_model.eval()
        self.contradiction_label_id = 2 

        self.diversity_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.likelihood_weight = likelihood_weight
        self.answer_prompt_template = answer_prompt_template
        self.termination_token_id = termination_token_id
        self.min_len = min_len
        self.temperature = temperature
        self.nli_batch_size = nli_batch_size
        
        self.use_hybrid_reward = use_hybrid_reward
        self.penalized_reward_end_step = penalized_reward_end_step
        self.contradiction_threshold = contradiction_threshold
        self.failure_penalty = failure_penalty
        
        self.answer_quality_weight = answer_quality_weight
        self.answer_quality_threshold = answer_quality_threshold
        self.answer_failure_penalty = answer_failure_penalty
        
        self.device = device

    @torch.no_grad()
    def score(self, generated_trajectories, z_prime_text: str, prompt_length: int, current_step: int = None):
        batch_size, seq_len = generated_trajectories.shape
        gen_len = seq_len - prompt_length

        if gen_len <= 0:
            zeros = torch.zeros(batch_size, 1, device=self.base_model.device)
            reward_tuple = (zeros, zeros, zeros, zeros, zeros)
            answers_text = [""] * batch_size
            return reward_tuple, answers_text

        final_questions_toks = generated_trajectories[:, prompt_length:]
        final_questions_text = self.base_tokenizer.batch_decode(final_questions_toks, skip_special_tokens=True)
        
        full_prompts_for_answers = [self.answer_prompt_template.format(question=q) for q in final_questions_text]
        answers_text = self._generate_answers(full_prompts_for_answers)
        
        log_c = self._calculate_contradiction_score(answers_text, z_prime_text)
        log_p_question = self._calculate_log_p_likelihood(generated_trajectories, prompt_length)
        
        log_p_answer = torch.zeros_like(log_p_question)
        if self.answer_quality_weight != 0 or self.answer_failure_penalty != 0:
             log_p_answer = self._calculate_log_p_answer(full_prompts_for_answers, answers_text)

        is_penalized_phase = self.use_hybrid_reward and current_step is not None and current_step < self.penalized_reward_end_step

        if is_penalized_phase:
            unscaled_reward = log_p_question.clone()
            passes_check = log_c > self.contradiction_threshold
            unscaled_reward[~passes_check] += self.failure_penalty
        else:
            unscaled_reward = log_c + self.likelihood_weight * log_p_question + self.answer_quality_weight * log_p_answer

        unscaled_penalized_reward = unscaled_reward.clone()

        if self.answer_failure_penalty != 0:
            low_quality_mask = log_p_answer < self.answer_quality_threshold
            unscaled_penalized_reward[low_quality_mask] += self.answer_failure_penalty

        actual_lengths = (final_questions_toks != self.base_tokenizer.pad_token_id).sum(dim=-1)
        len_penalty_mask = actual_lengths < self.min_len
        unscaled_penalized_reward[len_penalty_mask] = -99.0

        log_reward_final_scaled = unscaled_penalized_reward / self.temperature
        
        # --- THIS IS THE MODIFIED RETURN STATEMENT ---
        reward_tuple = (
            log_reward_final_scaled.unsqueeze(1), 
            unscaled_penalized_reward.unsqueeze(1), 
            log_c.unsqueeze(1), 
            log_p_question.unsqueeze(1),
            log_p_answer.unsqueeze(1)
        )
        return reward_tuple, answers_text
    
    @torch.no_grad()
    def calculate_semantic_diversity(self, texts: list[str]):
        if len(texts) < 2:
            return 0.0

        embeddings = self.diversity_model.encode(
            texts, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        similarity_matrix = torch.mm(embeddings, embeddings.t())
        
        upper_triangle_indices = torch.triu_indices(len(texts), len(texts), offset=1)
        pairwise_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
        
        average_distance = (1 - pairwise_similarities).mean().item()
        return average_distance

    def _calculate_log_p_likelihood(self, trajectories, prompt_length: int):
        logits = self.base_model(trajectories).logits
        question_logits = logits[:, prompt_length - 1 : -1]
        question_tokens = trajectories[:, prompt_length:]
        
        log_probs = question_logits.log_softmax(-1)
        token_log_probs = log_probs.gather(-1, question_tokens.unsqueeze(-1)).squeeze(-1)
        
        mask = (question_tokens != self.base_tokenizer.pad_token_id)
        token_log_probs = token_log_probs * mask
        
        return token_log_probs.sum(dim=-1)

    def _calculate_log_p_answer(self, full_prompts_for_answers: list[str], answers: list[str]):
        safe_answers = [ans if ans else " " for ans in answers]
        full_sequences = [prompt + ans for prompt, ans in zip(full_prompts_for_answers, safe_answers)]
        
        inputs = self.base_tokenizer(full_sequences, return_tensors='pt', padding=True, truncation=True, max_length=self.base_model.config.n_positions).to(self.base_model.device)
        prompt_inputs = self.base_tokenizer(full_prompts_for_answers, return_tensors='pt', padding=True, truncation=True, max_length=self.base_model.config.n_positions)
        prompt_lengths = prompt_inputs.input_ids.shape[1]
        
        with torch.no_grad():
            logits = self.base_model(**inputs).logits

        answer_logits = logits[:, prompt_lengths - 1 : -1]
        answer_tokens = inputs.input_ids[:, prompt_lengths:]
        
        log_probs = answer_logits.log_softmax(-1)
        token_log_probs = log_probs.gather(-1, answer_tokens.unsqueeze(-1)).squeeze(-1)
        
        mask = (answer_tokens != self.base_tokenizer.pad_token_id)
        token_log_probs = token_log_probs * mask
        
        return token_log_probs.sum(dim=-1)

    def _calculate_contradiction_score(self, answers: list[str], z_prime_text: str):
        self.nli_model.to(self.base_model.device)
        safe_answers = [ans if ans else "no answer" for ans in answers]
        premise_hypothesis_pairs = [[z_prime_text, ans] for ans in safe_answers]

        all_log_c_scores = []
        for i in range(0, len(premise_hypothesis_pairs), self.nli_batch_size):
            batch = premise_hypothesis_pairs[i : i + self.nli_batch_size]
            inputs = self.nli_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.nli_model.device)
            logits = self.nli_model(**inputs).logits
            log_probs = log_softmax(logits, dim=-1)
            all_log_c_scores.append(log_probs[:, self.contradiction_label_id])
        
        return torch.cat(all_log_c_scores)
        
    def _generate_answers(self, full_prompts: list[str]) -> list[str]:
        answers = []
        for i in range(0, len(full_prompts), self.nli_batch_size):
            batch_prompts = full_prompts[i : i + self.nli_batch_size]
            inputs = self.base_tokenizer(batch_prompts, return_tensors='pt', padding=True).to(self.base_model.device)
            
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=30,
                min_new_tokens=2, 
                eos_token_id=self.base_tokenizer.eos_token_id,
                pad_token_id=self.base_tokenizer.eos_token_id,
                do_sample=False
            )
            
            new_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            decoded_answers = self.base_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            cleaned_answers = [ans.strip().split('\n')[0] for ans in decoded_answers]
            answers.extend(cleaned_answers)
        return answers