# gflownet/replay_buffer.py

import heapq
import editdistance
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class ReplayBuffer:
    """
    A replay buffer that stores high-reward trajectories for each unique prompt.

    It uses a min-heap for each prompt to efficiently keep the `buffer_size`
    items with the highest rewards. It also enforces diversity by rejecting
    new items that are too similar to existing ones.
    """

    def __init__(self, buffer_size: int, termination_token_id: int, diversity_tolerance: float = 0.25):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.diversity_tolerance = diversity_tolerance
        self.reset()

    def reset(self):
        """Clears the buffer."""
        self._buffer = {}

    def add(self, item: dict):
        """
        Adds a single item to the buffer for a specific prompt.

        Args:
            item (dict): A dictionary containing 'str_prompt', 'logreward', 'tensor_sentence', etc.
        """
        str_prompt = item["str_prompt"]

        # If the generated sentence is already in the buffer for this prompt, ignore.
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return

        tokenized_sentence = [
            t for t in item["tensor_sentence"].tolist() if t != self.termination_token_id
        ]
        
        # Check for similarity with existing sentences in the buffer for this prompt
        for buffer_item in self._buffer[str_prompt]["sentences"]:
            # buffer_item is a tuple: (logreward, str_sentence, tensor_sentence, full_logrewards)
            tokenized_existing = [
                t for t in buffer_item[2].tolist() if t != self.termination_token_id
            ]
            
            # Calculate normalized edit distance
            dist = editdistance.eval(tokenized_sentence, tokenized_existing)
            norm_dist = dist / (len(tokenized_sentence) + len(tokenized_existing) + 1e-9)

            if norm_dist < self.diversity_tolerance:
                # If the new item is too similar but has a lower reward, ignore it.
                if buffer_item[0] >= item["logreward"]:
                    return
                # If it's similar but has a higher reward, replace the old one.
                else:
                    self._buffer[str_prompt]["sentences"].remove(buffer_item)
                    self._buffer[str_prompt]["exists"].remove(buffer_item[1])
                    heapq.heapify(self._buffer[str_prompt]["sentences"]) # Re-organize heap
                    break # Stop checking and proceed to add the new item

        new_entry = (
            item["logreward"],
            item["str_sentence"],
            item["tensor_sentence"],
            item["full_logrewards"],
        )
        self._buffer[str_prompt]["exists"].add(item["str_sentence"])

        if len(self._buffer[str_prompt]["sentences"]) >= self.buffer_size:
            # If buffer is full, push the new item and pop the one with the lowest reward
            popped = heapq.heappushpop(self._buffer[str_prompt]["sentences"], new_entry)
            self._buffer[str_prompt]["exists"].remove(popped[1])
        else:
            heapq.heappush(self._buffer[str_prompt]["sentences"], new_entry)

    def add_batch(self, prompt_tokens: torch.Tensor, generated_questions: torch.Tensor, logrewards: torch.Tensor, tokenizer):
        """
        Adds a batch of generated items to the buffer.

        Args:
            prompt_tokens (torch.Tensor): The tokenized prompt (e.g., [task_prompt] + [z]).
            generated_questions (torch.Tensor): The generated questions (the 'x' part).
            logrewards (torch.Tensor): The log-rewards for each step of each trajectory.
            tokenizer: The Hugging Face tokenizer.
        """
        # Use a string representation of the prompt tensor as a dictionary key
        str_prompt = " ".join([str(t) for t in prompt_tokens.tolist()])
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {"sentences": [], "exists": set()}

        # Decode all generated questions
        decoded_questions = tokenizer.batch_decode(generated_questions, skip_special_tokens=True)

        for i in range(generated_questions.size(0)):
            # Find the final log-reward at the point of termination
            termination_idx = (generated_questions[i] == self.termination_token_id).nonzero(as_tuple=True)[0]
            final_logreward = logrewards[i, termination_idx[0]].item() if len(termination_idx) > 0 else logrewards[i, -1].item()
            
            self.add(
                {
                    "logreward": final_logreward,
                    "str_prompt": str_prompt,
                    "str_sentence": decoded_questions[i],
                    "tensor_sentence": generated_questions[i],
                    "full_logrewards": logrewards[i, :],
                }
            )

    def sample(self, prompt_tokens: torch.Tensor, batch_size: int):
        """
        Samples a batch of trajectories for a given prompt from the buffer.
        
        Returns:
            A tuple of (action_sequences, log_rewards). Returns (None, None) if prompt not in buffer.
        """
        str_prompt = " ".join([str(t) for t in prompt_tokens.tolist()])
        if str_prompt not in self._buffer or not self._buffer[str_prompt]["sentences"]:
            return None, None

        prompt_buffer = self._buffer[str_prompt]["sentences"]
        
        # Uniformly sample indices from the buffer for this prompt
        indices = np.random.choice(len(prompt_buffer), size=batch_size, replace=True)
        
        sampled_items = [prompt_buffer[i] for i in indices]
        
        action_seqs = [item[2] for item in sampled_items] # item[2] is tensor_sentence
        log_rewards = [item[3] for item in sampled_items] # item[3] is full_logrewards
        
        # Pad sequences to the same length for batching
        padded_actions = pad_sequence(action_seqs, batch_first=True, padding_value=self.termination_token_id)
        padded_rewards = pad_sequence(log_rewards, batch_first=True, padding_value=0.0)

        return padded_actions, padded_rewards