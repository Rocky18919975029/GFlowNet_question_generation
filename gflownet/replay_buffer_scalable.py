# gflownet/replay_buffer_scalable.py

import redis
import torch
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class RedisReplayBuffer:
    """
    A DDP-safe replay buffer using a Redis backend.

    It stores trajectories as serialized PyTorch tensors in a Redis sorted set,
    with the log-reward as the score. This allows for efficient retrieval of
    high-reward samples.
    """
    def __init__(self, host: str, port: int, buffer_size: int, termination_token_id: int):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        try:
            self.client = redis.Redis(host=self.host, port=self.port, db=0)
            # Ping the server to check the connection
            self.client.ping()
            print(f"--- Successfully connected to Redis at {host}:{port} ---")
        except redis.exceptions.ConnectionError as e:
            print(f"--- FATAL: Could not connect to Redis at {host}:{port}. Please ensure Redis server is running. ---")
            raise e

    def reset(self):
        """Clears all buffered trajectories from Redis by deleting the keys."""
        print("--- Clearing all keys from Redis replay buffer... ---")
        count = 0
        for key in self.client.scan_iter("gfn_buffer:*"):
            self.client.delete(key)
            count += 1
        print(f"--- Cleared {count} keys. ---")

    def _get_key(self, prompt_tokens: torch.Tensor) -> str:
        """Creates a consistent key from a prompt tensor."""
        prompt_hash = hash(str(prompt_tokens.cpu().tolist()))
        return f"gfn_buffer:{prompt_hash}"

    def add_batch(self, prompt_tokens: torch.Tensor, generated_questions: torch.Tensor, logrewards: torch.Tensor, tokenizer):
        """
        Adds a batch of generated items to the Redis buffer.
        'logrewards' should be the final reward for each trajectory (shape [batch_size] or [batch_size, 1]).
        """
        if generated_questions.shape[0] == 0:
            return

        redis_key = self._get_key(prompt_tokens)

        with self.client.pipeline() as pipe:
            for i in range(generated_questions.size(0)):
                question_tensor = generated_questions[i]
                final_logreward = logrewards[i].item()
                serialized_tensor = pickle.dumps(question_tensor.cpu())
                
                pipe.zadd(redis_key, {serialized_tensor: final_logreward})

            pipe.zremrangebyrank(redis_key, 0, -self.buffer_size - 1)
            pipe.execute()

    def sample(self, prompt_tokens: torch.Tensor, batch_size: int):
        """Samples a batch of high-reward trajectories from the buffer."""
        redis_key = self._get_key(prompt_tokens)

        if not self.client.exists(redis_key):
            return None, None

        try:
            # ZREVRANGE gets the items with the HIGHEST scores (rewards)
            sampled_items = self.client.zrevrange(redis_key, 0, self.buffer_size - 1, withscores=True)
        except redis.exceptions.ResponseError:
            return None, None

        if not sampled_items:
            return None, None
        
        # We now have a list of top-reward items, let's randomly sample from it
        num_available = len(sampled_items)

        # --- FIX: Manually set a seed for this operation to ensure all DDP ranks make the same choice ---
        # This seed is deterministic based on the prompt, ensuring all ranks sample the same indices.
        np.random.seed(abs(hash(redis_key)) % (2**32 - 1))
        
        indices_to_sample = np.random.choice(num_available, size=batch_size, replace=True)

        action_seqs = []
        log_rewards = []
        for index in indices_to_sample:
            serialized_tensor, score = sampled_items[index]
            try:
                action_seqs.append(pickle.loads(serialized_tensor))
                log_rewards.append(torch.tensor(score))
            except pickle.UnpicklingError:
                continue
        
        if not action_seqs:
            return None, None

        padded_actions = pad_sequence(action_seqs, batch_first=True, padding_value=self.termination_token_id)
        padded_rewards = torch.stack(log_rewards)

        return padded_actions, padded_rewards