# gflownet/trajectory.py 

import torch

def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    reward_fn,
    termination_token_id: int,
    max_len: int,
    min_len: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    action_seq: torch.Tensor = None,
    skip_rewards: bool = False,
):
    """
    Generates sequences token-by-token and returns all necessary components for GFlowNet loss calculation.

    This function is the core generative loop of the GFlowNet. It can operate in two modes:
    1. Sampling mode (action_seq is None): Samples new trajectories from the model's policy.
    2. Replay mode (action_seq is not None): Re-evaluates existing trajectories from a buffer.
    
    Args:
        model: The model to generate with (the fine-tuned GFlowNet sampler).
        encoded_prompt (torch.Tensor): The input prompt (e.g., tokenized [task_prompt] + [z]).
        reward_fn: A callable function that takes the full trajectories and returns log-rewards.
        termination_token_id (int): The ID of the token that signifies the end of a sequence.
        max_len (int): The maximum number of tokens to generate.
        min_len (int): The minimum number of tokens before termination is allowed.
        temperature (float): Temperature for sampling.
        top_k (int): If > 0, sample from the top-k most likely tokens.
        top_p (float): If < 1.0, sample from the smallest set of tokens whose cumulative probability exceeds top_p.
        action_seq (torch.Tensor, optional): A pre-defined sequence of actions to take, used for replay.
        skip_rewards (bool, optional): If True, do not calculate rewards (useful for simple generation).

    Returns:
        A tuple containing:
        - state (torch.Tensor): The completed trajectories (prompt + generated tokens).
        - log_pf (torch.Tensor): Log-probabilities of the forward policy P_F(a_t|s_t).
        - log_pterm (torch.Tensor): Log-probabilities of terminating at each step.
        - log_r (torch.Tensor): Log-rewards for each intermediate state.
        - log_r_unpenalized (torch.Tensor): Log-rewards without the min_len penalty.
    """
    device = encoded_prompt.device
    batch_size = encoded_prompt.size(0)

    # Keep track of which sequences in the batch are still active
    active_seqs = torch.ones(batch_size, dtype=torch.bool, device=device)
    state = encoded_prompt.clone()
    
    log_pf = []
    log_pterm = []
    
    # Use key-value caching for efficient generation
    token_ids_for_cache = state
    past_key_values = None
    
    for i in range(max_len + 1):
        output = model(input_ids=token_ids_for_cache, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :] # Get logits for the next token

        if action_seq is None:
            # --- Sampling Mode ---
            with torch.no_grad():
                modified_logits = logits.clone()

                # Apply top-k and top-p filtering
                if top_k > 0:
                    v, _ = torch.topk(modified_logits, top_k)
                    modified_logits[modified_logits < v[:, -1].unsqueeze(-1)] = -torch.inf

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(modified_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    modified_logits[indices_to_remove] = -torch.inf

                # Enforce min/max length constraints by penalizing/forcing the termination token
                if i < min_len:
                    modified_logits[:, termination_token_id] = -torch.inf
                elif i >= max_len:
                    # --- FIX FOR INDEXERROR ---
                    # if we've reached the maximum length, force termination by setting all
                    # non-termination logits to -inf.
                    # First, create a boolean mask for all vocabulary items.
                    mask = torch.ones(modified_logits.shape[1], dtype=torch.bool, device=modified_logits.device)
                    # Set the termination token's position in the mask to False.
                    mask[termination_token_id] = False
                    # Use this 1D tensor mask to index the 2D logits tensor.
                    modified_logits[:, mask] = -torch.inf
                    # --- END OF FIX ---
                
                # Sample the next token
                probs = (modified_logits / temperature).softmax(dim=-1)
                token_ids = torch.multinomial(probs, num_samples=1)
        else:
            # --- Replay Mode ---
            if i >= action_seq.size(-1):
                # If the replay sequence is shorter than max_len, force termination
                token_ids = torch.full_like(action_seq[:, 0], termination_token_id).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)

        # Ensure that sequences that have already terminated just keep generating termination tokens
        # (their log-probs will be masked to zero later)
        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, termination_token_id)
        
        # Calculate log-probabilities from the original, unmodified logits
        logprob = logits.log_softmax(dim=-1)
        
        # Record the log-probability of terminating at this step
        log_pterm.append(torch.where(active_seqs, logprob[:, termination_token_id], 0.0))
        
        # Record the log-probability of the action taken (the forward policy log_pf)
        log_pf.append(torch.where(active_seqs, logprob.gather(-1, token_ids).squeeze(-1), 0.0))

        # Update the active sequences mask
        active_seqs = active_seqs & (token_ids.squeeze(-1) != termination_token_id)
        
        # Append the new token to the state
        state = torch.cat([state, token_ids], dim=-1)
        token_ids_for_cache = token_ids # For the next loop, only pass the new token

        # Early exit if all sequences have terminated
        if torch.all(~active_seqs):
            break

    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # The reward function needs the full trajectories to do its job
        log_r, log_r_unpenalized = reward_fn(state)
        
    return state, log_pf, log_pterm, log_r, log_r_unpenalized