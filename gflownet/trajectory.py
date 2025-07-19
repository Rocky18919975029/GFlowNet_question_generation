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
    # --- NEW: Add current_step parameter ---
    current_step: int = None,
):
    device = encoded_prompt.device
    batch_size = encoded_prompt.size(0)
    active_seqs = torch.ones(batch_size, dtype=torch.bool, device=device)
    state = encoded_prompt.clone()
    log_pf = []
    log_pterm = []
    token_ids_for_cache = state
    past_key_values = None
    
    for i in range(max_len + 1):
        output = model(input_ids=token_ids_for_cache, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        if action_seq is None:
            # Sampling Mode...
            with torch.no_grad():
                modified_logits = logits.clone()
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
                if i < min_len:
                    modified_logits[:, termination_token_id] = -torch.inf
                elif i >= max_len:
                    mask = torch.ones(modified_logits.shape[1], dtype=torch.bool, device=modified_logits.device)
                    mask[termination_token_id] = False
                    modified_logits[:, mask] = -torch.inf
                probs = (modified_logits / temperature).softmax(dim=-1)
                token_ids = torch.multinomial(probs, num_samples=1)
        else:
            # Replay Mode...
            if i >= action_seq.size(-1):
                token_ids = torch.full_like(action_seq[:, 0], termination_token_id).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)

        token_ids = torch.where(active_seqs.unsqueeze(-1), token_ids, termination_token_id)
        logprob = logits.log_softmax(dim=-1)
        log_pterm.append(torch.where(active_seqs, logprob[:, termination_token_id], 0.0))
        log_pf.append(torch.where(active_seqs, logprob.gather(-1, token_ids).squeeze(-1), 0.0))
        active_seqs = active_seqs & (token_ids.squeeze(-1) != termination_token_id)
        state = torch.cat([state, token_ids], dim=-1)
        token_ids_for_cache = token_ids
        if torch.all(~active_seqs):
            break

    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    
    if skip_rewards:
        reward_tuple = (None, None, None, None)
    else:
        # --- CHANGE: Pass current_step to the reward function call ---
        reward_tuple = reward_fn(state, current_step=current_step)

    return state, log_pf, log_pterm, reward_tuple, None