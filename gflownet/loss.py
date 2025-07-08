# gflownet/loss.py

import torch

def modified_subtb_loss(
    log_pf: torch.Tensor,
    log_r: torch.Tensor,
    log_pterm: torch.Tensor,
    generated_text: torch.Tensor,
    termination_token_id: int,
    prompt_len: int,
    subtb_lambda: float = 1.0,
):
    """
    Computes the Sub-Trajectory Balance (SubTB) loss for a batch of trajectories.

    This loss function is designed to enforce consistency in the GFlowNet framework.
    It checks that for any sub-trajectory, the flow entering it equals the flow leaving it.
    The loss is the mean squared error of this balance condition, aggregated over
    sub-trajectories of varying lengths.

    Args:
        log_pf (torch.Tensor): Log-probabilities of the forward policy P_F(a_t|s_t).
        log_r (torch.Tensor): Log-rewards for each intermediate state.
        log_pterm (torch.Tensor): Log-probabilities of terminating at each step.
        generated_text (torch.Tensor): The full generated trajectories.
        termination_token_id (int): The ID of the termination token.
        prompt_len (int): The length of the initial prompt.
        subtb_lambda (float): A discount factor for weighting errors from longer sub-trajectories.

    Returns:
        torch.Tensor: The final computed loss value for the batch.
    """
    # Ensure all inputs have the same trajectory length dimension
    gen_len = generated_text.shape[1] - prompt_len
    if not (log_pf.shape[1] == log_r.shape[1] == log_pterm.shape[1] == gen_len):
         # This can happen if max_len is 0
         return torch.tensor(0.0, device=log_pf.device, requires_grad=True)

    # For SubTB, at least one transition (2 states) is required.
    if log_pf.shape[1] <= 1:
        return torch.tensor(0.0, device=log_pf.device, requires_grad=True)

    # The SubTB balance condition for a single transition (s_t -> s_{t+1}):
    # log(F(s_t) * P_F(s_{t+1}|s_t)) = log(F(s_{t+1}) * P_B(s_t|s_{t+1}))
    # Using P_B = 1 and F(s) = R(s), this simplifies to:
    # log(R(s_t)) + log(P_F(s_{t+1}|s_t)) = log(R(s_{t+1}))
    # However, GFlowNets use a forward-looking perspective with termination probs:
    # F(s_t) = R(s_t) * P(terminate|s_t) + sum_{s'} F(s') * P(s'|s_t)
    # The SubTB loss is derived from this, comparing flow through sub-trajectories.
    # delta(t) = log(R_t) + log(P_F_{t,t+1}) + log(P_term_{t+1}) - log(R_{t+1}) - log(P_term_t)
    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Create a mask to ignore any calculations for steps after a trajectory has terminated
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    for subtraj_len in range(1, gen_len):
        # The squared error of the balance condition for sub-trajectories of this length
        subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
        # Apply the mask
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        
        # Weight the loss term by lambda and the number of valid (unmasked) elements
        weight = subtb_lambda ** (subtraj_len - 1)
        batch_loss += weight * subtb_term.sum()
        total_lambda += weight * (~mask[:, subtraj_len - 1 :]).sum()
        
    # Normalize the loss by the total weight
    if total_lambda > 0:
        batch_loss /= total_lambda
    else:
        return torch.tensor(0.0, device=log_pf.device, requires_grad=True)

    return batch_loss