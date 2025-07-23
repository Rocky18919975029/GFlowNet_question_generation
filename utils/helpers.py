# utils/helpers.py

import torch

def lora_to_base(model):
    """
    Disables the LoRA adapter layers, allowing inference with the original base model weights.
    Sets the model to evaluation mode.
    """
    try:
        model.base_model.disable_adapter_layers()
    except AttributeError:
        # Handle cases where the model might not be a PeftModel, though it should be.
        print("Warning: Model does not have 'base_model.disable_adapter_layers'. May not be a PEFT model.")
    model.eval()


def base_to_lora(model):
    """
    Enables the LoRA adapter layers for fine-tuning.
    Sets the model to training mode.
    """
    try:
        model.base_model.enable_adapter_layers()
    except AttributeError:
        print("Warning: Model does not have 'base_model.enable_adapter_layers'. May not be a PEFT model.")
    model.train()


def get_termination_vals(
    generated_text: torch.Tensor,
    log_pf: torch.Tensor,
    log_pterm: torch.Tensor,
    log_r: torch.Tensor,
    log_r_unpenalized: torch.Tensor,
    termination_token_id: int,
    prompt_len: int,
):
    """
    Extracts values from the completed trajectories at their termination step.

    This function finds where each sequence in a batch terminated and returns the
    log-probabilities and log-rewards at that specific point. It's useful for logging.

    Args:
        generated_text (torch.Tensor): The full generated trajectories.
        log_pf (torch.Tensor): Log-probabilities of the forward policy.
        log_pterm (torch.Tensor): Log-probabilities of terminating.
        log_r (torch.Tensor): Log-rewards.
        log_r_unpenalized (torch.Tensor): Unpenalized log-rewards.
        termination_token_id (int): The ID of the termination token.
        prompt_len (int): The length of the initial prompt.

    Returns:
        A tuple of (log_pfs, final_log_r, final_log_r_unpenalized, gen_len).
    """
    batch_size = generated_text.size(0)
    batch_idx = torch.arange(batch_size, device=generated_text.device)
    
    # Find the index of the first termination token in the generated part of the text
    generated_part = generated_text[:, prompt_len:]
    # Add a termination token at the end in case the sequence maxed out its length
    generated_part_with_eos = torch.cat(
        [generated_part, torch.full((batch_size, 1), termination_token_id, device=generated_part.device, dtype=torch.long)],
        dim=1
    )

    gen_len = (generated_part_with_eos == termination_token_id).int().argmax(dim=-1)

    # --- Extract final log P_F(s) ---
    # log P_F(s) = sum(log P_F(a_t|s_t)) + log P_term(s_T)
    if log_pf is None or log_pterm is None:
        log_pfs = None
    else:
        # Cumulative sum of forward log-probs up to each step
        log_pf_cumsum = log_pf.cumsum(dim=-1)
        # Prepend a zero for the initial state
        log_pf_cumsum = torch.cat([torch.zeros_like(log_pf_cumsum[:, :1]), log_pf_cumsum], dim=1)
        
        # Get the sum of log_pf up to the step *before* termination
        sum_log_pf = log_pf_cumsum[batch_idx, gen_len]
        # Get the log_pterm at the termination step
        term_log_p = log_pterm[batch_idx, gen_len]
        log_pfs = sum_log_pf + term_log_p

    # --- Extract final log R(s) ---
    if log_r is not None:
        final_log_r = log_r[batch_idx, gen_len]
    else:
        final_log_r = None
        
    if log_r_unpenalized is not None:
        final_log_r_unpenalized = log_r_unpenalized[batch_idx, gen_len]
    else:
        final_log_r_unpenalized = None
        
    return log_pfs, final_log_r, final_log_r_unpenalized, gen_len