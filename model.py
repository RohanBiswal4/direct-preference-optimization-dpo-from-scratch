"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # convert logits into numerically stable log-probabilities along axis
    max_vals = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp

# Step 2 - softmax
def softmax(logits, axis=-1):
    # Convert an array of logits into a probability distribution along a given axis
    max_vals = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_vals
    sum_exp = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    return np.exp(shifted)/sum_exp

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # Extract the log-probability of each observed token from a full vocab log-prob tensor...
    B,T=token_ids.shape
    return log_probs[np.arange(B)[:, None], np.arange(T)[None, :], token_ids]

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    new_probs=np.where(mask,token_logprobs,0) # sum only at the positions where masks are valid
    return np.sum(new_probs,axis=-1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    # Initialize the policy language-model parameters with small random values
    if rng is None:
        rng=np.random.default_rng()
    D={} 
    # random normal based initialization
    D['embed']=rng.normal(loc=0.0, scale=0.02, size=(vocab_size, d_model))
    D['W_out']=rng.normal(loc=0.0, scale=0.02, size=(d_model,vocab_size))
    D['b_out']=np.zeros(vocab_size)
    return D

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    # Compute next-token logits for every position from policy params and token ids.
    embedding_matrix=params['embed']
    return embedding_matrix[token_ids]@ params['W_out'] + params['b_out']

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    # Compute the total masked sequence log-probability under the current policy...
    logits=policy_token_logits(params, token_ids)
    log_probs=log_softmax(logits, axis=-1)
    token_logprobs=gather_token_logprobs(log_probs, token_ids)
    return masked_sequence_logprob(token_logprobs, mask)

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    # Compute gradients of the summed sequence log-probability w.r.t. params
    B,T=token_ids.shape 
    V=params['embed'].shape[0]
    # Forward pass
    embed=params['embed']
    hidden = embed[token_ids]
    logits = policy_token_logits(params, token_ids)
    probs = softmax(logits)
    d_logits=-probs
    d_logits[np.arange(B)[:,None],np.arange(T)[None,:],token_ids]+=1
    d_logits=mask[:,:,None]*d_logits
    dhidden = d_logits @ params['W_out'].T
    grad_embed = np.zeros_like(params['embed'])
    np.add.at(grad_embed, token_ids, dhidden)
    dW=hidden.transpose(0,2,1)@ d_logits
    return {
        'embed':grad_embed ,
        'W_out':np.sum(dW,axis=0),
        'b_out': np.sum(d_logits,axis=(0,1))
    }

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    # Compute the mean Bradley-Terry pairwise preference loss...
    reward_margin=(reward_chosen-reward_rejected)
    probs=1/(np.exp(-reward_margin)+1) # sigmoid 
    log_loss=np.log(probs)
    return -np.mean(log_loss).item() # return the average log loss

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    # Fraction of pairs where chosen reward is strictly higher than rejected.
    return np.mean(reward_chosen>reward_rejected)

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # Package raw arrays into a list of preference-pair dictionaries
    pairs=[]
    N= len(prompts)
    for i in range(N):
        D={}
        D['prompt']=prompts[i]
        D['chosen_ids']=chosen_ids[i]
        D['rejected_ids']=rejected_ids[i]
        D['chosen_mask']=chosen_mask[i]
        D['rejected_mask']=rejected_mask[i]
        pairs.append(D)
    return pairs

# Step 12 - sample_preference_batch
def sample_preference_batch(pairs, batch_size, rng=None):
    # Sample a mini-batch of preference pairs for one training step.
    if rng is None:
        rng=np.random.default_rng()
    N=len(pairs)
    replacement = batch_size > N # if batch is smaller then dont allow replacement
    indices = rng.choice(N,size=batch_size,replace=replacement) # sample indices
    # stack into batchwise tensors
    chosen_ids=np.stack([pairs[i]['chosen_ids'] for i in indices],axis=0)
    rejected_ids=np.stack([pairs[i]['rejected_ids'] for i in indices],axis=0)
    chosen_mask=np.stack([pairs[i]['chosen_mask'] for i in indices],axis=0)
    rejected_mask=np.stack([pairs[i]['rejected_mask'] for i in indices],axis=0)
    D= {
        'chosen_ids':chosen_ids,
        'rejected_ids':rejected_ids,
        'chosen_mask':chosen_mask,
        'rejected_mask':rejected_mask }
    if "prompt" in pairs[0]:
        D['prompt']=np.array([pairs[i]['prompt'] for i in indices])
    return D

# Step 13 - freeze_reference_logprobs
def freeze_reference_logprobs(ref_params, pairs):
    # Precompute and freeze reference-model sequence log-probabilities for every chosen and rejected response...
    L=[]
    if len(pairs)>0:
        for i in range(len(pairs)):
            out={}
            out['chosen']=policy_sequence_logprob(ref_params, pairs[i]['chosen_ids'][None,:], pairs[i]['chosen_mask'][None,:])[0]
            out['rejected']=policy_sequence_logprob(ref_params, pairs[i]['rejected_ids'][None,:], pairs[i]['rejected_mask'][None,:])[0]
            L.append(out)
    return L

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    # Computes the per-sequence log-ratio log pi_theta(y) - log pi_ref(y)
    return (policy_logprob-reference_logprob)

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    #Compute the scaled DPO pair margin for a batch of preference pairs
    chosen_log_ratio = policy_logprob_chosen - ref_logprob_chosen # policy/ref log-ratio for chosen
    rejected_log_ratio = policy_logprob_rejected - ref_logprob_rejected # policy/ref log-ratio for rejected
    margins = beta * (chosen_log_ratio - rejected_log_ratio)
    return np.asarray(margins).reshape(-1) # of shape (B,) 1D array

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: return the mean logistic loss on the DPO pair margins as a scalar float
    margins = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta)
    losses = np.logaddexp(0.0, -margins) # stable softplus equivalent to softmax
    return np.mean(losses).item()

# Step 17 - dpo_loss_grad
def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    # Evaluate DPO loss and return parameter gradients for the policy
    B = len(batch['chosen_ids']) # The batch size
    policy_logprob_chosen = policy_sequence_logprob(params,batch['chosen_ids'],batch['chosen_mask'] )
    policy_logprob_rejected = policy_sequence_logprob(params,batch['rejected_ids'],batch['rejected_mask'])
    ref_logprob_chosen = np.asarray( ref_logprobs_batch['chosen'])
    ref_logprob_rejected = np.asarray( ref_logprobs_batch['rejected'])
    # DPO margins
    margins =dpo_pair_margin(policy_logprob_chosen, 
                            policy_logprob_rejected, 
                            ref_logprob_chosen, 
                            ref_logprob_rejected,
                            beta)
    # Mean DPO loss
    loss = dpo_loss(policy_logprob_chosen, 
                    policy_logprob_rejected, 
                    ref_logprob_chosen, 
                    ref_logprob_rejected, 
                    beta)

    # dL/dm
    sigmoid = 1.0 / (1.0 + np.exp(-margins))
    dL_dm = sigmoid - 1.0

    # Initialize parameter gradients
    grads = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    for b in range(B):
        # Adds batch dimension because the gradient function
        # expects (B, T)
        chosen_ids = batch['chosen_ids'][b:b+1]
        chosen_mask = batch['chosen_mask'][b:b+1]
        rejected_ids = batch['rejected_ids'][b:b+1]
        rejected_mask = batch['rejected_mask'][b:b+1]

        # Gradients of sequence log-probabilities
        grad_chosen = sequence_logprob_grad(
            params,
            chosen_ids,
            chosen_mask
        )
        grad_rejected = sequence_logprob_grad(
            params,
            rejected_ids,
            rejected_mask
        )

        # Chain rule:
        #
        # dL/dtheta =
        # beta * (sigmoid(m)-1) / B
        # * (d log pi_chosen/dtheta
        #    - d log pi_rejected/dtheta)

        scale = beta * dL_dm[b] / B
        # for each key attach the full gradients
        for key in params:
            grads[key] += scale * (
                grad_chosen[key] - grad_rejected[key]
            )

    return loss, grads

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

