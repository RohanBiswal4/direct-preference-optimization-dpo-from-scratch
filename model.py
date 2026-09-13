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
        'rejected_mask':rejected_mask,
        'indices':indices }
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

# Step 18 - dpo_train_step
import numpy as np
def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    # Execute one DPO gradient-descent update; return updated params + metrics
    loss,grad=dpo_loss_grad(params, batch, ref_logprobs_batch, beta)
    m={'loss':float(loss)}
    updated={} # the new parameters after updates 
    for key in params:
        updated[key]=params[key]-learning_rate*grad[key]
    return updated,m

# Step 19 - train_dpo
def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    # Sample batches, run DPO train steps, record per-step metrics.
    history=[]
    for step in range(num_steps):
        batch=sample_preference_batch(pairs, batch_size, rng=rng)
        ref_logprobs_batch={
            'chosen':ref_logprobs['chosen'][batch['indices']],
            'rejected':ref_logprobs['rejected'][batch['indices']]
        }
        params,metric=dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate)
        metric['step']=step 
        history.append(metric)
    return params,history

# Step 20 - length_normalized_logprob
def length_normalized_logprob(seq_logprob, mask):
    # Normalize sequence log-probabilities by their valid token counts.So that longer seq don't get penalized more
    valid_counts=np.sum(mask,axis=-1) # valid only if mask is 1
    return seq_logprob/valid_counts # Normalize

# Step 21 - ipo_loss
def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # Evaluate mean squared IPO loss on unscaled log-ratio margins
    # margin given beta=1 for unscaled
    margin=dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected,1)
    return np.mean((margin -0.5/beta)**2).item() # the squred ipo loss

# Step 22 - implicit_reward
def implicit_reward(policy_logprob, reference_logprob, beta):
    # return the vector of DPO implicit rewards for a batch of sequences
    reward=policy_reference_logratio(policy_logprob, reference_logprob)*beta 
    return reward # the KL divergence

# Step 23 - preference_accuracy
def preference_accuracy(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # fraction of pairs where chosen has higher implicit DPO reward
    # Chosen rewards
    chosen_reward=implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)
    # Rejected rewards
    rejected_reward=implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)
    # return higher chosen reward fraction
    return np.sum(np.where(chosen_reward>rejected_reward,1,0))/len(chosen_reward)

# Step 24 - kl_to_reference
def kl_to_reference(policy_logprob, reference_logprob):
    # Estimate the mean KL divergence of the policy from the reference...
    return np.mean(policy_reference_logratio(policy_logprob, reference_logprob)).item()

# Step 25 - reward_margin_stats
def reward_margin_stats(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # Summarize implicit-reward margins with mean, std, and frac positive.
    reward_margin=implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)-implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)
    return {
        'mean_margin': np.mean(reward_margin),
        'std_margin': np.std(reward_margin),
        'frac_positive':preference_accuracy(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta)
    }

# Step 26 - evaluate_dpo
def evaluate_dpo(params, pairs, ref_logprobs, beta):
    # Result dict of all the summary
    result = {}

    policy_chosen = []
    policy_rejected = []
    ref_chosen=[]
    ref_rejected=[]
    # Evaluate every preference pair separately
    for pair,ref in zip(pairs,ref_logprobs):

        chosen_lp = policy_sequence_logprob(
            params,
            pair['chosen_ids'][None, :],
            pair['chosen_mask'][None, :]
        )

        rejected_lp = policy_sequence_logprob(
            params,
            pair['rejected_ids'][None, :],
            pair['rejected_mask'][None, :]
        )

        policy_chosen.append(chosen_lp)
        policy_rejected.append(rejected_lp)
        ref_chosen.append(ref['chosen'])
        ref_rejected.append(ref['rejected'])

    # Convert policy log-probs to arrays
    policy_chosen = np.asarray(policy_chosen)
    policy_rejected = np.asarray(policy_rejected)

    # Reference log-probs are already stored as arrays
    ref_chosen = np.asarray(ref_chosen)
    ref_rejected = np.asarray(ref_rejected)

    # DPO loss
    result['dpo_loss'] = dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta
    )

    # Preference accuracy
    result['preference_accuracy'] = preference_accuracy(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta
    )

    # KL to reference
    kl_chosen = kl_to_reference(
        policy_chosen,
        ref_chosen
    )

    kl_rejected = kl_to_reference(
        policy_rejected,
        ref_rejected
    )

    result['kl_to_reference'] = 0.5 * (
        kl_chosen + kl_rejected
    )

    # Reward-margin statistics
    margin_stats = reward_margin_stats(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta
    )

    for key, value in margin_stats.items():
        result[key] = value

    return result

# Step 27 - run_dpo_pipeline
def run_dpo_pipeline(vocab_size, d_model, prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta, learning_rate, num_steps, batch_size, rng=None):
    # Wire the full DPO pipeline end-to-end from raw arrays to eval...
    if rng is None:
        rng=np.random.default_rng()
    params=init_policy_params(vocab_size, d_model, rng=rng)
    pairs=build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    ref_logprobs=freeze_reference_logprobs(params, pairs)
    params,hist=train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=rng)
    res=evaluate_dpo(params, pairs, ref_logprobs, beta) 
    return {'params': params, 'history': hist, 'eval_metrics': res}

