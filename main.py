import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from time import time
import random
from typing import Tuple

def relu_normalize(p, q):
    tmp_dist = torch.relu(p-q)
    return tmp_dist / tmp_dist.sum(dim=-1, keepdim=True)

def truncate_kv_cache(kv_cache: Tuple, truncation_size: int):
    kv_cache = list(kv_cache)
    for i in range(len(kv_cache)):
        kv_cache[i] = list(kv_cache[i])
        kv_cache[i][0] = kv_cache[i][0][:, :, :-truncation_size, :]
        kv_cache[i][1] = kv_cache[i][1][:, :, :-truncation_size, :]
    return kv_cache

def logits_adapter(logits: torch.Tensor, temperature: float, top_p: float):
    flag = False
    if logits.ndim==3:
        bsz = logits.shape[0]
        l = logits.shape[1]
        logits = logits.view(-1, logits.shape[-1])
        flag = True
    prob = torch.softmax(logits / temperature, dim=-1)
    sorted_prob, sorted_prob_idx = torch.sort(prob, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = (cumsum - sorted_prob) > top_p
    sorted_prob[mask] = 0.0
    sorted_prob.div_(sorted_prob.sum(dim=-1, keepdim=True))
    _, gather_pos = torch.sort(sorted_prob_idx, descending=False, dim=-1)
    final_prob = torch.gather(sorted_prob, -1, gather_pos)
    if flag: final_prob = final_prob.view(bsz, l, -1)
    return final_prob


@torch.inference_mode()
def auto_regressive_sampling(input_prompt: str, model, tokenizer, gen_kwargs: dict):
    """
    Standard auto-regressive sampling
    """
    max_new_tokens = gen_kwargs.get('max_new_tokens', 20)
    top_p = gen_kwargs.get('top_p', 1.0)
    temperature = gen_kwargs.get('temperature', 1.0)

    inputs = tokenizer([input_prompt], return_tensors='pt').to(model.device)
    outputs_prefilling = model(input_ids=inputs['input_ids'], use_cache=True)
    prefix_token_lst = inputs['input_ids'][0].cpu().numpy().tolist()
    past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
    logits_prev_step = logits[:, -1, :]
    prob_prev_step = torch.softmax(logits_prev_step / temperature, dim=-1)

    n = 0
    output_ids = []
    s_time = time()
    while n < max_new_tokens:
        sorted_prob, sorted_prob_idx = torch.sort(prob_prev_step, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_prob, dim=-1)
        mask = (cumsum - sorted_prob) > top_p
        sorted_prob[mask] = 0.0
        sorted_prob.div_(sorted_prob.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(sorted_prob, num_samples=1)
        next_token = torch.gather(sorted_prob_idx, -1, next_token) # (bsz, 1)
        output_ids.append(next_token[0].item())
        n += 1
        # decoded_output = tokenizer.decode(prefix_token_lst + output_ids, skip_special_tokens=True)
        # sys.stdout.write('\r' + repr(decoded_output))
        # sys.stdout.flush()
        if output_ids[-1] == tokenizer.eos_token_id: break
        outputs = model(input_ids=next_token, 
                        past_key_values=past_key_values,
                        attention_mask=torch.ones(next_token.shape[0], 1+past_key_values[0][0].shape[2], dtype=torch.long, device=next_token.device),
                        position_ids=torch.LongTensor([past_key_values[0][0].shape[2]]).to(model.device).view(-1, 1),
                        use_cache=True)
        past_key_values = outputs.past_key_values
        logits_prev_step = outputs.logits[:, -1, :]
        prob_prev_step = torch.softmax(logits_prev_step / temperature, dim=-1)
    e_time = time()
    print(f"{n} tokens generated, Speed: {n/(e_time-s_time):.3f} tokens/s")
    decoded_output = tokenizer.decode(prefix_token_lst + output_ids, skip_special_tokens=True)
    print(f'Auto-regressive output: {decoded_output}')
    return decoded_output


@torch.inference_mode()
def speculative_sampling(input_prompt: str, tgt_model, draft_model, tokenizer, k: int, gen_kwargs: dict):
    """
    Speculative sampling
    """
    max_new_tokens = gen_kwargs.get('max_new_tokens', 20)
    top_p = gen_kwargs.get('top_p', 1.0)
    temperature = gen_kwargs.get('temperature', 1.0)

    # prefill for draft model
    inputs = tokenizer([input_prompt], return_tensors='pt').to(model.device)
    outputs_prefilling = draft_model(input_ids=inputs['input_ids'], use_cache=True)
    prefix_token_lst = inputs['input_ids'][0].cpu().numpy().tolist()
    draft_past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
    logits_prev_step = logits[:, -1, :]
    prob_prev_step = logits_adapter(logits_prev_step, temperature=temperature, top_p=top_p)

    # prefill for target model
    tgt_outputs_prefilling = tgt_model(input_ids=inputs['input_ids'][:, :-1], use_cache=True)
    tgt_past_key_values = tgt_outputs_prefilling.past_key_values
    tgt_prev_token_id = inputs['input_ids'][0, -1].cpu().item()

    output_ids = []

    n = 0
    acc_tokens = 0
    draft_times = 0
    s_time = time()
    while n < max_new_tokens:
        # ================= drafting stage ========================== #
        draft_tokens = []
        draft_tokens_prob = []
        draft_prob = [prob_prev_step]
        for _ in range(k):
            next_draft_token = torch.multinomial(prob_prev_step, num_samples=1)
            next_draft_token_prob = torch.gather(prob_prev_step, -1, next_draft_token) # (bsz, 1)
            draft_tokens.append(next_draft_token[0, 0].cpu().item())
            draft_tokens_prob.append(next_draft_token_prob[0, 0].cpu().item())
            draft_outputs = draft_model(input_ids=next_draft_token, 
                past_key_values=draft_past_key_values,
                attention_mask=torch.ones(next_draft_token.shape[0], 1+draft_past_key_values[0][0].shape[2], dtype=torch.long, device=next_draft_token.device),
                position_ids=torch.LongTensor([draft_past_key_values[0][0].shape[2]]).to(model.device).view(-1, 1),
                use_cache=True)
            draft_past_key_values = draft_outputs.past_key_values
            draft_logits = draft_outputs.logits[:, -1, :]
            prob_prev_step = logits_adapter(draft_logits, temperature=temperature, top_p=top_p)
            draft_prob.append(prob_prev_step)
        draft_times += 1

        # ================= verification stage ========================== #
        tgt_input_ids = torch.tensor([[tgt_prev_token_id, *draft_tokens]], device=tgt_model.device)
        tgt_attention_mask = torch.ones(1, tgt_past_key_values[0][0].shape[2]+k+1, dtype=torch.long, device=next_draft_token.device)
        tgt_position_ids = torch.arange(tgt_past_key_values[0][0].shape[2], tgt_past_key_values[0][0].shape[2]+k+1).unsqueeze(0).to(next_draft_token.device)
        tgt_outputs = tgt_model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask, position_ids=tgt_position_ids, past_key_values=tgt_past_key_values, use_cache=True)
        tgt_past_key_values = tgt_outputs.past_key_values
        tgt_prob = logits_adapter(tgt_outputs.logits, temperature=temperature, top_p=top_p)
        # tgt_prob = torch.softmax(tgt_outputs.logits / temperature, dim=-1)
        all_accept = True
        for i in range(k):
            tgt_token_prob = tgt_prob[0, i, draft_tokens[i]].cpu().item()
            if random.random() <= (tgt_token_prob / draft_tokens_prob[i]): pass
            else:
                all_accept = False
                acc_tokens += i
                n += (i+1)
                # reject current token
                modified_dist = relu_normalize(tgt_prob[0, i], draft_prob[i][0]) # (vocab_size, )
                resampled_token = modified_dist.multinomial(num_samples=1).unsqueeze(0)
                output_ids.extend([tgt_prev_token_id, *draft_tokens[:i]])
                tgt_prev_token_id = resampled_token[0,0].cpu().item()
                # modify draft kv cache
                tgt_past_key_values = truncate_kv_cache(tgt_past_key_values, truncation_size=k-i)
                draft_past_key_values = truncate_kv_cache(draft_past_key_values, truncation_size=k-i)
                draft_outputs = draft_model(input_ids=resampled_token, 
                    attention_mask=torch.ones(next_draft_token.shape[0], 1+draft_past_key_values[0][0].shape[2], dtype=torch.long, device=next_draft_token.device),
                    position_ids=torch.LongTensor([draft_past_key_values[0][0].shape[2]]).to(model.device).view(-1, 1),
                    past_key_values=draft_past_key_values,
                    use_cache=True)
                draft_past_key_values = draft_outputs.past_key_values
                draft_logits = draft_outputs.logits[:, -1, :]
                prob_prev_step = logits_adapter(draft_logits, temperature=temperature, top_p=top_p)
                break
        if all_accept:
            acc_tokens += k
            output_ids.extend([tgt_prev_token_id, *draft_tokens])
            tgt_next_token = tgt_prob[0, -1].multinomial(num_samples=1).unsqueeze(0)
            draft_outputs = draft_model(input_ids=tgt_next_token, 
                past_key_values=draft_past_key_values,
                attention_mask=torch.ones(next_draft_token.shape[0], 1+draft_past_key_values[0][0].shape[2], dtype=torch.long, device=next_draft_token.device),
                position_ids=torch.LongTensor([draft_past_key_values[0][0].shape[2]]).to(model.device).view(-1, 1),
                use_cache=True)
            draft_past_key_values = draft_outputs.past_key_values
            draft_logits = draft_outputs.logits[:, -1, :]
            prob_prev_step = logits_adapter(draft_logits, temperature=temperature, top_p=top_p)
            tgt_prev_token_id = tgt_next_token[0, 0].cpu().item()
            n += (k+1)
        # decoded_output = tokenizer.decode(prefix_token_lst[:-1] + output_ids, skip_special_tokens=True)
        # sys.stdout.write('\r' + repr(decoded_output))
        # sys.stdout.flush()
    e_time = time()
    print(f"{n} tokens generated, Speed: {n/(e_time-s_time):.3f} tokens/s")
    decoded_output = tokenizer.decode(prefix_token_lst + output_ids, skip_special_tokens=True)
    print(f'Speculative output: {decoded_output}')
    print(f"Speculative acceptance rate: {acc_tokens / (draft_times*k)*100:.3f}%")
    print(f"Draft steps: {draft_times*4}")
    return decoded_output


if __name__ == '__main__':
    # prompt and gen kwargs
    input_prompt = "Below is a piece of Python code to efficienctly compute the n-th Fibonacci number using cache(a lookup table):\n"
    print(f"Input prompt: {input_prompt}")
    gen_kwargs = dict(
        temperature=1e-8,
        max_new_tokens=80
    )

    # model path
    path = '/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--codellama--CodeLlama-34b-Python-hf/snapshots/bf21d9502411be2f59900007374ae9d0c37f0d54/'

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(path)
    print('HF generation: ', tokenizer.batch_decode(model.generate(input_ids=tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device), do_sample=True, **gen_kwargs), skip_special_tokens=True)[0])
    o_as = auto_regressive_sampling(input_prompt, model, tokenizer, gen_kwargs=gen_kwargs)

    draft_path = '/cpfs01/user/rensiyu/ssd/tinyllama-1.1B'
    draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=torch.float16, device_map='auto')
    o_ss = speculative_sampling(input_prompt, model, draft_model, tokenizer, k=4, gen_kwargs=gen_kwargs)