# Adapted from https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/eval_model.py

from datasets import load_dataset
import torch, time
import numpy as np
from tqdm import tqdm

import gc

__all__ = ["eval_wikitext2"]

def cleanup():
    # torch.cuda.empty_cache()
    gc.collect()


def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True, limit=100):
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    encodings["input_ids"] = encodings["input_ids"].to(model.device)

    lls, t = [], []
    cnt = 0
    for i in tqdm(range(0, encodings["input_ids"].size(1), stride), disable=not verbose):
        if cnt > limit:
            break
        cnt += 1
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings["input_ids"].size(1))
        trg_len = end_loc - i
        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context

        t1 = time.time()
        with torch.no_grad():
            log_likelihood = model(input_ids, labels=target_ids).loss * trg_len
        # torch.cuda.synchronize()
        log_likelihood = log_likelihood.cpu()
        t2 = time.time()
        t.append((t2 - t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    ppl = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    if verbose:
        print("perplexity", ppl)
        print("time", str(pred_time) + "  sec")

    del encodings
    cleanup()

    result = {"perplexity": ppl, "prediction_time": pred_time}
    print(result)
    return result
