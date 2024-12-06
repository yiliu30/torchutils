from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# from torch.utils._python_dispatch import TorchDispatchMode

from torch.overrides import TorchFunctionMode
import collections


class Counter:
    shape_counter = collections.defaultdict(dict)
    _stage = 0


class LoggingMode(TorchFunctionMode):
    # def __torch_dispatch__(self, func, types, args, kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     res = func(*args, **kwargs)
    #     print(f"f{func.__module__}:{func.__name__}")
    #     return res

    def __torch_function__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        res = func(*args, **kwargs)
        # print(f"f{func.__module__}:{func.__name__}")
        if func == torch.nn.functional.linear:
            Counter.shape_counter[Counter._stage][(args[0].shape, args[1].shape)] = (
                Counter.shape_counter[Counter._stage].get((args[0].shape, args[1].shape), 0) + 1
            )
            print(f"  - input: {args[0].shape}, weight: {args[1].shape}")
        return res


def inspect_model(args):
    model_name_or_path = args.model_name_or_path
    with torch.no_grad():
        with LoggingMode():
            torch.set_default_device("cuda:0")
            # model_name_or_path = "/models/Llama-2-7b-chat-hf/"
            # model_name_or_path = "/models/Llama-2-13b-hf/"

            # To use a different branch, change revision
            # For example: revision="gptq-4bit-64g-actorder_True"
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )

            def increase_stage_for_each_forward_pre_hook(module, input):
                Counter._stage += 1

            model.register_forward_pre_hook(increase_stage_for_each_forward_pre_hook)

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

            prompt = "Tell me about AI"

            print("\n\n*** Generate:")

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            output = model.generate(
                inputs=input_ids,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                max_new_tokens=5,
            )
            print(f"output: {output}")
            decode = tokenizer.decode(output[0])
            print(f"Generated text: {decode}")

            # # Inference can also be done using transformers' pipeline

            # print("*** Pipeline:")
            # pipe = pipeline(
            #     "text-generation",
            #     model=model,
            #     tokenizer=tokenizer,
            #     max_new_tokens=512,
            #     do_sample=True,
            #     temperature=0.7,
            #     top_p=0.95,
            #     top_k=40,
            #     repetition_penalty=1.1,
            # )

            # print(pipe(prompt_template)[0]["generated_text"])

    for stage, shapes in Counter.shape_counter.items():
        print(f"Stage {stage}:")
        for shape, count in shapes.items():
            print(f"  - input: {shape[0]}, weight: {shape[1]}: {count}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default="/models/Llama-2-7b-chat-hf/")
    args = parser.parse_args()
    inspect_model(args)
