import argparse
import gc
import math
import os
import time
import json
import torch
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--name", type=str, help="Name path", required=True,default="bigscience/bloom")
    parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")

    return parser.parse_args()


batch_size = 1
num_tokens = 30
args = get_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()

rank = local_rank


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


print_rank0(f"Using {world_size} gpus")
model_name = args.name
print_rank0(f"Loading model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)



# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16

# print(get_max_memory_per_gpu_dict())

infer_dtype = args.dtype
if infer_dtype == "int8":
    dtype = torch.int8

kwargs = dict(
    device_map="auto",
)


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


# balanced_low_0 - because it allows a larger batch size with multiple GPUs
if get_world_size() > 1:
    kwargs["device_map"] = "balanced_low_0"


if infer_dtype == "int8":
    print_rank0("Using `load_in_8bit=True` to use quanitized model")
    kwargs["load_in_8bit"] = True
else:
    kwargs["torch_dtype"] = dtype


model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


### Generate



generate_kwargs = dict(max_new_tokens=num_tokens, min_new_tokens=num_tokens, do_sample=False)

def generate(input_length):

    input_tokens = torch.randint(0,10000,(batch_size,input_length)).to("cuda:0")
    print(input_tokens.shape)
    start_time = time.time()
    _ = model.generate(input_tokens, **generate_kwargs,pad_token_id=tokenizer.eos_token_id)
    end_time = time.time()

    return end_time - start_time
    
input_length = {
    "zero_shot":[101,161],
    'one_shot':[198,517],
    'ten_shot':[952,2490],
}

### Benchmark
torch.cuda.empty_cache()
gc.collect()

print_rank0("*** Running benchmark")
# warm up
for i in range(1):
    _ = generate(input_length=512)
torch.cuda.synchronize()
results = {}
for setting in ['zero_shot','one_shot','ten_shot']:
    iscl_len = input_length[setting][1]
    gpt4_len = input_length[setting][0]
    results[setting] = {}
    for name,length in {'iscl':iscl_len,'gpt4':gpt4_len}.items():
        total_time = generate(input_length = length)
        torch.cuda.empty_cache()
        gc.collect()
        results[setting][name] = total_time
print(json.dumps(results,indent=4))

