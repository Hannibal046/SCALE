import argparse
import datasets
import pickle
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from CONSTANT import flores_code2lang
from evaluation import eval_translation
import os
os.environ["TOKENIZERS_PARALLELISM"]="false"


def generate(batch,model,tokenizer,device,trg_lang,SPIECE_UNDERLINE = "▁",num_return_sequences=1):
    hyps = []
    token_score_pairs = []
    
    inputs = tokenizer(batch, padding=True,return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang],
        max_length=100,
        return_dict_in_generate=True,
        output_scores=True,
        num_return_sequences = num_return_sequences,
    )
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    ).cpu()
    hyps.extend(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True,clean_up_tokenization_spaces=True))
    
    ## maybe further improved https://github.com/huggingface/transformers/blob/e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6/src/transformers/tokenization_utils_base.py#L3544
    for batch_index in range(len(batch)*num_return_sequences):
        token_score_pair = []
        for tok, score in zip(outputs.sequences[batch_index][2:], transition_scores[batch_index][2:]): ## 2 is bos+lang_id
            if tok == tokenizer.eos_token_id: break
            if tok == tokenizer.unk_token_id: continue
            ## for subword merging
            current_token = tokenizer.convert_ids_to_tokens(tok.item())
            current_score = np.exp(score.numpy())

            if current_token.startswith(SPIECE_UNDERLINE):
                token_score_pair.append([current_token,[current_score]])
            else:
                if len(token_score_pair)==0:
                    token_score_pair.append([current_token,[current_score]])
                else:
                    token_score_pair[-1][0] += current_token
                    token_score_pair[-1][1].append(current_score)
            
        token_score_pair = [(x[0].replace(SPIECE_UNDERLINE,""),sum(x[1])/len(x[1])) for x in token_score_pair]
        token_score_pairs.append(token_score_pair)

    return hyps,token_score_pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--src_lang",required=True)
    parser.add_argument("--trg_lang",required=True)
    parser.add_argument("--model_type",default="facebook/nllb-200-3.3B")
    parser.add_argument("--data_split",required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    device = 0
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type,src_lang=args.src_lang)
    dataset = datasets.load_dataset("facebook/flores","all")
    
    src = [x[f"sentence_{args.src_lang}"] for x in dataset[args.data_split]]
    batched_src = [src[idx:idx+args.batch_size] for idx in range(0,len(src),args.batch_size)]
    refs = [x[f"sentence_{args.trg_lang}"] for x in dataset[args.data_split]]

    hyps = []
    token_score_pairs = []
    for batch in batched_src:
        batch_hyps,batch_pairs = generate(batch,model,tokenizer,device,args.trg_lang)
        hyps.extend(batch_hyps)
        token_score_pairs.extend(batch_pairs)

    # print(eval_translation(src,hyps,refs))

    if args.output_dir is not None:
        os.makedirs(args.output_dir,exist_ok=True)
        with open(os.path.join(args.output_dir,"hyps.txt"),'w') as f:
            for hyp in hyps:
                f.write(hyp+'\n')
        with open(os.path.join(args.output_dir,"hyps.scores.txt"),'w') as f:
            for token_score_pair in token_score_pairs:
                f.write(" ".join([f'{x[0]}({x[1]:.2f})' for x in token_score_pair])+'\n')