import sys  
import os  
import logging

def get_chrfpp_score(hyps,refs):
    import sacrebleu
    assert len(hyps) == len(refs)
    score = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    return score

def get_bleu_score(hyps,refs,**kwargs,):
    import sacrebleu
    assert len(hyps) == len(refs)
    score = sacrebleu.corpus_bleu(hyps,[refs],**kwargs).score
    return score

def get_comet_score(src,hyps,refs):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src": s,
            "mt": h,
            "ref": r
        }
        for s,h,r in zip(src,hyps,refs)
    ]
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.system_score

def get_comet_kiwi_score(src,hyps):
    from comet import load_from_checkpoint
    model_path = "wmt22-cometkiwi-da/checkpoints/model.ckpt"
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src":s,
            "mt":m,
        }
        for s,m in zip(src,hyps)
    ]
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.system_score

def get_bleurt_score(hyps,refs,batch_size=8,device='cuda:0'):
    # pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
    import torch
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
    model.eval()
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

    hyps = [hyps[idx:idx+batch_size] for idx in range(0,len(hyps),batch_size)]
    refs = [refs[idx:idx+batch_size] for idx in range(0,len(refs),batch_size)]

    ret = []
    for hyp,ref in zip(hyps,refs):
        with torch.no_grad():
            inputs = tokenizer(ref, hyp, padding='longest', return_tensors='pt', max_length=model.config.max_position_embeddings,truncation=True).to(device)
            res = model(**inputs).logits.flatten().cpu().tolist()
            ret.extend(res)
    return sum(ret)/len(ret)

def eval_translation(src,hyps,refs,tokenize='flores200'):
    assert len(refs)==len(hyps)==len(src),f"len(refs)={len(refs)},len(hyps)={len(hyps)},len(src)={len(src)}"
    return {
        "comet":round(get_comet_score(src,hyps,refs)*100,1),
        'comet_kiwi':round(get_comet_kiwi_score(src,hyps)*100,1),
        "bleurt":round(get_bleurt_score(hyps,refs)*100,1),
        "bleu":round(get_bleu_score(hyps,refs,tokenize=tokenize),1),
    }