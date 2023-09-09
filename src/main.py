from bert_model.bert_utils import BertArgs
from bert_model.bert import BertQA
from module_train.train_bert import train_squad
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaConfig
import torch
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

def train_model_bert(args: BertArgs):
    config = RobertaConfig.from_pretrained(args.R)

    config = config.to_dict()
    config.update({"device": args.device})
    config.update({"use_pooler": args.use_pooler})
    config.update({"class_weights": args.class_weights})
    config.update({"output_hidden_states": args.output_hidden_states})
    config = RobertaConfig.from_dict(config)

    tokenizer = AutoTokenizer.from_pretrained(args.R)
    model = BertQA(config=config)
    model = model.to(args.device)

    if args.folder_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.folder_model)
        torch_model_checkpoint = torch.load(os.path.join(args.folder_model, 'torch_model_checkpoint.pt'))
        model.load_state_dict(torch_model_checkpoint['model_state_dict'], strict=False)

    train_squad(args=args, model=model, tokenizer=tokenizer)
    
    # AutoConfig.register("BertQA", BertQAConfig)
    # AutoModel.register(BertQAConfig, BertQA)

if __name__ == '__main__':
    args = BertArgs()
    args.folder_model = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\src\\module_train\\ouput\\epoch0_step26'
    train_model_bert(args)