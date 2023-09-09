import torch
from transformers import AutoModel, AutoTokenizer
import py_vncorenlp
# from typing import NamedTuple

def load_pretrained_bert(config, path="vinai/phobert-base-v2"):
    return AutoModel.from_pretrained(path, config=config)

def load_tokenizer(path="vinai/phobert-base-v2"):
    return AutoTokenizer.from_pretrained(path)

class BertArgs():
    do_lower_case: bool = True
    R: str = 'vinai/phobert-base-v2'
    folder_model: str = None

    path_input_train_data: str = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\data\\train\\word_segmented\\dev_v3_viet_facebook_500_sent.csv'
    path_input_test_data: str = None
    path_input_validation_data = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\data\\train\\word_segmented\\val_origin_1k.csv'

    load_data_from_pt: bool = False
    path_pt_train_dataset: str = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\data\\torch_data\\train.pkl'
    path_pt_test_dataset: str = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\data\\torch_data\\test.pkl'
    path_pt_validation_dataset: bool = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\data\\torch_data\\validate.pkl'

    path_log_file: str = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\src\\module_train\\log.txt'
    output_dir: str = 'E:\\OneDrive - Hanoi University of Science and Technology\\Co so nganh\\Engineering Practicum - NLP Lab\\Hai-QAZalo\\src\\module_train\\ouput'

    max_seq_length: int = 256
    max_query_length: int = 50

    batch_size: int = 16

    num_labels: int = 2
    class_weights: list = [1, 1]

    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    use_pooler: bool = True
    output_hidden_states: bool = True
    # if use_pooler= False (mean concat 4 CLS in 4 last hidden_state BERT)
    # you need to set output_hidden_states=True.

    num_train_epochs: int = 1
    save_steps: int = int(400/ gradient_accumulation_steps)

    no_cuda: bool = True
    n_gpu: int = 1
    device = 'cpu'
    seed: int = 42