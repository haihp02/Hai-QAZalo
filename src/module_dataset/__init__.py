import torch
from random import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# file data loader custom from util_squad in hugging face transfomrers
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 question_text,
                 doc_text,
                 is_has_answer=None):
        self.question_text = question_text
        self.doc_text = doc_text
        self.is_has_answer = is_has_answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "question_text: %s" % (
            self.question_text)
        s += ", doc_text: [%s]" % (self.doc_text)
        s += ", is_has_answer: %r" % (self.is_has_answer)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_index,
                 input_ids,
                 input_mask,
                #  segment_ids,
                 is_has_answer=None):
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        # self.segment_ids = segment_ids
        self.is_has_answer = is_has_answer

def read_squad_example_from_file(input_data, is_training=True):
    with open(input_data, 'r', encoding='utf8') as rf:
        examples = []
        for e_line in rf.readlines():
            e_line = e_line.replace('\n', '')
            arr_e_line = e_line.split('\t')

            if is_training:
                if arr_e_line[2] == 'true':
                    is_has_answer = 1
                else:
                    is_has_answer = 0
            else:
                is_has_answer = None
            
            question = arr_e_line[0].casefold().strip()
            doc = arr_e_line[1].casefold().strip()

            example = SquadExample(question_text=question,
                                   doc_text=doc,
                                   is_has_answer=is_has_answer)
            examples.append(example)
        
    return examples
    
def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in enumerate(examples):

        # segment_ids = []

        query_ids = tokenizer.encode(example.question_text)
        sep_id = query_ids[-1]
        query_ids = query_ids[:-1]
        if len(query_ids) > max_query_length:
            query_ids = query_ids[0:max_query_length]
        query_ids.append(sep_id)
        
        # segment_ids.extend([0] * len(query_ids))

        doc_ids = tokenizer.encode(example.doc_text)[:-1]
        doc_ids[0] = sep_id
        # The -1 accounts for [SEP] at the end of doc
        max_tokens_for_doc = max_seq_length - len(query_ids) - 1
        if len(doc_ids) > max_tokens_for_doc:
            doc_ids = doc_ids[0:max_tokens_for_doc]
        doc_ids.append(sep_id)
        # segment_ids.extend([1] * len(doc_ids))

        input_ids = query_ids + doc_ids

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        input_len = len(input_ids)
        if input_len < max_seq_length:
            input_ids.extend([0] * (max_seq_length - input_len))
            input_mask.extend([0] * (max_seq_length - input_len))
            # segment_ids.extend([0] * (max_seq_length - input_len))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                example_index=example_index,
                input_ids=input_ids,
                input_mask=input_mask,
                # segment_ids=segment_ids,
                is_has_answer=example.is_has_answer,
            )
        )
    
    return features

def load_squad_to_torch_dataset(path_input_data,
                                tokenizer,
                                max_seq_length=256,
                                max_query_length=64,
                                batch_size=20,
                                is_training=True):
    examples = read_squad_example_from_file(path_input_data)
    shuffle(examples)

    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=max_seq_length, max_query_length=max_query_length)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if is_training:
        all_answerable = torch.tensor([f.is_has_answer for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_answerable)
        # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_answerable)

    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return dataset, train_dataloader

