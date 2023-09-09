import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert_utils import *
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers import AutoConfig

# class BertQAConfig(PretrainedConfig):

#     model_type = "BertQA"

#     def __init__(
#             self,
#             vocab_size=64001,
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#             hidden_act="relu",
#             hidden_dropout_prob=0.1,
#             attention_probs_dropout_prob=0.1,
#             max_position_embeddings=258,
#             type_vocab_size=1,
#             initializer_range=0.02,
#             layer_norm_eps=1e-05,
#             pad_token_id=1,
#             bos_token_id=0,
#             eos_token_id=2,
#             position_embedding_type="absolute",
#             use_cache=True,
#             classifier_dropout=None,
#             tokenizer_class="PhobertTokenizer",
#             torch_dtype="float32",
#             **kwargs
#     ):
#         super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.intermediate_size = intermediate_size
#         self.hidden_act = hidden_act
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.max_position_embeddings = max_position_embeddings
#         self.type_vocab_size = type_vocab_size
#         self.initializer_range = initializer_range
#         self.layer_norm_eps = layer_norm_eps
#         self.pad_token_id = pad_token_id
#         self.bos_token_id = bos_token_id
#         self.eos_token_id = eos_token_id
#         self.position_embedding_type = position_embedding_type
#         self.use_cache = use_cache
#         self.classifier_dropout = classifier_dropout
#         self.tokenizer_class = tokenizer_class
#         self.torch_dtype = torch_dtype


class BertQA(nn.Module):

    # config_class = BertQAConfig

    def __init__(self, config, freeze_bert=False, dropout_rate=0.1, hidden_units=768, class_weights=None):
        super(BertQA, self).__init__()
        self.device = config.device

        self.bert = load_pretrained_bert(config)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.bert.trainable = False
        self.use_pooler = config.use_pooler
        if self.use_pooler:
            self.qa_outputs = nn.Linear(in_features=hidden_units, out_features=config.num_labels)
        else:
            self.qa_outputs_cat = nn.Linear(in_features=hidden_units*4, out_features=config.num_labels)

        self.class_weights = config.class_weights

        self.bert.init_weights()

    def compute(self, input_ids, attention_mask=None):

        outputs = self.bert(input_ids,
                           attention_mask=attention_mask)
        pooler_output = outputs[1]  # pooler output using for classification task next sent

        if self.use_pooler:
            final_output = self.dropout(pooler_output)
        else:
            last4_hidden_states = outputs['hidden_states'][-4:]
            last4_hidden_states = torch.stack(last4_hidden_states, dim=1)

            last4_cls_hs = last4_hidden_states[:,:,0,:]
            last4_cls_hs = torch.reshape(last4_cls_hs, (last4_cls_hs.shape[0], last4_cls_hs.shape[1]*last4_cls_hs.shape[2]))

            final_output = self.dropout(last4_cls_hs)

        return final_output
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            final_output = self.compute(input_ids, attention_mask)
            if self.use_pooler:
                logits = self.qa_outputs(final_output)
            else:
                logits = self.qa_outputs_cat(final_output)
            return logits
    
    def loss(self, input_ids, attention_mask, label):
        target = label

        final_output = self.compute(input_ids, attention_mask)
        if self.use_pooler:
            logits = self.qa_outputs(final_output)
        else:
            logits = self.qa_outputs_cat(final_output)

        class_weight = torch.FloatTensor(self.class_weights).to(self.device)
        loss = F.cross_entropy(logits, target, weight=class_weight)

        predict_value = torch.max(logits, 1)[1]
        list_predict = predict_value.cpu().numpy().tolist()
        list_target = target.cpu().numpy().tolist()

        return loss, list_predict, list_target








        

