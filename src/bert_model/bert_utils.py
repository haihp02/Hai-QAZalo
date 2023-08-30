import torch
from transformers import AutoModel, AutoTokenizer
import py_vncorenlp

def load_pretrained_bert(path="vinai/phobert-base-v2"):
    return AutoModel.from_pretrained(path)

def load_tokenizer(path="vinai/phobert-base-v2")
    return AutoTokenizer.from_pretrained(path)



# py_vncorenlp.download_model(save_dir='E:\\vncorenlp')
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='E:\\vncorenlp')

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

sentence = 'Chúng_tôi_là_mèo_méo_meo'

# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
# #
# word_segmented_sentence = ' '.join(rdrsegmenter.word_segment(sentence))
# print(word_segmented_sentence)
encoded_sentence = tokenizer.encode(sentence)
print(encoded_sentence)
print(tokenizer.decode(encoded_sentence))
for token in encoded_sentence:
    print(tokenizer.decode(token))


print()

# inputs_ids = torch.tensor([tokenizer.encode(sentence)])
# inputs_ids_word_segmented_sentence = torch.tensor([tokenizer.encode(word_segmented_sentence)])
#
# print(inputs_ids)
# # print()
# print(inputs_ids_word_segmented_sentence)

# with torch.no_grad():
#     features = phobert(inputs_ids)
