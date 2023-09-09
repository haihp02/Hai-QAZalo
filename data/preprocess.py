import os

import py_vncorenlp
import csv

word_segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='E:\\vncorenlp')

# test_file = open(r'E:/OneDrive - Hanoi University of Science and Technology/Co so nganh/Engineering Practicum - NLP Lab/Hai-QAZalo/data/train/dev_v3_viet_facebook_500_sent.csv', 'r', encoding='utf8')
# test_output = open('./train/word_segmented/test_output.csv')

def text_word_segment(text):
    return ' '.join(word_segmenter.word_segment(text))

def qa_word_segment(qa: str):
    question, answer, match = qa.split('\t')
    segmented_q, segmented_a = text_word_segment(question), text_word_segment(answer)
    segmented_qa = '\t'.join((segmented_q, segmented_a, match))
    return segmented_qa

def text_file_segment(input_filepath, output_filepath):
    input_file = open(input_filepath, 'r', encoding='utf8')
    output_file = open(output_filepath, 'a', encoding='utf8')
    for line in input_file.readlines():
        word_segmented = qa_word_segment(line)
        output_file.writelines(word_segmented)
    input_file.close()
    output_file.close()

def merge_files_dir(dir_path, output_file_name):
    in_files = os.listdir(dir_path)
    out_file_path = os.path.join(dir_path, output_file_name)
    output_file = open(out_file_path, 'w', encoding='utf8')
    data = None
    for file in in_files:
        f_path = dir_path + '/' + file
        with open(f_path, 'r', encoding='utf8') as f:
            for line in f:
                output_file.write(line)

    output_file.close()






# if __name__ == '__main__':
    # dir_path = 'E:/OneDrive - Hanoi University of Science and Technology/Co so nganh/Engineering Practicum - NLP Lab/Hai-QAZalo/data/train'
    # input_filepaths = [f for f in os.listdir(dir_path) if os.path.isfile(dir_path+'/'+f)]
    # output_dir_path = 'E:/OneDrive - Hanoi University of Science and Technology/Co so nganh/Engineering Practicum - NLP Lab/Hai-QAZalo/data/train/word_segmented'
    # for filepath in input_filepaths:
    #     input_filepath = dir_path+'/'+filepath
    #     output_filepath = output_dir_path+'/'+filepath
    #     text_file_segment(input_filepath, output_filepath)