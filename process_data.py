import numpy as np
import collections
import operator

def read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    sentences=[]
    sentences_label=[]
    word_list,label_list=[],[]
    for line in lines:
        if len(line)==0 or line=="\n" or line.startswith('-DOCSTART-'):
            if len(word_list)>0:
                sentences.append(word_list)
                sentences_label.append(label_list)
                word_list=[]
                label_list=[]
            continue
        line_split=line.strip().split()
        word=line_split[0]
        label=line_split[-1]
        word_list.append(word)
        label_list.append(label)
    if len(word_list)>0:
        sentences.append(word_list)
        sentences_label.append(label_list)
    return sentences,sentences_label

def get_word_tag2id(sentences,sentences_label):
    word2id={}
    word2id['UNK']=len(word2id)
    tag2id={}
    tag_set=set()
    all_words=[]
    for word_list,label_list in zip(sentences,sentences_label):
        assert len(word_list)==len(label_list)
        for word,tag in zip(word_list,label_list):
            tag_set.add(tag)
            all_words.append(word)
    for tag in tag_set:
        tag2id[tag]=len(tag2id)
    print(tag2id)
    print("There are totally %d words in the corpus" % len(all_words))
    counter_words=collections.Counter(all_words)
    sorted_words=sorted(counter_words.items(),key=operator.itemgetter(1),reverse=True)
    for word,freq in sorted_words:
        word2id[word]=len(word2id)
    print("length of word2id is ",len(word2id))
    return word2id,tag2id

def get_embedding(word2id,embedding_dim,pre_trained_path=None):
    embedding_size=len(word2id)
    if pre_trained_path==None:
        return np.random.uniform(-1.0,1.0,size=(embedding_size,embedding_dim))
    assert pre_trained_path!=None
    embedding_matrix=[]
    with open(pre_trained_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    print("length of glove 100d.txt",len(lines))
    word_list=[]
    for word in word2id.keys():
        word_list.append(word)
    word_embedding={}
    for line in lines:
        line_split=line.strip().split()
        word=line_split[0]
        vector_string=line_split[1:]
        assert len(vector_string)==embedding_dim
        word_embedding[word]=[float(number) for number in vector_string]
        
    for each_word in word_list:
        if each_word in word_embedding:
            embedding_matrix.append(word_embedding[each_word])
        else:#the word in the current train_file not in glove pre_trained file
            embedding_matrix.append(np.random.uniform(-1.0,1.0,size=(embedding_dim,)))
    return np.array(embedding_matrix)
            
def sentence_to_id(sentences,sentences_label,word2id,tag2id):
    sentences_id,sentences_label_id=[],[]
    for word_list,label_list in zip(sentences,sentences_label):
        assert len(word_list)==len(label_list)
        word_id_list,label_id_list=[],[]
        for word,tag in zip(word_list,label_list):
            assert tag in tag2id
            tag_id=tag2id[tag]
            if word in word2id:
                word_id=word2id[word]
            else:
                word_id=word2id['UNK']
            label_id_list.append(tag_id)
            word_id_list.append(word_id)
        sentences_id.append(word_id_list)
        sentences_label_id.append(label_id_list)
    return sentences_id,sentences_label_id

def pad_seq(sentences_id,sentences_label_id,max_seq_length):
    pad_word_id=[]
    pad_tag_id=[]
    actual_length=[]
    for word_id_list,tag_id_list in zip(sentences_id,sentences_label_id):
        assert len(word_id_list)==len(tag_id_list)
        length=len(word_id_list)
        if length>=max_seq_length:
            pad_word_id.append(word_id_list[:max_seq_length])
            pad_tag_id.append(tag_id_list[:max_seq_length])
            actual_length.append(max_seq_length)
        else:
            pad_word_id.append(word_id_list[:max_seq_length]+[0]*(max_seq_length-length))
            pad_tag_id.append(tag_id_list[:max_seq_length]+[0]*(max_seq_length-length))
            actual_length.append(length)
    return pad_word_id,pad_tag_id,actual_length

def batch_yield(pad_word_id,pad_tag_id,actual_length,batch_size):
    total_length=len(pad_word_id)
    assert total_length==len(actual_length)==len(pad_tag_id)
    shuffled=np.random.permutation(total_length)
    #将每一个句子随机打乱送进网络，因为句子的先后顺序对于预测来说是没有影响的
    pad_word_id=np.array(pad_word_id)[shuffled]
    pad_tag_id=np.array(pad_tag_id)[shuffled]
    actual_length=np.array(actual_length)[shuffled]
    num_batches=total_length//batch_size
    start=0
    for i in range(num_batches):
        yield pad_word_id[start:start+batch_size],pad_tag_id[start:start+batch_size],actual_length[start:start+batch_size]
        start+=batch_size