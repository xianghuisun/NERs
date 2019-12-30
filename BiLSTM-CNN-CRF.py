import numpy as np
import operator
import pickle
from char_consider_processing import *
from process_data import read_file
import tensorflow as tf

import os
os.environ['CUDA_VISIABLE_DEVICES']='0'
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True

    

class Config:
    def __init__(self,num_tags):
        self.batch_size=64
        self.max_seq_len=125
        self.max_word_len=20
        self.embedding_dim=100
        self.hidden_dim=128
        self.num_tags=num_tags
        self.char_embedding_dim=30
        
        

class Char_NER:
    def __init__(self,config):
        self.batch_size=config.batch_size
        self.max_seq_len=config.max_seq_len
        self.max_word_len=config.max_word_len
        self.embedding_dim=100
        self.char_hidden_dim=25
        self.hidden_dim=config.hidden_dim
        self.num_tags=config.num_tags
        self.char_embedding_dim=config.char_embedding_dim
        self.model_save_path="/home/sun_xh/ner_code/assignment_final/log/char_model.ckpt"
        self.use_crf=True
        tf.reset_default_graph()
    
    def placeholder_op(self):
        self.word_ids=tf.placeholder(shape=[self.batch_size,self.max_seq_len],dtype=tf.int32)
        self.label_ids=tf.placeholder(shape=[self.batch_size,self.max_seq_len],dtype=tf.int32)
        self.char_ids=tf.placeholder(shape=[self.batch_size,self.max_seq_len,self.max_word_len],dtype=tf.int32)
        self.seq_length=tf.placeholder(shape=[self.batch_size],dtype=tf.int32)
    
    def embedding_op(self,embedding_matrix,char_embedding_matrix):
        embedding_matrix=tf.constant(embedding_matrix,dtype=tf.float32)
        char_embedding_matrix=tf.constant(char_embedding_matrix,dtype=tf.float32)
        self.word_embeddings=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.word_ids)
        self.char_embeddings=tf.nn.embedding_lookup(params=char_embedding_matrix,ids=self.char_ids)#[84,30]
        print(self.char_embeddings.shape)
        print('-'*100)
    
    def char_bilstm_layer(self):
        char_cell_fw=tf.contrib.rnn.BasicLSTMCell(num_units=self.char_hidden_dim)
        char_cell_bw=tf.contrib.rnn.BasicLSTMCell(num_units=self.char_hidden_dim)
        
        assert self.char_embeddings.shape==(self.batch_size,self.max_seq_len,self.max_word_len,self.char_embedding_dim)
        lstm_input=tf.reshape(tensor=self.char_embeddings,shape=[self.batch_size,self.max_seq_len,self.max_word_len*self.char_embedding_dim])
        outputs,states=tf.nn.bidirectional_dynamic_rnn(char_cell_fw,char_cell_bw,inputs=lstm_input,dtype=tf.float32)
        self.char_outputs_concat=tf.concat(values=[outputs[0],outputs[1]],axis=-1)
        assert self.char_outputs_concat.shape==(self.batch_size,self.max_seq_len,self.char_hidden_dim*2)
    
    def BiLSTM_layer(self):
        assert self.word_embeddings.shape==(self.batch_size,self.max_seq_len,self.embedding_dim)
        bilstm_input=tf.concat(values=[self.word_embeddings,self.char_outputs_concat],axis=-1)
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=bilstm_input,sequence_length=self.seq_length,dtype=tf.float32)
        bilstm_out=tf.concat(values=[outputs[0],outputs[1]],axis=-1)
        self.lstm_out=tf.nn.dropout(bilstm_out,keep_prob=0.5)
        assert self.lstm_out.shape==(self.batch_size,self.max_seq_len,self.hidden_dim*2)

    def project_layer(self):
        
        word_char_embedding=tf.concat(values=[self.char_outputs_concat,self.lstm_out],axis=-1)
        assert word_char_embedding.shape==(self.batch_size,self.max_seq_len,self.hidden_dim*2+self.char_hidden_dim*2)
        concat_dim=word_char_embedding.get_shape().as_list()[-1]

        
        weights1=tf.Variable(tf.random_normal(shape=[concat_dim,200],dtype=tf.float32))
        biases1=tf.Variable(tf.random_normal(shape=[200],dtype=tf.float32))
        weights2=tf.Variable(tf.random_normal(shape=[200,self.num_tags],dtype=tf.float32))
        biases2=tf.Variable(tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        
        x=tf.reshape(tensor=word_char_embedding,shape=[-1,concat_dim])
        out1=tf.matmul(x,weights1)+biases1
        assert out1.shape==(self.batch_size*self.max_seq_len,200)
        out2=tf.matmul(tf.nn.sigmoid(out1),weights2)+biases2
        self.logits=tf.reshape(tensor=out2,shape=[self.batch_size,self.max_seq_len,self.num_tags])
        
        
    def loss_layer(self,train_test="train"):
        #由于是多分类问题，所以采用交叉商作为损失函数，最后一层接softmax，
        #tensorflow中softmax_cross_entropy_with_logits()已经将两个步骤合二为一
        if train_test=="train":
            if self.use_crf==True:
                log_likelihood,self.trans_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,sequence_lengths=self.seq_length)
                self.loss=tf.reduce_mean(-log_likelihood)
            else:
                losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label_ids)
                assert losses.shape==(self.batch_size,self.max_length)
                mask=tf.sequence_mask(self.sequence_length,maxlen=self.max_length)
                assert mask.shape==(self.batch_size,self.max_length)#mask是一个这样的矩阵，每一行长度为max_length，每个位置为True或者false,取决于当前句子的长度
                boolean_mask=tf.boolean_mask(losses,mask)#len(boolean_mask)==这batch_size个句子的每一个句子的真实长度之和
                self.loss=tf.reduce_mean(boolean_mask)
        else:
            assert train_test=="test"
            if self.use_crf==True:
                _,self.trans_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,sequence_lengths=self.seq_length)
            else:
                assert self.use_crf==False
                argmax_logits=tf.argmax(self.logits,axis=-1)#type(argmax_logits)==tf.int64
                self.predict_index=tf.cast(argmax_logits,dtype=tf.int32)
        
        self.train_op=tf.train.AdamOptimizer(0.001).minimize(self.loss)
    
    def build_graph(self,embedding_matrix,char_embedding_matrix,train_test="train"):
        self.placeholder_op()
        self.embedding_op(embedding_matrix,char_embedding_matrix)
        self.char_bilstm_layer()
        self.BiLSTM_layer()
        self.project_layer()
        self.loss_layer(train_test)
        print("The graph has been built!")
        
    
    def train(self,pad_word,pad_tag,pad_char,actual_length):
        saver=tf.train.Saver()
        assert pad_word.shape==pad_tag.shape==(pad_char.shape[0],pad_char.shape[1])
        assert actual_length.shape[0]==pad_char.shape[0]
        num_batches=pad_word.shape[0]//self.batch_size
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                batches=batch_generate(pad_word,pad_tag,pad_char,actual_length,self.batch_size)
                total_loss=0.0
                for step,(batch_seq,batch_seq_tag,batch_char,batch_length) in enumerate(batches):
                    assert batch_char.shape==self.char_ids.shape==(self.batch_size,self.max_seq_len,self.max_word_len)
                    assert batch_seq.shape==batch_seq_tag.shape==self.word_ids.shape==self.label_ids.shape==(self.batch_size,self.max_seq_len)
                    assert batch_length.shape==self.seq_length.shape==(self.batch_size,)
                    feed_dict={self.word_ids:batch_seq,self.label_ids:batch_seq_tag,self.char_ids:batch_char,self.seq_length:batch_length}
                    loss_val,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                    total_loss+=loss_val
                print("In %d epoch, the loss value is %f" %(epoch,total_loss/num_batches))
                saver.save(sess,save_path=self.model_save_path)
    
    def test(self,pad_word,pad_tag,pad_char,actual_length,tag2id):
        saver=tf.train.Saver()
        TP_matrix=np.zeros(shape=(self.num_tags,))
        TP_FP_matrix=np.zeros(shape=(self.num_tags,))
        TP_FN_matrix=np.zeros(shape=(self.num_tags,))  
        correct=0
        total=0
        with tf.Session(config=gpu_config) as sess:
            saver.restore(sess,save_path=self.model_save_path)
            batches=batch_generate(pad_word,pad_tag,pad_char,actual_length,self.batch_size)
            for step,(batch_seq,batch_tag,batch_char,batch_length) in enumerate(batches):
                assert batch_char.shape==self.char_ids.shape==(self.batch_size,self.max_seq_len,self.max_word_len)
                assert batch_seq.shape==batch_seq_tag.shape==self.word_ids.shape==self.label_ids.shape==(self.batch_size,self.max_seq_len)
                assert batch_length.shape==self.seq_length.shape==(self.batch_size,)
                feed_dict={self.word_ids:pad_word,self.char_ids:pad_char,self.seq_length:batch_length}
                if self.use_crf:
                    predict_logits,trans_matrix=sess.run([self.logits,self.trans_matrix],feed_dict=feed_dict)
                    assert predict_logits.shape==(self.batch_size,self.max_seq_len,self.num_tags)
                    assert trans_matrix.shape==(self.num_tags,self.num_tags)
                    for batch_predict,batch_golden_tag,batch_true_len in zip(predict_logits,batch_tag,batch_length):
                        assert batch_predict.shape==(self.max_seq_len,self.num_tags)
                        assert batch_golden_tag.shape==(self.num_tags,)
                        batch_predict=batch_predict[:batch_true_len]
                        batch_golden_tag=batch_golden_tag[:batch_true_len]
                        assert batch_predict.shape==(batch_true_len,self.num_tags)
                        viterbi_seq,viterbi_score=tf.contrib.crf.viterbi_decode(batch_predict,trans_matrix)
                        assert type(viterbi_seq)==list and len(viterbi_seq)==batch_true_len and type(viterbi_score)==np.float32
                        golden_score=list(batch_golden_tag)
                        assert len(golden_score)==len(viterbi_seq)
                        for predict_id,golden_id in zip(viterbi_seq,golden_score):
                            if predict_id==golden_id:
                                correct+=1
                                TP_matrix[predict_id]+=1
                            total+=1
                            TP_FP_matrix[predict_id]+=1
                            TP_FN_matrix[golden_id]+=1
                else:
                    assert self.use_crf==False
                    predict_index=sess.run(self.predict_index,feed_dict=feed_dict)
                    assert predict_index.shape==(self.batch_size,self.max_seq_len)==batch_tag.shape
                    for predict_,golden_,true_len in zip(predict_index,batch_tag,batch_length):
                        predict_=predict_[:true_len]
                        golden_=golden_[:true_len]
                        for predict_id,golden_id in zip(predict_,golden_):
                            if predict_id==golden_id:
                                correct+=1
                                TP_matrix[predict_id]+=1
                            total+=1
                            TP_FP_matrix[predict_id]+=1
                            TP_FN_matrix[golden_id]+=1                            
            result={}
            id2tag={key_:value for value,key_ in tag2id.items()}
            print(id2tag)
            f1_score_avg=0.0
            for tag in tag2id.keys():
                result[tag]={"precision":0.0,"recall":0.0,"f1_score":0.0}
            for tag in tag2id.keys():
                id_=tag2id[tag]
                result[tag]["precision"]=TP_matrix[id_]/TP_FP_matrix[id_]
                result[tag]["recall"]=TP_matrix[id_]/TP_FP_matrix[id_]
                result[tag]['f1_score']=2*result[tag]["precision"]*result[tag]["recall"]/(result[tag]["precision"]+result[tag]["recall"])
                f1_score_avg+=result[tag]["f1_score"]
            print("correct / total is ",correct/total)
            print("avg f1 score is ",f1_score_avg/self.num_tags)
            for tag in tag2id.keys():
                print(tag+"/"+str(result[tag]["f1_score"]))
                

def train_model(train_path,store_path,con):
    sentences,sentences_label=read_file(train_path)
    word2id,tag2id,char2id=get_parameter2id(sentences,sentences_label)
    word_sens_id, tag_sens_id, char_sens_id=sentence_to_id(sentences,sentences_label,word2id,tag2id,char2id)
    pad_word, pad_tag, pad_char, actual_length=pad_data(word_sens_id,tag_sens_id,char_sens_id,con.max_seq_len,con.max_word_len)
    word_matrix,char_matrix=get_embedding_matrix(word2id, char2id,embedding_dim=100, char_embedding_dim=30, glove_path=None)
    data=(word2id,tag2id,char2id,word_matrix,char_matrix)
    with open(store_path,'wb') as f:
        pickle.dump(data,f)
        
    model=Char_NER(config=con)
    model.build_graph(word_matrix,char_matrix,train_test="train")
    model.train(pad_word,pad_tag,pad_char,actual_length)
    print("Model has been trained over!")

def test_model(test_path,store_path,con):
    with open(store_path,'rb') as f:
        word2id,tag2id,char2id,word_embedding_matrix,char_embedding_matrix=pickle.load(f)
    sentences,sentences_label=read_file(test_path)
    word_sens_id, tag_sens_id, char_sens_id=sentence_to_id(sentences,sentences_label,word2id,tag2id,char2id)
    pad_word, pad_tag, pad_char, actual_length=pad_data(word_sens_id,tag_sens_id,char_sens_id,con.max_seq_len,con.max_word_len)
    model=Char_NER(config=con)
    model.build_graph(embedding_matrix=word_embedding_matrix,char_embedding_matrix=char_embedding_matrix,train_test="test")
    model.test(pad_word,pad_tag,pad_char,actual_length,tag2id)
    
if __name__ == "__main__":
    train_path='/home/sun_xh/ner_code/data/train.txt'
    test_path='/home/sun_xh/ner_code/data/test.txt'
    store_path='/home/sun_xh/ner_code/data/char_parameter.pkl'
    con=Config(num_tags=len(tag2id))
    train_model(train_path,store_path,con)
    test_model(test_path,store_path,con)
    
    
    
                
                
                
            
        
        

