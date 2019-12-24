import tensorflow as tf
from process_data import *

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True

class Config_class:
    def __init__(self,vocab_size,num_tags,model_save_path):
        self.embedding_dim=100
        self.max_seq_length=125
        self.embedding_size=vocab_size
        self.hidden_dim=128
        self.num_tags=num_tags
        self.model_save_path=model_save_path
        self.batch_size=64


class BiLSTM_CRF:
    def __init__(self,Config):
        self.embedding_dim=Config.embedding_dim
        self.embedding_size=Config.embedding_size
        self.num_tags=Config.num_tags
        self.max_seq_length=Config.max_seq_length
        self.hidden_dim=Config.hidden_dim
        self.model_save_path=Config.model_save_path
        self.batch_size=Config.batch_size
           
    def placeholder_layer(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.max_seq_length])
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.max_seq_length])
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[None])
    
    def embedding_layer(self,embedding_matrix):
        embedding_matrix=tf.constant(embedding_matrix,dtype=tf.float32)
        assert embedding_matrix.shape==(self.embedding_size,self.embedding_dim)
        self.embeddings=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.word_ids)
        #self.embeddings.shape==(batch_size,max_seq_length,embedding_dim)
        
    def bilstm_layer(self):
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=self.embeddings,
                                                       sequence_length=self.seq_length,dtype=tf.float32)
        outputs_concat=tf.concat(values=[outputs[0],outputs[1]],axis=-1)
        assert outputs_concat.shape[-1]==2*self.hidden_dim
        self.lstm_out=tf.nn.dropout(outputs_concat,keep_prob=0.5)
    
    def affine_layer(self):
        weights=tf.Variable(tf.random_normal(shape=[2*self.hidden_dim,self.num_tags],dtype=tf.float32))
        biases=tf.Variable(tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        
        affine_input=tf.reshape(tensor=self.lstm_out,shape=[-1,self.hidden_dim*2])
        predicts=tf.matmul(affine_input,weights)+biases
        self.logits=tf.reshape(tensor=predicts,shape=[-1,self.max_seq_length,self.num_tags])
    
    def loss_layer(self):
        log_likelihood,self.transition_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,sequence_lengths=self.seq_length)
        self.loss=tf.reduce_mean(-log_likelihood)
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.loss)
    
    def build_graph(self,embedding_matrix):
        self.placeholder_layer()
        self.embedding_layer(embedding_matrix)
        self.bilstm_layer()
        self.affine_layer()
        self.loss_layer()

    def train(self,pad_word_id,pad_tag_id,actual_length):
        saver=tf.train.Saver()
        num_batches=len(pad_tag_id)//self.batch_size#num_batche means how many times will be executed in for step,(...) in enumerate(batches)
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(5):
                batches=batch_yield(pad_word_id,pad_tag_id,actual_length,self.batch_size)
                total_loss=0.0
                for step,(batch_x,batch_y,batch_length) in enumerate(batches):
                    feed_dict={self.word_ids:batch_x,self.label_ids:batch_y,self.seq_length:batch_length}
                    loss_val,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                    total_loss+=loss_val
                print("In epoch %d,loss value is %f " %(epoch,total_loss/num_batches))#total_loss/num_batches is average loss value
                saver.save(sess,self.model_save_path)
                
    def test(self,pad_word_id,pad_tag_id,actual_length):
        

if __name__ == "__main__":
    train_path=r'D:\NER\ner_assigment\NERs\data\train.txt'
    sentences,sentences_label=read_file(train_path)
    word2id,tag2id=get_word_tag2id(sentences,sentences_label)
    Config=Config_class(len(word2id),len(tag2id),r'D:\NER\ner_assigment\NERs\log\model.ckpt')
    glove_path=r'D:\NER\ner_data\glove\glove.6B.100d.txt'
    embedding_matrix=get_embedding(word2id,Config.embedding_dim,pre_trained_path=None)
    sentences_id,sentences_label_id=sentence_to_id(sentences,sentences_label,word2id,tag2id)
    pad_word_id,pad_tag_id,actual_length=pad_seq(sentences_id,sentences_label_id,max_seq_length=Config.max_seq_length)
    
    model=BiLSTM_CRF(Config=Config)
    model.build_graph(embedding_matrix)
    print("The graph has been built successfully!")
    model.train(pad_word_id,pad_tag_id,actual_length)
            
        