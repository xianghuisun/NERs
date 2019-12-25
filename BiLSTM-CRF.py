import tensorflow as tf
from process_data import *
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True

class Config_class:
    def __init__(self,vocab_size,num_tags,model_save_path,tag2id):
        self.embedding_dim=100
        self.max_seq_length=125
        self.embedding_size=vocab_size
        self.hidden_dim=128
        self.num_tags=num_tags
        self.model_save_path=model_save_path
        self.batch_size=64
        self.use_CRF=True
        self.tag2id=tag2id
        

class BiLSTM_CRF:
    def __init__(self,Config):
        self.embedding_dim=Config.embedding_dim
        self.embedding_size=Config.embedding_size
        self.num_tags=Config.num_tags
        self.max_seq_length=Config.max_seq_length
        self.hidden_dim=Config.hidden_dim
        self.model_save_path=Config.model_save_path
        self.batch_size=Config.batch_size
        self.use_CRF=Config.use_CRF
        self.tag2id=Config.tag2id
        tf.reset_default_graph()
        
           
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
        if self.use_CRF:
            log_likelihood,self.transition_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,sequence_lengths=self.seq_length)
            self.loss=tf.reduce_mean(-log_likelihood)
        else:
            print('Do not use CRF ')
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.loss)
    
    def build_graph(self,embedding_matrix):
        self.placeholder_layer()
        self.embedding_layer(embedding_matrix)
        self.bilstm_layer()
        self.affine_layer()
        self.loss_layer()
        #log_likelihood,self.transition_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,sequence_lengths=self.seq_length)
        #self.decode_tags,best_score=tf.contrib.crf.crf_decode(potentials=self.logits,transition_params=self.transition_matrix,sequence_length=self.seq_length)
            

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
    
    def test_viterbi_decode(self,pad_word_id,pad_tag_id,actual_length):
        saver=tf.train.Saver()
        TP_matrix=np.zeros(shape=(self.num_tags,))
        TP_FP_matrix=np.zeros(shape=(self.num_tags,))
        TP_FN_matrix=np.zeros(shape=(self.num_tags,))    
        with tf.Session(config=gpu_config) as sess:
            saver.restore(sess,self.model_save_path)
            batches=batch_yield(pad_word_id,pad_tag_id,actual_length,self.batch_size)
            total_correct=0
            total=0     
            for step,(batch_x,batch_y,batch_length) in enumerate(batches):
                feed_dict={self.word_ids:batch_x,self.seq_length:batch_length}           
                predict_logits,transition_matrix_=sess.run([self.logits,self.transition_matrix],feed_dict=feed_dict)
                #predict_logits.shape==(batch_size,max_seq_len,num_tags)
                for predict_score,actual_length_,golden_score in zip(predict_logits,batch_length,batch_y):
                    assert predict_score.shape==(self.max_seq_length,self.num_tags)
                    predict_score=predict_score[:actual_length_]
                    viterbi_seq,viterbi_score=tf.contrib.crf.viterbi_decode(predict_score,transition_matrix_)#返回的是一个列表和一个float32类型的分数
                    assert type(viterbi_seq)==list and type(viterbi_score)==np.float32
                    golden_score=list(golden_score[:actual_length_])
                    assert len(golden_score)==len(viterbi_seq)
                    for predict_id,golden_id in zip(viterbi_seq,golden_score):
                        if predict_id==golden_id:
                            total_correct+=1
                            TP_matrix[predict_id]+=1
                        TP_FP_matrix[predict_id]+=1
                        TP_FN_matrix[golden_id]+=1
                        total+=1
            id2tag={key_:value_ for value_,key_ in self.tag2id.items()}
            print(id2tag)        
            result={}
            for tag in self.tag2id.keys():
                result[tag]={"precision":0.0,"recall":0.0,"f1_score":0.0}
            f1_score_avg=0.0
            for i in range(self.num_tags):
                tag=id2tag[i]
                result[tag]["precision"]=TP_matrix[i]/TP_FP_matrix[i]
                result[tag]["recall"]=TP_matrix[i]/TP_FN_matrix[i]
                
                result[tag]['f1_score']=2*result[tag]["precision"]*result[tag]["recall"]/(result[tag]["precision"]+result[tag]["recall"])
                f1_score_avg+=result[tag]['f1_score']
            print("total_correct/total is ",total_correct/total)
            print("average f1 score is ",f1_score_avg/self.num_tags)  
            
                    
                    
          
    def test_crf_decode(self,pad_word_id,pad_tag_id,actual_length):
        saver=tf.train.Saver()
        TP_matrix=np.zeros(shape=(self.num_tags,))
        TP_FP_matrix=np.zeros(shape=(self.num_tags,))
        TP_FN_matrix=np.zeros(shape=(self.num_tags,))
        decode_tags,best_score=tf.contrib.crf.crf_decode(potentials=self.logits,transition_params=self.transition_matrix,sequence_length=self.seq_length)
        if self.use_CRF:
            with tf.Session(config=gpu_config) as sess:
                saver.restore(sess,self.model_save_path)
                batches=batch_yield(pad_word_id,pad_tag_id,actual_length,self.batch_size)
                total_correct=0
                total=0
                for step,(batch_x,batch_y,batch_length) in enumerate(batches):
                    feed_dict={self.word_ids:batch_x,self.seq_length:batch_length}
                    predict_matrix=sess.run(decode_tags,feed_dict=feed_dict)
                    assert predict_matrix.shape==(self.batch_size,self.max_seq_length)==batch_y.shape
                    for each_predict_seq,each_golden_seq,each_length in zip(predict_matrix,batch_y,batch_length):
                        each_predict_seq=each_predict_seq[:each_length]
                        each_golden_seq=each_golden_seq[:each_length]
                        for predict_id,golden_id in zip(each_predict_seq,each_golden_seq):
                            if predict_id==golden_id:
                                total_correct+=1
                                TP_matrix[predict_id]+=1
                            TP_FP_matrix[predict_id]+=1#FP是False Positive，假正类，就是把不是当前标签的标签预测成当前的标签
                            TP_FN_matrix[golden_id]+=1
                            total+=1
                        #each_golden_seq,each_predict_seq分别是真实的标签序列和预测的标签序列，每一个值都是int32类型，代表标签在tag2id中的id
                
                id2tag={key_:value_ for value_,key_ in self.tag2id.items()}
                print(id2tag)        
                result={}
                for tag in self.tag2id.keys():
                    result[tag]={"precision":0.0,"recall":0.0,"f1_score":0.0}
                f1_score_avg=0.0
                for i in range(self.num_tags):
                    tag=id2tag[i]
                    result[tag]["precision"]=TP_matrix[i]/TP_FP_matrix[i]
                    result[tag]["recall"]=TP_matrix[i]/TP_FN_matrix[i]
                    
                    result[tag]['f1_score']=2*result[tag]["precision"]*result[tag]["recall"]/(result[tag]["precision"]+result[tag]["recall"])
                    f1_score_avg+=result[tag]['f1_score']
                print("total_correct/total is ",total_correct/total)
                print("average f1 score is ",f1_score_avg/self.num_tags)
    

    


    
def train_model(train_path):
    #train_path=r'D:\NER\ner_assigment\NERs\data\train.txt'
    sentences,sentences_label=read_file(train_path)
    word2id,tag2id=get_word_tag2id(sentences,sentences_label)
    parameter_=(word2id,tag2id)
    with open(r'D:\NER\ner_assigment\NERs\data\word_tag2id.pkl','wb') as f:
        pickle.dump(parameter_,f)
    Config=Config_class(len(word2id),len(tag2id),r'D:\NER\ner_assigment\NERs\log\model.ckpt',tag2id)
    glove_path=r'D:\NER\ner_data\glove\glove.6B.100d.txt'
    embedding_matrix=get_embedding(word2id,Config.embedding_dim,pre_trained_path=None)
    with open(r'D:\NER\ner_assigment\NERs\data\embedding_matrix.pkl','wb') as f:
        pickle.dump(embedding_matrix,f)
        
    sentences_id,sentences_label_id=sentence_to_id(sentences,sentences_label,word2id,tag2id)
    pad_word_id,pad_tag_id,actual_length=pad_seq(sentences_id,sentences_label_id,max_seq_length=Config.max_seq_length)
    
    model=BiLSTM_CRF(Config=Config)
    model.build_graph(embedding_matrix)
    print("The graph has been built successfully!")
    model.train(pad_word_id,pad_tag_id,actual_length)
    
                   
def test_model(test_path,embedding_matrix):
    sentences,sentences_label=read_file(test_path)
    with open(r'D:\NER\ner_assigment\NERs\data\word_tag2id.pkl','rb') as f:
        word2id,tag2id=pickle.load(f)
    Config=Config_class(len(word2id),len(tag2id),r'D:\NER\ner_assigment\NERs\log\model.ckpt',tag2id)
    glove_path=r'D:\NER\ner_data\glove\glove.6B.100d.txt'
    sentences_id,sentences_label_id=sentence_to_id(sentences,sentences_label,word2id,tag2id)
    pad_word_id,pad_tag_id,actual_length=pad_seq(sentences_id,sentences_label_id,max_seq_length=Config.max_seq_length)
    model=BiLSTM_CRF(Config=Config)
    model.build_graph(embedding_matrix)
    model.test_viterbi_decode(pad_word_id,pad_tag_id,actual_length)   
    print("*/"*100)
    model.test_crf_decode(pad_word_id,pad_tag_id,actual_length)       

if __name__ == "__main__":
    train_path=r'D:\NER\ner_assigment\NERs\data\train.txt'
    test_path=r'D:\NER\ner_assigment\NERs\data\test.txt'
    #train_model(train_path)
    with open(r'D:\NER\ner_assigment\NERs\data\embedding_matrix.pkl','rb') as f:
        embedding_matrix=pickle.load(f)
    print('-'*1000)
    test_model(test_path,embedding_matrix)
            
        