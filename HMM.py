import numpy as np


def read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    sentence,sentence_label=[],[]
    word_list=[]
    label_list=[]
    for line in lines:
        if len(line)==0 or line.startswith('-DOCSTART-') or line=="\n":
            if len(word_list)>0:
                sentence.append(word_list)
                sentence_label.append(label_list)
                word_list=[]
                label_list=[]
            continue
        line_split=line.strip().split()
        word=line_split[0]
        label=line_split[-1]
        word_list.append(word)
        label_list.append(label)
    if len(word_list)>0:
        sentence.append(word_list)
        sentence_label.append(label_list)
    return sentence,sentence_label

def get_word_tag_2id(sentence,sentence_label):
    word2id={}
    tag2id={}
    tag_set=set()
    all_words=[]
    for each_sentence,each_sentence_label in zip(sentence,sentence_label):
        assert len(each_sentence)==len(each_sentence_label)
        for word,tag in zip(each_sentence,each_sentence_label):
            tag_set.add(tag)
            all_words.append(word)
    import collections
    import operator
    for tag in tag_set:
        tag2id[tag]=len(tag2id)
    print(tag2id)
    count_words=collections.Counter(all_words)
    sorted_list=sorted(count_words.items(),key=operator.itemgetter(1),reverse=True)
    for word,frequency in sorted_list:
        word2id[word]=len(word2id)
    print("length of word2id : ",len(word2id))
    print(len(all_words))
    return word2id,tag2id
    
class HMM:
    def __init__(self,obser_nums,state_nums,word2id,tag2id):
        self.state_nums=state_nums
        self.obser_nums=obser_nums+1
        self.word2id=word2id
        self.word2id['UNK']=len(word2id)
        self.tag2id=tag2id
        self.A=np.zeros(shape=[state_nums,state_nums])
        self.pi=np.zeros(shape=(state_nums,))
        self.B=np.zeros(shape=[state_nums,self.obser_nums])
    
    def init_parameter(self,sentence,sentence_label,word2id,tag2id):
        for each_sentence,each_sentence_label in zip(sentence,sentence_label):
            sentence_length=len(each_sentence)
            for i in range(sentence_length-1):
                if i==0:
                    self.pi[tag2id[each_sentence_label[i]]]+=1
                current_state=tag2id[each_sentence_label[i]]
                next_state=tag2id[each_sentence_label[i+1]]
                self.A[current_state][next_state]+=1
                current_obser=word2id[each_sentence[i]]
                self.B[current_state][current_obser]+=1
            self.B[tag2id[each_sentence_label[sentence_length-1]]][word2id[each_sentence[sentence_length-1]]]+=1
        self.pi[self.pi==0.]=1e-4
        self.A[self.A==0.]=1e-4
        self.B[self.B==0.]=1e-4
        self.pi/=np.sum(self.pi,keepdims=True)
        self.A/=np.sum(self.A,axis=1,keepdims=True)
        self.B/=np.sum(self.B,axis=1,keepdims=True)
    
    def viterbi_decode(self,predict_sentence):
        T=len(predict_sentence)
        gamma=np.zeros(shape=[self.state_nums,T])
        delta=np.zeros(shape=[self.state_nums,T],dtype=np.int32)
        for i in range(self.state_nums):
            gamma[i][0]=self.pi[i]*self.B[i][self.word2id.get(predict_sentence[0],self.word2id['UNK'])]
        for t in range(1,T):
            for i in range(self.state_nums):
                temp=[]
                for j in range(self.state_nums):
                    temp.append(gamma[j][t-1]*self.A[j][i])
                gamma[i][t]=max(temp)*self.B[i][self.word2id.get(predict_sentence[t],self.word2id['UNK'])]
                delta[i][t]=temp.index(max(temp))
        
        lastest_column=[]
        for i in range(self.state_nums):
            lastest_column.append(gamma[i][T-1])
        best_path_prob=max(lastest_column)
        last_index=lastest_column.index(best_path_prob)
        best_path=[last_index]
        for t in range(T-1,0,-1):
            last_index=delta[last_index][t]
            best_path.append(last_index)
        best_path.reverse()
        return best_path
    
    def metric_calculation(self,test_sentence,test_sentence_label):
        
        TP_matrix=np.zeros(shape=(self.state_nums,))
        TP_FP_matrix=np.zeros(shape=(self.state_nums,))
        TP_FN_matrix=np.zeros(shape=(self.state_nums,))
        id2tag={ids:tag for tag,ids in self.tag2id.items()}
        print(id2tag)
        print(self.tag2id)
        for each_test_sentence,golden_sentence_label in zip(test_sentence,test_sentence_label):
            best_path=self.viterbi_decode(each_test_sentence)
            predict_tag_list=[id2tag[id_] for id_ in best_path]
            for each_predict_tag,each_golden_tag in zip(predict_tag_list,golden_sentence_label):
                if each_predict_tag==each_golden_tag:
                    TP_matrix[self.tag2id[each_predict_tag]]+=1
                TP_FP_matrix[self.tag2id[each_predict_tag]]+=1
                TP_FN_matrix[self.tag2id[each_golden_tag]]+=1
        #precision,recall,f1_score=np.zeros(shape=(self.state_nums,)),np.zeros(shape=(self.state_nums,)),np.zeros(shape=(self.state_nums,))
        result_dict={}
        for tag in self.tag2id.keys():
            result_dict[tag]={'precision':0.0,'recall':0.0,'f1_score':0.0}
        for i in range(self.state_nums):
            tag=id2tag[i]
            result_dict[tag]['precision']=TP_matrix[i]/TP_FP_matrix[i]
            result_dict[tag]['recall']=TP_matrix[i]/TP_FN_matrix[i]
            precision=result_dict[tag]['precision']
            recall=result_dict[tag]['recall']
            result_dict[tag]['f1_score']=2*precision*recall/(precision+recall)
        
        print(result_dict)      
        
        sum_f1_score=0.0
        for tag in self.tag2id.keys():
            sum_f1_score+=result_dict[tag]['f1_score']
        return sum_f1_score/self.state_nums
        

if __name__ == "__main__":
    train_file=r'D:\NER\ner_code\data\train.txt'
    test_file=r'D:\NER\ner_code\data\test.txt'
    sentence,sentence_label=read_file(train_file)
    word2id,tag2id=get_word_tag_2id(sentence,sentence_label)
    hmm=HMM(len(word2id),len(tag2id),word2id,tag2id)
    hmm.init_parameter(sentence,sentence_label,word2id,tag2id)
    
    my_id2tag={i:tag for tag,i in tag2id.items()}
    print(my_id2tag)
    my_test_sentence="China and American has a bad relationship Allen Iverson and Michael jordan are famous basketball".split()
    best_path_22=hmm.viterbi_decode(my_test_sentence)
    for word,id_predict in zip(my_test_sentence,best_path_22):
        print(word+"/"+my_id2tag[id_predict])
    
    test_sentence,test_sentence_label=read_file(test_file)
    f1_score=hmm.metric_calculation(test_sentence,test_sentence_label)
    print("-"*100)
    print(f1_score)
    