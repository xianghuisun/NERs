import numpy as np

def get_parameter2id(sentences, sentences_label):
    word2id, tag2id, char2id = {}, {}, {}
    all_words = []
    tag_set = set()
    for word_list, label_list in zip(sentences, sentences_label):
        for word, tag in zip(word_list, label_list):
            all_words.append(word)
            tag_set.add(tag)
            for char in word:
                if char not in char2id:
                    char2id[char] = len(char2id)
    for tag in tag_set:
        tag2id[tag] = len(tag2id)
    print(tag2id)
    import collections
    import operator
    counter_words = collections.Counter(all_words)
    sort_words = sorted(counter_words.items(), key=operator.itemgetter(1), reverse=True)
    for word, freq in sort_words:
        word2id[word] = len(word2id)
    word2id['UNK'] = len(word2id)
    char2id['UNK'] = len(char2id)
    return word2id, tag2id, char2id


def sentence_to_id(sentences, sentences_label, word2id, tag2id, char2id):
    word_sens_id = []
    tag_sens_id = []
    char_sens_id = []
    for word_list, tag_list in zip(sentences, sentences_label):
        word_id_list = []
        tag_id_list = []
        char_sen_id = []
        for word, tag in zip(word_list, tag_list):
            if word in word2id:
                word_id = word2id[word]
            else:
                assert word not in word2id.keys()
                word_id = word2id['UNK']  # 因为在测试集会出现没有见过的单词
            tag_id = tag2id[tag]
            word_id_list.append(word_id)
            tag_id_list.append(tag_id)

            char_id_list = []
            for char in word:
                if char in char2id:
                    char_id = char2id[char]
                else:
                    assert char not in char2id.keys()
                    char_id = char2id['UNK']
                char_id_list.append(char_id)  # char_id_list里面的是一个单词的所有字符的id表示
            char_sen_id.append(char_id_list)  # char_sen_id是一个句子的所有单词的所有字符的id表示

        char_sens_id.append(
            char_sen_id)  # char_sens_id is a three-dimension list, each element is a two-dimension list means a sentence
        # 最里面的列表就是字符级别的单词
        # ['EU','rejects','German','call','to','boycott','British','lamb','.']
        # 那么早char_sens_id中就是下面那种形式
        word_sens_id.append(word_id_list)
        tag_sens_id.append(tag_id_list)
    return word_sens_id, tag_sens_id, char_sens_id


'''
[[0, 1],                                                        
 [2, 3, 4, 3, 5, 6, 7],                                                     
 [8, 3, 2, 9, 10, 11],
 [5, 10, 12, 12],
 [6, 13],
 [14, 13, 15, 5, 13, 6, 6],
 [16, 2, 17, 6, 17, 7, 18],
 [12, 10, 9, 14],
 [19]]

'''


# Suppose that max_seq_len=125 and max_word_len=20
def pad_data(word_sens_id, tag_sens_id, char_sens_id, max_seq_len, max_word_len):
    pad_word, pad_tag = [], []
    actual_length = []
    for word_id_list, tag_id_list in zip(word_sens_id, tag_sens_id):
        length = len(word_id_list)
        assert length == len(tag_id_list)
        if length >= max_seq_len:
            actual_length.append(max_seq_len)
            pad_word.append(word_id_list[:max_seq_len])
            pad_tag.append(tag_id_list[:max_seq_len])
        else:
            actual_length.append(length)
            pad_word.append(word_id_list[:max_seq_len] + [0] * (max_seq_len - length))
            pad_tag.append(tag_id_list[:max_seq_len] + [0] * (max_seq_len - length))
    assert len(char_sens_id) == len(word_sens_id)
    pad_char = np.empty(shape=[len(char_sens_id), max_seq_len, max_word_len], dtype=np.int32)
    for i in range(len(char_sens_id)):
        current_seq = char_sens_id[i]
        current_seq_len = len(current_seq)
        for j in range(min(current_seq_len, max_seq_len)):
            current_word = current_seq[j]
            current_word_len = len(current_word)
            for k in range(min(current_word_len, max_word_len)):
                char_id = current_word[k]
                pad_char[i][j][k] = char_id
            pad_char[i][j][min(current_word_len, max_word_len):] = 0
        pad_char[i][min(current_seq_len, max_seq_len):][:] = 0
    return np.array(pad_word), np.array(pad_tag), pad_char, np.array(actual_length)


def batch_generate(pad_word, pad_tag, pad_char, actual_length, batch_size):
    total_length = pad_word.shape[0]
    assert total_length == pad_char.shape[0]
    shuffled = np.random.permutation(total_length)
    pad_word = pad_word[shuffled]
    pad_char = pad_char[shuffled]
    pad_tag = pad_tag[shuffled]
    actual_length = actual_length[shuffled]
    num_batches = total_length // batch_size
    start = 0
    for i in range(num_batches):
        yield pad_word[start:start + batch_size], pad_tag[start:start + batch_size], pad_char[
                                                                                     start:start + batch_size], actual_length[
                                                                                                                start:start + batch_size]
        start += batch_size


def get_embedding_matrix(word2id, char2id,embedding_dim=100, char_embedding_dim=30, glove_path=None):
    word_matrix = []
    char_matrix = []
    if glove_path == None:
        word_matrix = np.random.uniform(-1.0, 1.0, size=(len(word2id), embedding_dim))
    else:
        with open(glove_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        words_list = [word for word, _ in word2id.items()]
        glove_dict = {}
        embedding_matrix = []
        for line in lines:
            line_split = line.strip().split()
            word = line_split[0]
            vector_string_list = line_split[1:]
            glove_dict[word] = vector_string_list
        for word in words_list:
            if word in glove_dict:
                vector = [float(number) for number in glove_dict[word]]
                word_matrix.append(vector)
            else:
                word_matrix.append(np.random.uniform(-1.0, 1.0, size=(100,)))
    word_matrix = np.array(word_matrix)
    char_matrix = np.random.uniform(-3.0 / np.sqrt(char_embedding_dim), 3.0 / np.sqrt(char_embedding_dim),
                                    size=(len(char2id), char_embedding_dim))
    return word_matrix, char_matrix