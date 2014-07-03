import sys, os
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR
import random

vocab = [] #the features used in the classifier
pos_dir = "/Users/muazzezmira/projects/sentiment/sentiment/student_sentiment/pos/"
neg_dir = "/Users/muazzezmira/projects/sentiment/sentiment/student_sentiment/neg/"

def build_posneg_dict(path_, dict_):
    """
    Counts the appaerances of every words in
    specified path and returns a dictionary
    """
    stopwords = open('stopwords.txt').read().lower().split()
    for  filename in os.walk(path_):
        for item in filename[2]:
            if not item.endswith('.txt'):
                continue
            # import pdb; pdb.set_trace()
            file_name_path = path_ + item
            with open(file_name_path) as f:
                for line in f:
                    new_text = line.split()
                    for item_ in new_text:
                        if item_ in stopwords or len(item_) < 2:
                            continue
                        dict_[item_] = dict_.get(item_, 0) + 1


def buildvocab(pos=True):
    """
    returns 10,000 size vocablary classifier with chosen most frequent words in both
    positive and negative reviews
    """
    global vocab
    ###TODO: Populate vocab list with N most frequent words in training data, minus stopwords
    vocab_dict = {}
    build_posneg_dict(pos_dir, vocab_dict)
    build_posneg_dict(neg_dir, vocab_dict)
    import operator
    vocab_tuple = sorted(vocab_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[0:10000]
    for item in vocab_tuple:
        vocab.append(item[0])
    print '-----'

    # pos_dict = build_posneg_dict(pos_dir, pos_dict)
    # neg_dict = build_posneg_dict(neg_dir, neg_dict)
    # pos_class = []
    # neg_class = []
    # counter = 10001
    # import operator
    # pos_tuple = sorted(pos_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[0:10000]
    # neg_tuple = sorted(neg_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[0:10000]
    # for item in pos_tuple:
    #     pos_class.append(item[0])
    # for item in neg_tuple:
    #     neg_class.append(item[0])
    # return vocab_class


def vectorize(fn):
    global vocab
    vector = np.zeros(len(vocab))

    ###TODO: Create vector representation of 

    return vector


def create_lists(label_list, matrix_list):
    for  filename_pos in os.walk(pos_dir):
        file_list_pos = filename_pos[2]
    for  filename_neg in os.walk(neg_dir):
        file_list_neg = filename_neg[2]

    while  (len(file_list_pos) + len(file_list_neg) ) > 0:
        end_pos = len(file_list_pos)
        end_neg = len(file_list_neg)
        # print "------> pos", end_pos
        # print "------> neg", end_neg
        if end_pos > 0 and end_neg > 0 :
            my_type = random.choice([1,-1])
        elif end_neg == 0 and end_pos == 0:
            print "no review files left"
            return
        elif not end_pos:
            my_type = -1
        else:
            my_type = 1

        if my_type == 1:
            end = end_pos
            file_ = file_list_pos
            file_path = pos_dir
        else :
            end = end_neg
            file_ = file_list_neg
            file_path = neg_dir

        my_index = random.randrange(0, end)
        label_list.append(my_type)
        file_name = file_[my_index]

        if file_ == file_list_pos:
            file_list_pos.pop(my_index)
        else:
            file_list_neg.pop(my_index)

        # file_[my_index] = file_.pop()

        my_file = file_path+file_name
        with open(my_file) as f:
            file_text = f.read().split(' ')
            file_text_vector = []
            for word in vocab:
                file_text_vector.append(file_text.count(word))
        matrix_list.append(file_text_vector)

    import numpy as np
    matrix_list = np.matrix([ i for i in matrix_list])
    file_text_vector = np.matrix([file_text_vector])

    return matrix_list

def make_classifier():
    #TODO: Build X matrix of vector representations of review files, and y vector of labels
    label_list = []
    matrix_list = []
    matrix_list= create_lists(label_list, matrix_list)
    print "----------->"
    print label_list
    print "----------->"
    print matrix_list
    print "----------->"
    lr = LR()
    lr.fit(matrix_list,label_list)

    return lr

def test_classifier(lr):
    global vocab
    test = np.zeros((len(os.listdir('test')),len(vocab)))
    testfn = []
    i = 0
    y = []
    for fn in os.listdir('test'):
        testfn.append(fn)
        test[i] = vectorize(os.path.join('test',fn))
        ind = int(fn.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y)==0)
    p = lr.predict(test)

    r,w = 0,0
    for i,x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w +=1
            print(testfn[i])
    print(r,w)


if __name__=='__main__':
    # import pdb; pdb.set_trace()
    buildvocab()

    lr = make_classifier()

    # test_classifier(lr)
