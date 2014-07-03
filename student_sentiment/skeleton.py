import sys, os
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR

vocab = [] #the features used in the classifier

#build vocabulary
def buildvocab():
    global vocab
    stopwords = open('stopwords.txt').read().lower().split()
    ###TODO: Populate vocab list with N most frequent words in training data, minus stopwords
    pos_dict = {}
    neg_dict = {}
    pos_dir = "/Users/muazzezmira/projects/sentiment/sentiment/student_sentiment/pos/"
    neg_dir = "/Users/muazzezmira/projects/sentiment/sentiment/student_sentiment/neg/"

    def build_posneg_dict(path_, dict_):

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

                        # pos[item] = pos.get(item, 0) +1
        return dict_

    pos_dict = build_posneg_dict(pos_dir, pos_dict)
    neg_dict = build_posneg_dict(neg_dir, neg_dict)
    #10,000 1000 and 100
    pos_class = []
    neg_class = []
    counter = 10001
    import operator
    pos_tuple = sorted(pos_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[0:10000]
    for item in pos_tuple:
        pos_class.append(item[0])
    return pos_class



def vectorize(fn):
    global vocab
    vector = np.zeros(len(vocab))

    ###TODO: Create vector representation of 

    return vector

def make_classifier():
   
    #TODO: Build X matrix of vector representations of review files, and y vector of labels

    lr = LR()
    lr.fit(X,y)

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
    buildvocab()
    # lr = make_classifier()
    # test_classifier(lr)
