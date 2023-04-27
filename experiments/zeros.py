import random
from hyper_data_loader.HyperDataLoader import HyperDataLoader
import numpy
def random_gu(y,class_num):
    count=0
    correct=0
    for label in y:
        g=random.randint(1,class_num+1)
        if g==label:
            correct+=1
        count +=1
    return correct/count

def best_guess():
    pass
if __name__ == "__main__":
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables
    X, y = loader.filter_unlabeled(X, y)
    y=y.reshape(y.shape[0])
    unique, counts = numpy.unique(y, return_counts=True)

    #random=0.07787076865532074
    #best=counts.max()/len(y)#0.4359687675331962
    print(random_gu(y,10))