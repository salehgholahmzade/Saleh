import speech_recognition as sr
from sklearn import svm
import pickle
from sklearn import mixture
import numpy as np
import os

###########################################################################

#Processing train voices and save the SVM model as a pickle file

a_mfcc = np.zeros
train_path = os.path.join(os.getcwd(), "my_voice")
for i in range(10):

    for j in range(2):
        print('i,j' , i ,j)
        a_mfcc = sr.feature_extraction(os.path.join(train_path, "train" + str(i) +str(j) +".wav"))
        a_mfcc_vector = np.reshape(a_mfcc, (1, a_mfcc.shape[0] * a_mfcc.shape[1]))
        #print("a_mfcc size = ", a_mfcc_vector.shape)
        if i == 0 and j == 0:
            train = a_mfcc_vector
            print('yes')
            continue
        train = np.concatenate((train, a_mfcc_vector), axis = 0)
        print("train size = ", train.shape)

y = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
clf = svm.SVC()
clf.fit(train, y)
print(clf.predict(train))
#gmm = mixture.GaussianMixture(n_components = 10, covariance_type = 'full').fit(train)

##########################################################################################

f = open('clf.pckl', 'wb')
g = open('train.pckl', 'wb')
pickle.dump(clf, f)
pickle.dump(train, g)
f.close()
g.close()
#


# for i in range(10):
#     test_path = os.path.join(os.getcwd(), "trains", str(i))
#     for j in range(8, 10):
#         a_mfcc = sr.feature_extraction(os.path.join(test_path, str(i) + "_jackson_" + str(j) + ".wav"))
#         a_mfcc_vector = np.reshape(a_mfcc, (1, a_mfcc.shape[0] * a_mfcc.shape[1]))
#         #print("a_mfcc size = ", a_mfcc_vector.shape)
#         if i == 0 and j == 8:
#             test = a_mfcc_vector
#             print("test size = ", test.shape)
#             continue
#         test = np.concatenate((test, a_mfcc_vector), axis = 0)
#         print("test size = ", test.shape)


#test_mfcc = sr.feature_extraction("test1.wav")
#test = np.reshape(test_mfcc , (1,test_mfcc.shape[0]*test_mfcc.shape[1]))


# f = open('gmm.pckl', 'rb')
# g = open('train.pckl', 'rb')
# train = pickle.load(g)
# gmm = pickle.load(f)
# f.close()
# g.close()


#print(gmm.predict(train))
#print(gmm.predict(test))

