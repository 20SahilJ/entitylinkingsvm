#all import statements
import numpy as np
import matplotlib.pylab as pl
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


#Note: A entity/mention set is defined to be the statistics derived from a single line in newdata.txt

#get num_lines, which indicates how many entitity/mention sets there are total
num_lines = sum(1 for line in open('newdata.txt'))
#read in X data, which is in the file newdata.txt
data = pd.read_csv('newdata.txt', sep = '|', header = None)
#gets a list of the entities, which are located in the first column of the data table
entities = np.array(data)[:,0]
#The following line and for loop is used to get the amount of words in the entity with the most amount of words.
# This is needed in order to make each vectorizer fit have the same array size, which in this case, will be the value
# obtained from the for loop.
maxentitylen = 0
for entity in entities:
    entityarray = entity.split(" ")
    entitylen = len(entityarray)
    if entitylen>maxentitylen:
        maxentitylen = entitylen
#gets all the entity mentions, which are located in the data columns except for the first column, which contains
# the entities
entitymentionstotal = data.iloc[:,data.columns!=0]
#gets number of mentions, which is simply the amount of data columns minus one(since one of the columns is the entities)
num_mentions = len(data.columns)-1

#full list of all entitymention data i.e. all of the X data
w, h, q = maxentitylen, num_mentions, num_lines
fullentitymentiondata = [[[0 for x in range(w)] for y in range(h)]for z in range(q)]
# Vectorize the different entity mentions. Entity is first vectorized, and mentions are vectorized corresponding to the
# fit of the entity vectorization.
for i in range (0,num_lines):
    entity = data.iloc[:, 0].get(i)
    entity = entity.split(".")  # used to put entity into an array without seperating it
    entitymentions = entitymentionstotal.iloc[i]
    vectorizer = TfidfVectorizer()
    entityfit = vectorizer.fit_transform(entity)
    entitymentionfit = vectorizer.transform(entitymentions)
    entitymentiondatatfidfnparray = np.array(entitymentionfit.toarray())
    # A matrix of zeros is appended to keep size of x data consistent
    zeroscols = np.zeros((num_mentions, maxentitylen - entitymentiondatatfidfnparray.shape[1]))
    entitymentiondatatfidfnparray = np.append(entitymentiondatatfidfnparray, zeroscols, 1)
    # the matrix created is saved back into the 3d X data matrix
    fullentitymentiondata[i] = entitymentiondatatfidfnparray

#reads in all y data
y = pd.read_csv('newresults.txt', sep = '|', header = None).values
#indices (in range of the number of entities) for future use
indices = range(num_lines)

#Train and test data is split. This includes X and y data and the indices. Indices are there to indicate which X/y data
# pieces were split into training and testing data
X_train, X_test, y_train, y_test,indices_train, indices_test = train_test_split(fullentitymentiondata, y, indices,
                                                                                test_size=0.2, random_state=42)

#The SVM classifier is created with a linear kernel
clf = svm.SVC(kernel='linear', C=0.1, probability=True)
#In order to train the classifier, all data must be in 1d or 2d format. Therefore, the y training data is flattened from
#  2d to 1d, and the x training data is flattened from 3d to 2d.
y_trainfinal = np.array(y_train).ravel()
newX_traindimension = len(X_train)*num_mentions
X_trainfinal = np.array(X_train).reshape(newX_traindimension,maxentitylen)
#Training data is fit to classifier
clf.fit(X_trainfinal, y_trainfinal)
#From this fit, a coefficient array is extracted which is used in determining the probability that a mention relates to
#  the corresponding entity
coef = clf.coef_.ravel() / pl.linalg.norm(clf.coef_)
#the following represents the creation of the arrays for the storage of the accuracies, precisions, f1 scores, and
# recall scores of the different entity/mention sets
'''
accuracies = [0 for x in range(len(X_test))]
precisions = [0 for x in range(len(X_test))]
f1s = [0 for x in range(len(X_test))]
recalls = [0 for x in range(len(X_test))]
'''
y_trues, y_preds = [], []
#for loop used to perform predictions and metrics. Note that unlike the training data, the X testing data is not
# flattened into one 2d array(likewise, the y testing data is not flattened into a 1d array). This is done so that
# individual entity/mention sets could be analyzed.
for j in range(len(X_test)):
    #get x and y test data for current index
    X_testnew = X_test[j]
    y_testnew = y_test[j]

    #following 4 lines represents trying to predict probability with the classifier. Directly predicting probability
    # didn't work as well as the current method
    #probabilitypreds = clf.predict_proba(X_testnew)
    #first = np.array(probabilitypreds)[:,0]
    #second = np.array(probabilitypreds)[:,1]
    #print(np.hypot(first, second))

    #array to store all the probability scores.
    npmeasures = [0 for x in range(num_mentions)]
    #For loop used to calculate probability scores. Note that this for loop calculates the scores pre-scaling. That is,
    #  these scores are not yet in the range 0-1. For more information about how the probability scores are calculated,
    # please look at the comment down below.
    for i in range(num_mentions):
        npval = np.dot(X_testnew[i],coef)
        if npval == 1:
            npmeasure = 1-npval
        else:
            npmeasure = 1/(1-npval)
        npmeasures[i] = npmeasure
    #Gets the maximum value (and its index in the probability score array) of all the probability scores. This is used
    # to scale the probability scores into the range 0-1.
    maxval = np.amax(npmeasures)
    maxindex = npmeasures.index(maxval)
    #Sets up a variable that keeps track of the last index of a 1 in the y data. This is used for the different metrics,
    #  as the metrics are only calculated on elements that are supposed to be 1.
    onesindex = 0
    for i in range(num_mentions):
        print(X_testnew[i])
        print(y_testnew[i])
        # tau, _ = stats.kendalltau(npmeasure, y_testnew[i])
        npmeasures[i]=npmeasures[i]/maxval
        #note that the probability score is calculated by taking dot product of both the coefficents determined by the fit
        # and the given values of the testing data. The npmeasure value is calculated by the formula 1/(1-dotprodvalue)
        # (in the case that the dotprodvalue is 1, 0 is put as this npmeasure value
        #  Finally, the npmeasure value is divided by the maximum value to give a range of 0 - 1 for the probability
        # score.
        print('Probability Score for mention %s: %.5f' % (i, npmeasures[i]))
        #Continually adds one to the onesindex while the current value in the y data is a 1.
        if y_testnew[i]==1:
            onesindex = onesindex + 1
    #Prints a message displaying the predicted mention for the current entity
    dataindex = indices_test[j]
    print('Predicted mention for entity ' + data.iloc[dataindex,0] + ' is '+ data.iloc[dataindex,maxindex+1])
    #Calculates and prints y_true and y_pred for the metrics. As stated above, these metrics are only calculated on the
    # sections of the predicted and true data that are supposed to be 1s.
    y_true = y_testnew[:onesindex]
    y_pred = npmeasures[:onesindex]
    y_trues = y_trues + y_true.tolist()
    #Use following lines to set rounding threshold
    threshold = 0.9
    y_pred = np.array(y_pred)
    y_pred[y_pred>=threshold] = 1
    y_pred[y_pred<threshold] = 0
    y_preds = y_preds + y_pred.tolist()
    print('Y_true', y_true)
    print('Y_pred', y_pred)
    '''
    #calculates all the metrics and stores them into an array
    accuracy = metrics.accuracy_score(np.array(y_true), y_pred)
    accuracies[j] = accuracy
    print('Accuracy: ', accuracy)
    precision = metrics.precision_score(np.array(y_true), y_pred)
    precisions[j] = precision
    print('Precision: ', precision)
    f1 = metrics.f1_score(np.array(y_true),y_pred)
    f1s[j] = f1
    print('F1 Score: ', f1)
    recall = metrics.recall_score(np.array(y_true), y_pred)
    recalls[j] = recall
    print('Recall: ', recall)
    '''
#Calculates the a mean for each metric and prints them out
averageaccuracy = metrics.accuracy_score(np.array(y_trues), np.array(y_preds))
print('Average Accuracy', averageaccuracy)
averageprecision = metrics.precision_score(np.array(y_trues), np.array(y_preds))
print('Average Precision', averageprecision)
averagef1 = metrics.f1_score(np.array(y_trues), np.array(y_preds))
print('Average F1 Score', averagef1)
averagerecall = metrics.recall_score(np.array(y_trues), np.array(y_preds))
print('Average Recall', averagerecall)