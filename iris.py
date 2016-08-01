from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

filename = 'bezdekIris.csv'
features = []
labels = []

f = open(filename,'rw')
for line in f:
    linelist = line.strip().split(',')
    if len(linelist)==5:
        features.append([float(x) for x in linelist[:4]])
        labels.append(linelist[-1])

TEST_FRAC = 0.5
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = TEST_FRAC)

classifier1 = SVC()
parameters1 = {'kernel':('linear','rbf','sigmoid','poly'), 'C':(.01,.1,1,10,100),'gamma':(.01,.1,1,10,100)}
clf1 = GridSearchCV(classifier1,parameters1)
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
print "SVM accuracy and top parameters: "
print accuracy_score(pred1,labels_test)
print clf1.best_params_
print

classifier2 = RandomForestClassifier()
parameters2 = {'criterion':('gini','entropy'),'min_samples_split':(2,5,10,20),'n_estimators':(5,10,25)}
clf2 = GridSearchCV(classifier2,parameters2)
clf2.fit(features_train,labels_train)
pred2 = clf2.predict(features_test)
print "Random Forest accuracy and top parameters: "
print accuracy_score(pred2,labels_test)
print clf2.best_params_
print
