# Data Mining, Project2 - Classification Using Decision Tree
# Chun-Yii Liu, N16064103, 2018/11/17
# Visualizing Decision Trees: https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def loadData(filename = 'YellowLight.csv'):
	dataSet = pd.read_csv(filename)
	#print(dataSet)

	ftr = dataSet.values[:,0:3]
	lbl = dataSet.values[:,7]	# Get label: [1 1 0 1 ...]
	ftr = ftr.tolist()
	lbl = lbl.tolist()	# Make it [1,1,0,1, ...]

	for i in range(0, len(ftr)):
		if dataSet['Where from (_ TW)'][i] == 'South':
			ftr[i].append(0)
		elif dataSet['Where from (_ TW)'][i] == 'Middle':
			ftr[i].append(1)
		elif dataSet['Where from (_ TW)'][i] == 'North':
			ftr[i].append(2)
		elif dataSet['Where from (_ TW)'][i] == 'East':
			ftr[i].append(3)

		if dataSet['Gender'][i] == 'M':
			ftr[i].append(1)
		elif dataSet['Gender'][i] == 'F':
			ftr[i].append(0)

		if dataSet['Have backseat passenger'][i] == True:
			ftr[i].append(1)
		elif dataSet['Have backseat passenger'][i] == False:
			ftr[i].append(0)

		if dataSet['Wearing headphone'][i] == True:
			ftr[i].append(1)
		elif dataSet['Wearing headphone'][i] == False:
			ftr[i].append(0)

	#print(ftr)
	return(ftr,lbl)

#-----------------------------------------------------------------------------

# Load dataset
FTR, LBL = loadData()

# Split the training set and test set from dataset
Train_x, Test_x, Train_y, Test_y = train_test_split(FTR, LBL, test_size=0.2, random_state=42)	# Use 'random_state =' to replicate results

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)	# Try max_depth = , min_samples_leaf = 
clf_gini.fit(Train_x, Train_y)
gini_score = accuracy_score(Test_y,clf_gini.predict(Test_x))

clf_entr = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf_entr.fit(Train_x, Train_y)
entr_score = accuracy_score(Test_y,clf_entr.predict(Test_x))

print('Score of GINI =',gini_score,'\nScore of Entropy =',entr_score)

# Accuracy of Training Set
# gini_score = accuracy_score(Train_y,clf_gini.predict(Train_x))
# entr_score = accuracy_score(Train_y,clf_entr.predict(Train_x))
# print('Score of GINI (train) =',gini_score,'\nScore of Entropy (train) =',entr_score)

# Try predicting label of given features
# [Speed, Distance, YearsOfRiding, From(0:S/1:M/2:N/3:E), Sex(1:M/0:F), BackSeat, HeadPhone]
# clf_gini.predict([[65,10,3,0,0,1,0]])
# clf_entr.predict([[65,10,3,0,0,1,0]])

# Plot the decision tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus # Install to use

dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data,	# Change 'clf_gini' to 'clf_entr' for Decision tree of Entropy plotting
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())