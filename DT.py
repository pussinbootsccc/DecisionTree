from DataHandler import DataHandler
from math import log
from collections import Counter
import random

def log2(x):
	return log(x)/log(2)

class DTNode:
    def __init__(self, rule): # initialize the node with a rule
        self.rule = rule
        self.left = None
        self.right = None
        self.isLeaf = False # whether it is leaf node
        self.label = 'NA'

class Rule:
	def __init__(self, attributeId, isNumeral, criterion):
		self.attributeId = attributeId # 0 to 5
		self.isNumeral = isNumeral # whether attr is numeral
		self.criterion = criterion # e.g. "enginerr" or 0.2512

		# define the split function
		if not self.isNumeral:
			self.belongToLeft = lambda x:x[0][attributeId] == criterion
		else:
			self.belongToLeft = lambda x:x[0][attributeId] <= criterion

	def split(self, X):
		Xl, Xr = [], []
		for x in X:
			if self.belongToLeft(x):
				Xl.append(x)
			else:
				Xr.append(x)
		return Xl, Xr

class DT:
	def __init__(self, numFeature, choiceBox):
		self.numFeature = numFeature
		self.choiceBox = choiceBox # dict, attributeId:choices

	def computeEntropy(self, data):
		labels = map(lambda x:x[1], data)
		n = len(labels)
		cnt = Counter(labels) # recored counts of each label 
		frequency = map(lambda x: 1.0*x/n, cnt.values()) # list holds each label's frequency(percentage in total counts)
		entropy = sum(map(lambda p:-1.0*p*log2(p), frequency)) # for each frequency, generate -plog2p, get sum of them
		return entropy

	def computeInfoGain(self, X, Xl, Xr):
		Ep = self.computeEntropy(X) # parent entropy
		El = self.computeEntropy(Xl) # left child entropy
		Er = self.computeEntropy(Xr) # right child entropy
		nl = 1.0*len(Xl) # num of data in left
		nr = 1.0*len(Xr) # num of data in right
		n = nl + nr # total num of data
		G = Ep - (El*nl/n + Er*nr/n) # entropy * percentage
		return G

	def __buildTree(self, X):

		max_gain = -1.
		max_rule = None

		# enumerate over all possible attributes
		for i in xrange(self.numFeature): # i in 0-5, rule out label row 6:[c1,c2,c3,c4,c5] in choicebox
			if i in self.choiceBox: # if ith feature is not numeral choiceBox={0:..., 1:...}
				isReal = False
				choices = self.choiceBox[i] # ["engineer","student"...]
			else:
				isReal = True
				choices = map(lambda x:float(x)/10, range(11)) # split criterion [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

			# enumerate over all choices for the given attribute
			for c in choices:
				rule = Rule(i, isReal, c)
				Xl, Xr = rule.split(X)
				gain = self.computeInfoGain(X, Xl, Xr)
				if gain > max_gain:
					max_gain, max_rule = gain, rule
        
        # use the best rule to build the root
		root = DTNode(max_rule)
		labels = Counter(map(lambda x:x[1], X)).most_common() # record each label and its counts, get most common
		root.label = random.choice(labels)[0] # pair: C2:10, get "C2"

		if max_gain == 0:
			root.isLeaf = True
			return root

		Xl, Xr = max_rule.split(X) # split data into two branches by that rule
		root.left = self.__buildTree(Xl) # build tree recursively
		root.right = self.__buildTree(Xr)

		return root

	def __printTree(self, root, depth, fp):

		for _ in xrange(depth):
			fp.write('-----')
		fp.write('>')

		if root.isLeaf == True:
			fp.write(root.label+'\n')
			return

		fp.write("attributeId=%s, criterion=%s\n" % (root.rule.attributeId, str(root.rule.criterion)))

		self.__printTree(root.left, depth+1, fp)
		self.__printTree(root.right, depth+1, fp)

	def __predict(self, root, x):
		# if the leaf node, the end of branch, return the label
		if root.isLeaf:
			return root.label

		if root.rule.belongToLeft(x):
			return self.__predict(root.left, x)
		else:
			return self.__predict(root.right, x)

	def __pruneTree(self, root, X):
		n = len(X)
		true_labels = map(lambda x:x[1], X)
		error_parent = n - sum([1 for l in true_labels if l == root.label]) # number of misclassified data

		if root.isLeaf:
			return error_parent

		Xl, Xr = root.rule.split(X)

		error_l = self.__pruneTree(root.left, Xl)
		error_r = self.__pruneTree(root.right, Xr)

		error_child = error_l + error_r

		if error_child > error_parent:
			root.left = None
			root.right = None
			root.isLeaf = True

		min_error = min(error_child, error_parent) # mininum error that can be achieved in case we prune or not
		return min_error

	def printTree(self, fname):
		fp = open(fname, 'w')
		self.__printTree(self.root, 0, fp)
		fp.close()

	def buildTree(self, X):
		self.root = self.__buildTree(X) # we have self.root as class variable

	def predict(self, x):
		return self.__predict(self.root, x) # self.root obtained from previous step

	def pruneTree(self, X):
		return self.__pruneTree(self.root, X)

	def buildAndPruneTree(self, X, dh):
		trainData, pruneData = dh.splitData(X, gap=20)	
		self.buildTree(trainData)
		self.pruneTree(pruneData)

def evaluate(predicted_labels, validData):
    """
	:type predicted_labels: List[str]
	:type validData: List[List]
	:rtype: float
	:compute accuracy by comparing predict labels with true lables in validation dataset
    """
    true_labels = map(lambda x:x[1], validData)
    count = 0.
    for i, l in enumerate(true_labels):
    	if l == predicted_labels[i]:
    		count += 1
    accuracy = count/len(true_labels) # the number of correct guess/ total number
    return accuracy

if __name__ == '__main__':

	# numFeature = 6
	# numFolds = 20
	# trainFile = 'trainProdSelection.arff'
	# testFile = 'testProdSelection.arff'

	numFeature = 8
	numFolds = 20
	trainFile = 'trainProdIntro.binary.arff'
	testFile = 'testProdIntro.binary.arff'

	dh = DataHandler(numFeature)
	data = dh.parseData(dh.loadData(trainFile))

	accs = []
	# fold will change from 0 to 19
	for fold in xrange(numFolds):
		trainData, validData = dh.splitData(data, gap=numFolds, offset=fold) # trainData is partial of original train dataset

		tree = DT(dh.numFeature, dh.choiceBox)
		tree.buildAndPruneTree(trainData, dh)

		predictedLabels = []
		for x in validData:
			predictedLabels.append(tree.predict(x))

		accs.append(evaluate(predictedLabels, validData))

	# average accuracy in this trial after 20-way validation
	ave_acc = sum(accs)/numFolds # num of ways to partition data
	print "%d-fold cross validation accuracy=%f" % (numFolds, ave_acc)

	testData = dh.parseData(dh.loadData(testFile))
	predictedLabels = []
	for x in testData:
		predictedLabels.append(tree.predict(x))
	print predictedLabels


