#Paolo Tamagnini - 1536242
import time
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
countBreakBool = False
iter = 20000
print 'Unfortunately this logistic regression reaches convergence in about 30 minutes,'
print 'doing approximately 10k iteration per cuisine!'
print 'That is how I maximized the loglikelihood function and got the result for kaggle!'
print "If you don't have enough time to see that yourself, you can select the maximum number of iterations you want to do for estimating probabilities for each cuisine."
print 'It is advised to choose a number higher than 100. For example if you set 200 it will take around 5 minutes.'
yn = str(raw_input("Do you want to set a maximum number of iteration per cuisine? < y / n >"))
if yn == 'y':
	iter = int(raw_input("Please type the max number of iterations:"))
	countBreakBool = True

theStart = time.time()
#CREATING DATAFRAME TRAIN
print 'Analysing data..'
dfjsonTrain = pd.read_json('train.json')

nr = len(dfjsonTrain)

#CREATING LIST OF DIFFERENT SORTED INGREDIENTS IN TRAIN
ingrTrain = []
ingrTrain = set(ingrTrain)
for x in dfjsonTrain['ingredients']:
    for i in x:
        ingrTrain.add(i)

ingrTrain = sorted(ingrTrain)
N = len(ingrTrain)

#CREATING DATAFRAME TEST
dfjsonTest = pd.read_json('test.json')
nt=len(dfjsonTest)

#CREATING LIST OF DIFFERENT SORTED INGREDIENTS IN TEST
ingrTest = []
ingrTest = set(ingrTest)
for x in dfjsonTest['ingredients']:
    for i in x:
        ingrTest.add(i)

ingrTest = sorted(ingrTest)

n = len(ingrTest)

#COMPUTING THE 2 INGREDIENT LIST INTERSECTION TO GET RID OF USELESS INGREDIENTS
testNotInTrain = []
for i in ingrTest:
    if i not in ingrTrain:
        testNotInTrain.append(i)
ok = len(testNotInTrain)


#DELETING FROM TEST INGREDIENTS NOT IN TRAIN

ingrTest = [x for x in ingrTest if x not in testNotInTrain]
n = len(ingrTest)

trainNotInTest = []
for i in ingrTrain:
    if i not in ingrTest:
        trainNotInTest.append(i)
ok2 = len(trainNotInTest)


#DELETING FROM TRAIN INGREDIENTS NOT IN TEST

ingrTrain = [x for x in ingrTrain if x not in trainNotInTest]
N = len(ingrTrain)


#CREATING LIST OF DIFFERENT CUISINE

cucine = sorted(list(set(dfjsonTrain['cuisine'].tolist())))
nc = len(cucine)


dictIngr = dict(zip(ingrTrain,range(0,N)))

#CREATING SPARSE MATRIX FROM DATAFRAME TRAIN
B = dok_matrix((nr,N+1), dtype=float)

print 'Creating sparse matrix from train..'
for i in range(0,nr):
    B[i,0] = 1.0
    for j in range(0,len(dfjsonTrain.loc[i,'ingredients'])):
		if dfjsonTrain.loc[i,'ingredients'][j] in ingrTrain:
			B[i,dictIngr[dfjsonTrain.loc[i,'ingredients'][j]]+1] = 1.0
dcook = dict(zip(cucine,range(0,len(cucine))))
Ydf = pd.DataFrame()
for x in cucine:
    Ydf[dcook[x]] = np.zeros(nr)
for i in range(0,nr):
    Ydf.loc[i,dcook[dfjsonTrain.loc[i,'cuisine']]] = 1.0 

#CREATING LOGISTIC REGRESSION PARAMETERS
B = B.tocsr()
alfa = np.zeros(nc)
eps = np.zeros(nc)
for i in range(0,nc):
	eps[i] = 10**(-1)
	alfa[i] = 10**(-2.5)
eps[5] = 10**(-1.5)
eps[7] = 10**(-1.5)
eps[14] = 10**(-1.5)
eps[17] = 10**(-1.5)
	
alfa[0] = 10**(-2)
alfa[10] = 10**(-2)
alfa[12] = 10**(-2.2)
alfa[14] = 10**(-2.2)
alfa[3] = 10**(-3)
alfa[5] = 10**(-3)
alfa[7] = 10**(-3)
alfa[9] = 10**(-3)
alfa[13] = 10**(-3)
alfa[16] = 10**(-3)
alfa[17] = 10**(-3)

#STARTING THE LOGISTIC REGRESSION ONCE FOR EVERY CUISINE
print 'Logistic Regression for each cuisine:'
print ' '
allthetetas = []
alltheHdiXs = []
for myY in range(0,nc):
    
    y = Ydf[myY]
    teta = np.ones(N+1)
    tetatemp = np.ones(N+1)
    for i in range(0,N+1):
        teta[i] = -1
        tetatemp[i] = -1
    tetap = np.zeros(N+1)

    logLtetaTemp = -10000000

    count = 0
    print 'Computing', cucine[myY], 'cuisine..' 
    while((abs(tetap - tetatemp) > eps[myY]).any()):
        HdiX = 1 / (1 + np.exp(-B.dot(teta)))
        logLteta = sum(y*np.log(HdiX) + (1-y)*np.log(1 - HdiX))
        tetap = teta + alfa[myY]*B.T.dot(y - HdiX)
        tetatemp = teta
        teta = tetap
		#DECREASING BY 1% IF logLteta DECREASE
        if logLteta < logLtetaTemp:
            alfa[myY] = alfa[myY] * 0.99
		#INCREASING BY 1% ALFA IF logLteta DECREASE
        if logLteta > logLtetaTemp:
            alfa[myY] = alfa[myY] * 1.01
        
        logLtetaTemp = logLteta
        count = count + 1
        if countBreakBool:
			if count == iter:
				break
        
    allthetetas.append(teta)
    alltheHdiXs.append(HdiX)
    print '--------------------------------------------'

print 'Testing result on train...'
#COMPUTING % OF SUCCESS OVER THE TRAIN ITSELF
HdiXdf = pd.DataFrame() 

for i in range(0,len(alltheHdiXs)):
    HdiXdf[i] = alltheHdiXs[i] 

victory = 0
for i in range(0,nr):
    guessedIndex = HdiXdf.loc[i,:][HdiXdf.loc[i,:] == max(HdiXdf.loc[i,:])].index[0]
    if cucine[guessedIndex] == dfjsonTrain.loc[i,'cuisine']:
        victory += 1
print 'Percentage of success over train:', float(victory)/nr*100,'%'	
	

dictIngrTest = dict(zip(ingrTest,range(0,n)))

#CREATING SPARSE MATRIX FROM DATAFRAME TEST
Btest = dok_matrix((nt,n+1), dtype=float)

print 'Creating sparse matrix from test..'
    
for i in range(0,nt):
    Btest[i,0] = 1.0
    for j in range(0,len(dfjsonTest.loc[i,'ingredients'])):
		if dfjsonTest.loc[i,'ingredients'][j] in ingrTest:
			Btest[i,dictIngrTest[dfjsonTest.loc[i,'ingredients'][j]]+1] = 1.0



Btest = Btest.tocsr()

#COMPUTING ALL PROBABILITIES OF EACH RECIPE TO BELONG TO EACH CUISINE USING THETA
alltheHdiXsTest = []
for myY in range(0,nc):
	HdiXtest = 1 / (1 + np.exp(-Btest.dot(allthetetas[myY])))
	alltheHdiXsTest.append(HdiXtest)

#CREATING RELATED PROBABILITIES DATAFRAME
HdiXdfTest = pd.DataFrame() 

for i in range(0,len(alltheHdiXsTest)):
    HdiXdfTest[i] = alltheHdiXsTest[i] 



#CHECKING HIGHER PROBABILITY PER RECIPE
guessedYsTest=[]
for i in range(0,nt):
    guessedYsTest.append(cucine[HdiXdfTest.loc[i,:][HdiXdfTest.loc[i,:] == max(HdiXdfTest.loc[i,:])].index[0]])

#CREATING RESULT DATAFRAME
sub = pd.DataFrame()
sub['id']=dfjsonTest['id']
sub['cuisine']=guessedYsTest

#PRINTING RESULT
sub.to_csv('1536242-submission.csv',index = False)

theEnd = time.time()
print 'Elapsed time:',int(theEnd - theStart)/60, 'm,', int(theEnd - theStart)%60,'s'
	






