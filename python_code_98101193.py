student_number = 98101193
Name = 'علیرضا'
Last_Name = 'ایجی'

'''
each xi ---> 4 value
each xi ---> 1 value
784*4 + 2*1 = 3138
'''



def sample_z1():
    global bayes_net
    z1_axis = []
    size = len(bayes_net['prior_z1'])
    for i in range(size):
        if i == 0:
            z1_axis.append(get_p_z1(-3))
            continue
        z1_axis.append(get_p_z1(-3+0.25*i) + z1_axis[i-1])
    
    randomNumber = np.random.rand()
    sampleNum = -3.0
    for i in range(size):
        if randomNumber < z1_axis[i]:
            return sampleNum
        else:
            sampleNum = -3 + (i) * 0.25

def sample_z2():
    global bayes_net
    z2_axis = []
    size = len(bayes_net['prior_z2'])
    for i in range(size):
        if i ==0:
            z2_axis.append(get_p_z2(-3))
            continue
        z2_axis.append(get_p_z2(-3+0.25*i) + z2_axis[i-1])

    randomNumber = np.random.rand()
    sampleNum = -3.0
    for i in range(size):
        if randomNumber < z2_axis[i]:
            return sampleNum
        else:
            sampleNum = -3 + (i) * 0.25

def sample_xk(z1_value,z2_value,k):
    global bayes_net
    p = get_p_xk_cond_z1_z2(z1_value,z2_value,k)
    randomNum = np.random.rand()
    if randomNum < p :
        return 1
    else:
        return 0

def sample_joint_probability():
    z1Num = sample_z1()
    z2Num = sample_z2()
    x_samples = []
    for i in range(1,785):
        x_samples.append(sample_xk(z1Num,z2Num,i))

    
    arr = np.array(x_samples)
    sample_arr = arr.reshape(28,28)
    yPoints = np.array(sample_arr)
    string = "Z1="+str(z1Num) + "       Z2=" + str(z2Num)
    plt.xlabel(string)
    plt.imshow(yPoints)
    plt.show()


for i in range(5):
    sample_joint_probability()

def imagePlot():
    x = 1
    fig = plt.figure(figsize=(20, 20))
    for i in range(-12,13,1):
        for j in range(-12,13,1):
            fig.add_subplot(25, 25, x)
            imageArr = np.array(get_p_x_cond_z1_z2(i/4,j/4))
            image = imageArr.reshape(28,28)
            plt.imshow(image)
            plt.axis('off')
            x = x + 1
imagePlot()



def CreateConditionalDictionary():
    global bayes_net,disc_z1,disc_z2
    condDict = {}
    z1z2Joint = {}
    for z1 in disc_z1:
        for z2 in disc_z2:
            Px1 = get_p_x_cond_z1_z2(z1,z2) 
            condDict[(z1,z2)] = Px1
            z1z2Joint[(z1,z2)] = get_p_z1(z1) * get_p_z2(z2)
    return condDict,z1z2Joint

def GetLogLikelihoodOfSample(condDict,z1z2Joint,sample):
    sum = 0
    sample = np.array(sample)
    sampleNot = np.logical_not(sample).astype(int)
    for key in condDict.keys():
        x1Prob = condDict[key]
        difference = sampleNot - x1Prob
        sampleProb = np.abs(difference)
        sum = sum + np.prod(sampleProb) * z1z2Joint[key]

    return np.log10(sum)


def GetAllLikelihoods(data):
    likelihood_lst = []
    condDict,z1z1Joint = CreateConditionalDictionary()
    i = 0
    for sample in data:
        s = GetLogLikelihoodOfSample(condDict,z1z1Joint,sample)
        likelihood_lst.append(s)

    return np.array(likelihood_lst)

def GetDataStatistics(logLikelihoodLst):
    meanNum = np.mean(logLikelihoodLst)
    stdNum = np.std(logLikelihoodLst)
    return meanNum,stdNum


def TrainValidationData():
    global val_data
    val_log_likelihood = GetAllLikelihoods(val_data)
    meanValue,stdValue = GetDataStatistics(val_log_likelihood)
    return meanValue,stdValue

global mean_val,std_val

mean_val,std_val = TrainValidationData()


def TestClassification():
    global test_data,mean_val,std_val
    test_log_likelihoods = GetAllLikelihoods(test_data)
    real = [sample for sample in test_log_likelihoods if np.abs(sample-mean_val) <= 3*std_val]
    corrupted = [sample for sample in test_log_likelihoods if sample not in real]
    return real,corrupted

real,corrupted = TestClassification()

def PlotHistogram(data,title,xlabel,sf):
      plt.figure()
      plt.hist(data)
      plt.xlabel(xlabel)
      plt.ylabel("frequency")
      plt.title(title)
      plt.savefig(sf)
      plt.show()
      plt.close()

def removeInf(lst):
      size = len(lst)
      for i in range(0,size):
            if lst[i] == float('-inf'):
                  lst.remove(lst[i])

removeInf(corrupted)
removeInf(real)
PlotHistogram(real,"real images","log likelihood","real_hist")
PlotHistogram(corrupted,"corrupted images","log likelihood","corrupted_hist")


import pandas as pd
import numpy as np
import copy
from bisect import bisect_left
import matplotlib.pyplot as plt
import random

nodes_dict = {}
cpts_dict = {}

cpts_dict["a"] = pd.DataFrame({"a": [0, 1], "P": [0.2, 0.8]})
cpts_dict["b"] = pd.DataFrame({"b": [0, 1], "P": [0.45, 0.55]})
cpts_dict["e"] = pd.DataFrame({"e": [1, 1, 0, 0], "b": [1, 0, 1, 0], "P": [0.3, 0.9, 0.7, 0.1]})
cpts_dict["c"] = pd.DataFrame({"c": [0, 1, 0, 1, 0, 1, 0, 1], "a": [0, 0, 1, 1, 0, 0, 1, 1], "e": [0, 0, 0, 0, 1, 1, 1, 1], "P": [0.3, 0.7, 0.5, 0.5, 0.85, 0.15, 0.95, 0.05]})
cpts_dict["d"] = pd.DataFrame({"d": [0, 1, 0, 1, 0, 1, 0, 1], "a": [0, 0, 1, 1, 0, 0, 1, 1], "c": [0, 0, 0, 0, 1, 1, 1, 1], "P": [0.2, 0.8, 0.5, 0.5, 0.35, 0.65, 0.33, 0.67]})
cpts_dict["f"] = pd.DataFrame({"f": [1, 1, 0, 0], "d": [1, 0, 1, 0], "P": [0.2, 0.25, 0.8, 0.75]})

nodes_dict["a"] = {"parents": [], "children": ["c", "d"]}
nodes_dict["b"] = {"parents": [], "children": ["e"]}
nodes_dict["e"] = {"parents": ["b"], "children": ["c"]}
nodes_dict["c"] = {"parents": ["a"], "children": ["d"]}
nodes_dict["d"] = {"parents": ["a", "c"], "children": ["f"]}
nodes_dict["f"] = {"parents": ["d"], "children": []}



class BN(object):

    def __init__(self,n,nodes_dict,cpts_dict):
        self.n = n
        self.nodes_dict = nodes_dict
        self.cpts_dict = cpts_dict
        self.joinCpts = self.joinAllTables()
        self.probAxis = self.createProbabilityAxis(self.joinCpts)

    def cpt(self, node) -> dict:
        return self.cpts_dict[node]
    
    def pmf(self, queryLst, evidencesLst) -> float:
        evidenctTable = copy.deepcopy(self.joinCpts)
        
        for evidence in evidencesLst:
            evidenctTable = evidenctTable.loc[evidenctTable[evidence[0]] == evidence[1]]
        
        bottomSum = evidenctTable['P'].sum()
        for query in queryLst:
            evidenctTable = evidenctTable.loc[evidenctTable[query[0]] == query[1]]

        jointSum = evidenctTable['P'].sum()
        return float(jointSum/bottomSum)

    def joinAllTables(self):
        new_cpts = copy.deepcopy(self.cpts_dict)
        i = 0
        while True:
            first_key = list(new_cpts.keys())[0]
            first_cpt = new_cpts[first_key]
            cpt_for_join = None
            cpt_for_join_key = ''
            for table in new_cpts:
                if table == first_key:
                    continue

                if len(np.intersect1d(new_cpts[table].columns, first_cpt.columns)) > 1:
                    cpt_for_join = new_cpts[table]
                    cpt_for_join_key = table
                    break

            first_cpt.rename(columns={"P": "P0"}, inplace=True)
            cpt_for_join.rename(columns={"P": "P1"}, inplace=True)
            joined_cpt = pd.merge(first_cpt, cpt_for_join, how="inner")
            joined_cpt["P"] = joined_cpt["P0"] * joined_cpt["P1"]
            joined_cpt.drop("P0", axis=1, inplace=True)
            joined_cpt.drop("P1", axis=1, inplace=True)
            new_cpts.pop(first_key)
            new_cpts.pop(cpt_for_join_key)
            new_key = "join" + str(i)
            new_cpts[new_key] = joined_cpt
            i = i + 1

            if len(new_cpts) == 1:
                break

        key = "join" + str(i-1)
        joinAllCpt = new_cpts[key]
        return joinAllCpt

    def createProbabilityAxis(self,cpts):
        probList = cpts["P"].tolist()
        sum = 0
        probAxis = []
        for i in range(len(probList)):
            if i == 0:
                probAxis.append(probList[i])
                sum = sum + probList[i]
                continue
            sum = sum + probList[i]
            probAxis.append(sum)
        return probAxis

    def priorSampling(self,queryLst,evidenceLst,num_iteration):
        sampledData = pd.DataFrame({},columns=self.joinCpts.columns)
        iteration = 0
        while iteration < num_iteration:
            randomNum = np.random.rand()
            i = bisect_left(self.probAxis,randomNum)
            sampledData.loc[len(sampledData)] = self.joinCpts.iloc[i]
            iteration = iteration + 1

        totalTable = copy.deepcopy(sampledData)
        for evidence in evidenceLst:
            totalTable = totalTable.loc[ totalTable[evidence[0]] == evidence[1]]

        totalNum = len(totalTable)

        for query in queryLst:
            totalTable = totalTable.loc [ totalTable[query[0]] == query[1]]

        queryProb = float(len(totalTable) / totalNum)

        return queryProb

    def rejectionSampling(self,queryLst,evidenceLst,num_iteration):
        def acceptOrReject(sample,evidenceLst):
            accept = True
            for evidence in evidenceLst:
                if sample[evidence[0]] != evidence[1]:
                    accept = False
                    break
            return accept
        
        iteration = 0
        sampledData = pd.DataFrame({},columns=self.joinCpts.columns)
        while iteration < num_iteration:
            randomNum = np.random.rand()
            i = bisect_left(self.probAxis,randomNum)
            sample = self.joinCpts.iloc[i]
            acceptStatus = acceptOrReject(sample,evidenceLst) 
            if acceptStatus:
                sampledData.loc[len(sampledData)] = sample
                iteration = iteration + 1
        
        totalNum = len(sampledData)

        for query in queryLst:
            sampledData = sampledData.loc [ sampledData[query[0]] == query[1]]

        queryProb = float(len(sampledData)/totalNum)
        return queryProb

    def likelihoodWeighting(self,queryLst,evidenceLst,num_iteration):
        def getWeight(sample,evidenceLst):
            w = 1
            for evidence in evidenceLst:
                table = self.cpt(evidence[0])
                for col in self.nodes_dict.keys():
                    if col not in self.nodes_dict[evidence[0]]['parents']:
                        continue
                    table = table.loc [ table[col]== sample[col] ]
                table = table.loc [ table[evidence[0]] == evidence[1] ]
                w = w * table.iloc[-1]['P']
            return w

        weight_cpt = copy.deepcopy(self.joinCpts)
        weight_cpt['w'] = 0.1
        for evidence in evidenceLst:
            weight_cpt = weight_cpt.loc[weight_cpt[evidence[0]]==evidence[1]]

        for index,row in weight_cpt.iterrows() :
            w = getWeight(row,evidenceLst)
            weight_cpt.at[index,'w'] = w


        iteration = 0
        sampledData = pd.DataFrame({},columns=weight_cpt.columns)
        indexLst = list(weight_cpt.index.values)
        weight_cpt_prob_axis = self.createProbabilityAxis(weight_cpt)
        while iteration < num_iteration:
            randomNum = np.random.rand() * weight_cpt_prob_axis[-1]
            i = bisect_left(weight_cpt_prob_axis,randomNum)
            sampledData.loc[len(sampledData)] = weight_cpt.loc[indexLst[i]]
            iteration = iteration + 1

        totalW = sampledData['w'].sum()

        for query in queryLst:
            sampledData = sampledData.loc [ sampledData[query[0]] == query[1]]

        nW = sampledData['w'].sum()
        return float(nW/totalW)

    def gibbsSampling(self,queryLst,evidenceLst,num_iteration,num_burnin):
        def initialize(evidenceLst):
            evidenceDict = dict(evidenceLst)
            evidenceKey = list(evidenceDict.keys())
            allNodesKey = list(self.nodes_dict.keys())
            queryHiddenKey = [i for i in evidenceKey + allNodesKey if i not in evidenceKey or i not in allNodesKey]
            init = {}
            for key in queryHiddenKey:
                p = np.random.rand()
                if p < 0.5:
                    init[key] = 0
                else:
                    init[key] = 1
            return list(init.items())

        def createSingleSample(evidenceLst,queryHiddenLst):
            sample = {}
            sample.update(dict(evidenceLst))
            total = evidenceLst + queryHiddenLst
            for queryHidden in queryHiddenLst:
                total.remove(queryHidden)
                randomNum = np.random.rand()
                p = self.pmf([queryHidden],total)
                if randomNum < p :
                  sample[queryHidden[0]] = queryHidden[1]
                else:
                    sample[queryHidden[0]] = 1 - queryHidden[1]
                total.append(queryHidden)
            return sample

        iteration = 0
        sampledData = pd.DataFrame({},columns=self.joinCpts.columns)
        num_iteration = num_iteration + num_burnin
        while iteration < num_iteration:
            queryHiddenLst = initialize(evidenceLst)
            sample = createSingleSample(evidenceLst,queryHiddenLst)
            if iteration > num_burnin:
                sampledData.loc[len(sampledData)] = sample
            iteration = iteration + 1

        totalTable = copy.deepcopy(sampledData)
        for evidence in evidenceLst:
            totalTable = totalTable.loc[ totalTable[evidence[0]] == evidence[1]]

        totalNum = len(totalTable)

        for query in queryLst:
            totalTable = totalTable.loc [ totalTable[query[0]] == query[1]]

        queryProb = float(len(totalTable) / totalNum)

        return queryProb        

    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin = 1e2) -> float:
        if sampling_method == "prior":
            return self.priorSampling(query,evidence,num_iter)
        elif sampling_method == 'rejection':
            return self.rejectionSampling(query,evidence,num_iter)
        elif sampling_method == 'likelihood':
            return self.likelihoodWeighting(query,evidence,num_iter)
        elif sampling_method == 'gibbs':
            return self.gibbsSampling(query,evidence,num_iter,num_burnin)
        else:
            return -1
    

network = BN(nodes_dict=nodes_dict,cpts_dict=cpts_dict,n=6)





realP0 = network.pmf(queryLst=[('f',1)],evidencesLst=[('a',1),('e',0)])
realP1 = network.pmf(queryLst=[('c',0),('b',1)],evidencesLst=[('f',1),('d',0)]) 
print("inference by enumuaration: ")
print("P(f=1|a=1,e=0): " + str(realP0))
print("P(c=0,b=1|f=1,d=0): "+ str(realP1))
x = [100,500,1000,3000,10000]
priorErrorsQ1 = []
rejectErrorsQ1 = []
likelihoodErrorsQ1 = []
gibbsErrorsQ1 = []
priorErrorsQ2 = []
rejectErrorsQ2 = []
likelihoodErrorsQ2 = []
gibbsErrorsQ2 = []

priorErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'prior',100) - realP0)
priorErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'prior',500) - realP0)
priorErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'prior',1000) - realP0)
priorErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'prior',3000) - realP0)
priorErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'prior',10000) - realP0)
print("prior for query 1 ended")
rejectErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'rejection',100) - realP0)
rejectErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'rejection',500) - realP0)
rejectErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'rejection',1000) - realP0)
rejectErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'rejection',3000) - realP0)
rejectErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'rejection',10000) - realP0)
print("rejection for query1 ended")
likelihoodErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'likelihood',100) - realP0)
likelihoodErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'likelihood',500) - realP0)
likelihoodErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'likelihood',1000) - realP0)
likelihoodErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'likelihood',3000) - realP0)
likelihoodErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'likelihood',10000) - realP0)
print("likelihood for query1 ended")
gibbsErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',100) - realP0)
gibbsErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',500) - realP0)
gibbsErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',1000) - realP0)
gibbsErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',3000) - realP0)
gibbsErrorsQ1.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',10000) - realP0)
print("gibbs ended")

priorErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'prior',100) - realP1)
priorErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'prior',500) - realP1)
priorErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'prior',1000) - realP1)
priorErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'prior',3000) - realP1)
priorErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'prior',10000) - realP1)
print("prior for query 2 ended")
rejectErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'rejection',100) - realP1)
rejectErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'rejection',500) - realP1)
rejectErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'rejection',1000) - realP1)
rejectErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'rejection',3000) - realP1)
rejectErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'rejection',10000) - realP1)
print("rejection for query 2 ended")
likelihoodErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'likelihood',100) - realP1)
likelihoodErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'likelihood',500) - realP1)
likelihoodErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'likelihood',1000) - realP1)
likelihoodErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'likelihood',3000) - realP1)
likelihoodErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'likelihood',10000) - realP1)
print("likelihood for query 2 ended")
gibbsErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'gibbs',100) - realP1)
gibbsErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'gibbs',500) - realP1)
gibbsErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'gibbs',1000) - realP1)
gibbsErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'gibbs',3000) - realP1)
gibbsErrorsQ2.append(network.sampling([('c',0),('b',1)],[('f',1),('d',0)],'gibbs',10000) - realP1)
print("gibbs ended")


plt.title('Query 1')
plt.plot(x,np.abs(priorErrorsQ1))
plt.plot(x,np.abs(rejectErrorsQ1))
plt.plot(x,np.abs(likelihoodErrorsQ1))
plt.plot(x,np.abs(gibbsErrorsQ1))
plt.legend(["prior", "rejection" , "likelihood weighting",'gibbs'], loc ="upper right")
plt.show()
plt.title('Query 2')
plt.plot(x,np.abs(priorErrorsQ2))
plt.plot(x,np.abs(rejectErrorsQ2))
plt.plot(x,np.abs(likelihoodErrorsQ2))
plt.plot(x,np.abs(gibbsErrorsQ2))
plt.legend(["prior", "rejection" , "likelihood weighting",'gibbs'], loc ="upper right")
plt.show()

gibbs_diffrent = []

gibbs_diffrent.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',100,200) - realP0)
gibbs_diffrent.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',500,200) - realP0)
gibbs_diffrent.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',1000,200) - realP0)
gibbs_diffrent.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',3000,200) - realP0)
gibbs_diffrent.append(network.sampling([('f',1)],[('a',1),('e',0)],'gibbs',10000,200) - realP0)





plt.plot(x,np.abs(gibbsErrorsQ1))
plt.plot(x,np.abs(gibbs_diffrent))
plt.legend(["burn_in=100", "burn_in=200"], loc ="upper right")
plt.show()







def get_mean_towers_coor(time_step: int, tower_records: list):
    x = np.average([tower_coor[0] for tower_coor in tower_records[time_step]])
    y = np.average([tower_coor[1] for tower_coor in tower_records[time_step]])
    return x, y


def P_coor0(coor0):
    x0, y0 = coor0
    return scipy.stats.multivariate_normal.pdf([x0, y0], 
                            mean=moving_model.get('Peurto_coordinates'), cov=moving_model.get('INIT_COV'))



def P_coor_given_prevCoor(coor, prev_coor):
    px = expon.pdf(np.abs(coor[0] - prev_coor[0]) , 0 ,scale=moving_model["X_STEP"]) * 0.5
    py = expon.pdf(np.abs(coor[1] - prev_coor[1]) , 0 ,scale= moving_model["Y_STEP"])
    return (px,py)

    
def P_towerCoor_given_coor(tower_coor, tower_std, coor):
    px = norm.pdf(tower_coor[0],coor[0],tower_std)
    py = norm.pdf(tower_coor[1],coor[1],tower_std)
    return (px,py)
    
    
def P_record_given_coor(recordLst, coor, towers_info):
    totalPx = 1
    totalPy = 1
    i = 0
    for record in recordLst:
        info = towers_info[str(i+1)]["std"]
        px,py = P_towerCoor_given_coor(record,info,coor)
        totalPx = totalPx * px 
        totalPy = totalPy * py       
        i = i + 1

    return (totalPx,totalPy)












max_Px, max_Py = 0, 0
interval, step = 20, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None

towers_mean_x1, towers_mean_y1 = get_mean_towers_coor(1, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):
        
         for x1 in range(int(towers_mean_x1 - interval), int(towers_mean_x1 + interval), step):
            for y1 in range(int(towers_mean_y1 - interval), int(towers_mean_y1 + interval), step):
                    
                coor0 = (x0, y0)
                coor1 = (x1, y1)

                rec0 = tower_records[0]
                rec1 = tower_records[1]
                Px_coor1_given_coor0 , Py_coor1_given_coor0 = P_coor_given_prevCoor(coor1,coor0)

                p_c0 = P_coor0(coor0)
                
                P_rec0_given_x0, P_rec0_given_y0 = P_record_given_coor(rec0, coor0, towers_info)
                P_rec1_given_x1, P_rec1_given_y1 = P_record_given_coor(rec1, coor1, towers_info)

                Px = P_rec0_given_x0 * Px_coor1_given_coor0 * P_rec1_given_x1 * p_c0
                Py = P_rec0_given_y0 * Py_coor1_given_coor0 * P_rec1_given_y1 * p_c0
                
                if Px > max_Px:
                    best_x0 = x0
                    best_x1 = x1
                    max_Px = Px
                    
                if Py > max_Py:
                    best_y0 = y0
                    best_y1 = y1
                    max_Py = Py               


                    
            
coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))






max_Px, max_Py = 0, 0
interval, step = 10, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None
best_x2, best_y2 = None, None

towers_mean_x2, towers_mean_y2 = get_mean_towers_coor(2, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):
        
        for x1 in range(int(coor1_estimations[-1][0] - interval), int(coor1_estimations[-1][0] + interval), step):
            for y1 in range(int(coor1_estimations[-1][1] - interval), int(coor1_estimations[-1][1] + interval), step):

                for x2 in range(int(towers_mean_x2 - interval), int(towers_mean_x2 + interval), step):
                    for y2 in range(int(towers_mean_y2 - interval), int(towers_mean_y2 + interval), step):
                    
                        coor0 = (x0, y0)
                        coor1 = (x1, y1)
                        coor2 = (x2, y2)                        

                        rec0 = tower_records[0]
                        rec1 = tower_records[1]
                        rec2 = tower_records[2]

                        Px_coor1_given_coor0 , Py_coor1_given_coor0 = P_coor_given_prevCoor(coor1,coor0)
                        Px_coor2_given_coor1 , Py_coor2_given_coor1 = P_coor_given_prevCoor(coor2,coor1)

                        p_c0 = P_coor0(coor0)
                
                        P_rec0_given_x0, P_rec0_given_y0 = P_record_given_coor(rec0, coor0, towers_info)
                        P_rec1_given_x1, P_rec1_given_y1 = P_record_given_coor(rec1, coor1, towers_info)
                        P_rec2_given_x2, P_rec2_given_y2 = P_record_given_coor(rec2, coor2, towers_info)

                        Px = P_rec0_given_x0 * Px_coor1_given_coor0 * Px_coor2_given_coor1 * P_rec1_given_x1 * P_rec2_given_x2 * p_c0
                        Py = P_rec0_given_y0 * Py_coor1_given_coor0 * Py_coor2_given_coor1 * P_rec1_given_y1 * P_rec2_given_y2 * p_c0
                
                        if Px > max_Px:
                            best_x0 = x0
                            best_x1 = x1
                            best_x2 = x2
                            max_Px = Px

                        if Py > max_Py:
                            best_y0 = y0
                            best_y1 = y1
                            best_y2 = y2
                            max_Py = Py        


                    
            
coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))
coor2_estimations.append((best_x2, best_y2))
                    

print(f'real_coor0: {real_coor(0)} - Estimated_coor0: {best_x0, best_y0}')
print(f'Estimation_error: {dist((best_x0, best_y0), real_coor(0))}')
print()
print(f'real_coor1: {real_coor(1)} - Estimated_coor1: {best_x1, best_y1}')
print(f'Estimation_error: {dist((best_x1, best_y1), real_coor(1))}')
print()
print(f'real_coor2: {real_coor(2)} - Estimated_coor2: {best_x2, best_y2}')
print(f'Estimation_error: {dist((best_x2, best_y2), real_coor(2))}')



max_Px, max_Py = 0, 0
interval, step = 10, 5

best_x0, best_y0 = None, None
best_x1, best_y1 = None, None
best_x2, best_y2 = None, None
best_x3, best_y3 = None, None

towers_mean_x3, towers_mean_y3 = get_mean_towers_coor(3, tower_records)

for x0 in range(int(coor0_estimations[-1][0] - interval), int(coor0_estimations[-1][0] + interval), step):
    for y0 in range(int(coor0_estimations[-1][1] - interval), int(coor0_estimations[-1][1] + interval), step):

        for x1 in range(int(coor1_estimations[-1][0] - interval), int(coor1_estimations[-1][0] + interval), step):
            for y1 in range(int(coor1_estimations[-1][1] - interval), int(coor1_estimations[-1][1] + interval), step):

                for x2 in range(int(coor2_estimations[-1][0] - interval), int(coor2_estimations[-1][0] + interval), step):
                    for y2 in range(int(coor2_estimations[-1][1] - interval), int(coor2_estimations[-1][1] + interval), step):

                        for x3 in range(int(towers_mean_x3 - interval), int(towers_mean_x3 + interval), step):
                            for y3 in range(int(towers_mean_y3 - interval), int(towers_mean_y3 + interval), step):
                                
                                coor0 = (x0, y0)
                                coor1 = (x1, y1)
                                coor2 = (x2, y2)                        
                                coor3 = (x3, y3)

                                rec0 = tower_records[0]
                                rec1 = tower_records[1]
                                rec2 = tower_records[2]
                                rec3 = tower_records[3]

                                Px_coor1_given_coor0 , Py_coor1_given_coor0 = P_coor_given_prevCoor(coor1,coor0)
                                Px_coor2_given_coor1 , Py_coor2_given_coor1 = P_coor_given_prevCoor(coor2,coor1)
                                Px_coor3_given_coor2 , Py_coor3_given_coor2 = P_coor_given_prevCoor(coor3,coor2)                                

                                p_c0 = P_coor0(coor0)
                
                                P_rec0_given_x0, P_rec0_given_y0 = P_record_given_coor(rec0, coor0, towers_info)
                                P_rec1_given_x1, P_rec1_given_y1 = P_record_given_coor(rec1, coor1, towers_info)
                                P_rec2_given_x2, P_rec2_given_y2 = P_record_given_coor(rec2, coor2, towers_info)
                                P_rec3_given_x3, P_rec3_given_y3 = P_record_given_coor(rec3, coor3, towers_info)

                                Px = P_rec0_given_x0 * Px_coor1_given_coor0 * Px_coor2_given_coor1 * Px_coor3_given_coor2 * P_rec1_given_x1 * P_rec2_given_x2 * P_rec3_given_x3 * p_c0
                                Py = P_rec0_given_y0 * Py_coor1_given_coor0 * Py_coor2_given_coor1 * Py_coor3_given_coor2 * P_rec1_given_y1 * P_rec2_given_y2 * P_rec3_given_y3 * p_c0
                
                                if Px > max_Px:
                                    best_x0 = x0
                                    best_x1 = x1
                                    best_x2 = x2
                                    best_x3 = x3
                                    max_Px = Px

                                if Py > max_Py:
                                    best_y0 = y0
                                    best_y1 = y1
                                    best_y2 = y2
                                    best_y3 = y3
                                    max_Py = Py  

coor0_estimations.append((best_x0, best_y0))
coor1_estimations.append((best_x1, best_y1))
coor2_estimations.append((best_x2, best_y2)) 
coor3_estimations.append((best_x3, best_y3))                   

print(f'real_coor0: {real_coor(0)} - Estimated_coor0: {best_x0, best_y0}')
print(f'Estimation_error: {dist((best_x0, best_y0), real_coor(0))}')
print()
print(f'real_coor1: {real_coor(1)} - Estimated_coor1: {best_x1, best_y1}')
print(f'Estimation_error: {dist((best_x1, best_y1), real_coor(1))}')
print()
print(f'real_coor2: {real_coor(2)} - Estimated_coor2: {best_x2, best_y2}')
print(f'Estimation_error: {dist((best_x2, best_y2), real_coor(2))}')
print()
print(f'real_coor3: {real_coor(3)} - Estimated_coor3: {best_x3, best_y3}')
print(f'Estimation_error: {dist((best_x3, best_y3), real_coor(3))}')



error0 = []
error1 = []
error2 = []
error3 = []

error0.append(dist(coor0_estimations[0],real_coor(0)))
error0.append(dist(coor0_estimations[1],real_coor(0)))
error0.append(dist(coor0_estimations[2],real_coor(0)))
error0.append(dist(coor0_estimations[3],real_coor(0)))
print("coor0 errors: " + str(error0))
error1.append(dist(coor1_estimations[0],real_coor(1)))
error1.append(dist(coor1_estimations[1],real_coor(1)))
error1.append(dist(coor1_estimations[2],real_coor(1)))
print("coor1 errors: " + str(error1))
error2.append(dist(coor2_estimations[0],real_coor(2)))
error2.append(dist(coor2_estimations[1],real_coor(2)))
print("coor2 errors: " + str(error2))
error3.append(dist(coor3_estimations[0],real_coor(3)))
print("coor3 errors: " + str(error3))


index = [1 , 2 , 3 , 4]

plt.plot(index,error0)
plt.plot(index[1:],error1)
plt.plot(index[2:],error2)
plt.scatter(index[3:],error3)
plt.legend(["coor0_error","coor1_error","coor2_error","coor3_error"],loc ="lower left")
plt.show()





