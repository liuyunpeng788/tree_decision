#!/usr/bin/python
# --*--coding:UTF-8 -*-

from math import log

"""
计算信息熵
"""


def calcEntropy(dataset):
    diclabel = {} ## 标签字典，用于记录每个分类标签出现的次数
    for record in dataset:
        label = record[-1]
        if label not in diclabel.keys():
             diclabel[label] = 0
        
        diclabel[label] += 1

    ### 计算熵
    entropy = 0.0
    cnt = len(dataset)    
    for label in diclabel.keys():
        prob = float(1.0 * diclabel[label]/cnt)
        entropy -= prob * log(prob,2)
    return entropy

def initDataSet():
    dataset = [[1,0,"yes"],[1,1,"yes"],[0,1,"yes"],[0,0,"no"],[1,0,"no"]]
    label = ["male","female"]
    return dataset,label


#### 拆分dataset ,根据指定的过滤选项值，去掉指定的列形成一个新的数据集
def splitDataset(dataset , col, value):
    retset = []            ## 拆分后的数据集
    for record in dataset:
        if record[col] == value :
            reducedFeatVec = record[:col]
            reducedFeatVec.extend(record[col+1:])  ### 将指定的列剔除
            retset.append(reducedFeatVec)   ### 将新形成的特征值列表追加到返回的列表中
    
    return retset


### 找出信息熵增益最大的特征值
### 参数：
###    dataset : 原始的数据集

def findBestFeature(dataset):
    numFeatures = len(dataset[0]) - 1   ### 特征值的个数
    baseEntropy = calcEntropy(dataset)      ### 计算原始数据集的熵
    baseInfoGain = 0.0               ### 初始信息增益
    bestFeature = -1                 ### 初始的最优分类特征值索引

    ### 计算每个特征值的熵
    for col in range(numFeatures):
        features = [record[col] for record in dataset]  ### 提取每一列的特征向量 如此处col= 0 ，则features = [1,1,0,0]
        uniqueFeat = set(features)
        curInfoGain = 0                                 ### 根据每一列进行拆分，所获得的信息增益
        for featVal in uniqueFeat:
            subDataset = splitDataset(dataset,col,featVal)  ### 根据col列的featVal特征值来对数据集进行划分
            prob = 1.0 * len(subDataset)/numFeatures              ### 计算子特征数据集所占比例
            curInfoGain +=  prob * calcEntropy(subDataset)   ### 计算col列的特征值featVal所产生的信息增益

 #           print "col : " ,col , "  featVal : " , featVal , "  curInfoGain :" ,curInfoGain  ,"  baseInfoGain : " ,baseInfoGain
        print "col : " ,col ,  "  curInfoGain :" ,curInfoGain  ,"  baseInfoGain : " ,baseInfoGain

        if curInfoGain > baseInfoGain:
            baseInfoGain = curInfoGain
            bestFeature = col

    return baseInfoGain,bestFeature             ### 输出最大的信息增益，以获得该增益的列

dataset,label = initDataSet()
infogain , bestFeature = findBestFeature(dataset)
print "bestInfoGain :" , infogain,  "  bestFeature:",bestFeature
