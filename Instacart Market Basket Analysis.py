# -*- coding: utf-8 -*-
import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from functools import wraps
import time


def timer(unit):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = function(*args, **kwargs)
            t1 = time.time()
            print (f'{function.__name__} spend time: {t1 - t0:.3f} {unit} ')
            return result
        return wrapper
    return decorator

@timer('s') 
def readData(filePath):
    orderTrain = pd.read_csv(filePath + 'order_products__train.csv')
    orderPrior = pd.read_csv(filePath + 'order_products__prior.csv')
    product = pd.read_csv(filePath + 'products.csv')
    aisle = pd.read_csv(filePath + 'aisles.csv')
    department = pd.read_csv(filePath + 'departments.csv')
    return orderTrain, orderPrior, product, aisle, department

@timer('s') 
def dataProcess():    
    order = pd.concat([orderTrain,orderPrior], axis = 0)
    order = pd.merge(order, product, how = 'left', on = ['product_id'])
    aisleLevel = order[['order_id','aisle_id']]
    sample = pd.merge(aisleLevel, aisle, how = 'left', on = ['aisle_id'])
    sample = sample.drop('aisle_id',axis=1)
    sampleList = sample.groupby(['order_id'])['aisle'].agg(list) 
    return sample, sampleList

@timer('s') 
def dataFit(sampleList):
    te = TransactionEncoder()
    teArray = te.fit(sampleList).transform(sampleList)
    df = pd.DataFrame(teArray, columns=te.columns_)
    return df
    
    
@timer('s') 
def frequentItemset(df, minSupport, minLift):    
    frequentSets = apriori(df, min_support=minSupport, use_colnames=True)
    rule = association_rules(frequentSets, metric = 'lift' ,min_threshold = minLift)
    frequentSets['length'] = frequentSets['itemsets'].apply(lambda x: len(x))
    combo = frequentSets[ (frequentSets['length'] > 1) &  (frequentSets['support'] >= minSupport)]
    return combo, rule 



if __name__ == '__main__':
    filePath = 'C:/Users/mase0002/Desktop/Consumer Data/Basket/'
    orderTrain, orderPrior, product, aisle, department = readData(filePath)
    sample, sampleList = dataProcess()
    df = dataFit(sampleList)
    #df.to_csv('fitted data.csv', index = False)
    combo, rule = frequentItemset(df, 0.05, 0.1)
    #combo.to_csv('combo.csv', index = False)
    #rule.to_csv('rule.csv', index = False)