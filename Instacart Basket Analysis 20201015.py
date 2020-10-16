# -*- coding: utf-8 -*-
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from functools import wraps
import time
import os

def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(f'{function.__name__} spent time: {t1 - t0:.3f} s ')
        return result
    return wrapper


@timer
def readData(filePath):
    orderTrain = pd.read_csv(os.path.join(filePath, 'order_products__train.csv'))
    orderPrior = pd.read_csv(os.path.join(filePath, 'order_products__prior.csv'))
    product = pd.read_csv(os.path.join(filePath, 'products.csv'))
    aisle = pd.read_csv(os.path.join(filePath, 'aisles.csv'))
    
    order = pd.concat([orderTrain, orderPrior], axis=0)
    order = pd.merge(order, product, how='left', on=['product_id'])
    order = pd.merge(order, aisle, how='left', on=['aisle_id'])
    return order, product


@timer
def dataProcess():
    sample = order[['order_id', 'aisle']]
    sampleList = sample.groupby(['order_id'])['aisle'].agg(list)
    return sampleList

@timer
def dataProcessBestSeller(cutPoint):   
    itemVolume = order.groupby(['aisle', 'product_name'])['product_id'].count().rename('sales_volume')
    itemVolume = itemVolume.to_frame().reset_index()
    
    threshold = itemVolume.groupby(['aisle'])['sales_volume'].quantile(cutPoint).rename('threshold')
    threshold = threshold.to_frame().reset_index()
    
    orderBestSeller = pd.merge(order, threshold, how='left', on=['aisle'])
    orderBestSeller = pd.merge(orderBestSeller, itemVolume, how='left', on=['aisle', 'product_name'])
    orderBestSeller.loc[:, 'best_seller'] = orderBestSeller['sales_volume'] > orderBestSeller['threshold']
    
    bestSeller = orderBestSeller[(orderBestSeller['best_seller'])].copy()
    bestSeller.loc[:, 'label'] = bestSeller.loc[:,'aisle'].copy() + ' (BestSeller)'
   
    normalProduct = orderBestSeller[(orderBestSeller['best_seller'] == False)].copy()
    normalProduct.loc[:, 'label'] = normalProduct.loc[:,'aisle'].copy() 
    
    
    sampleBestSeller = pd.concat([bestSeller, normalProduct], axis=0)
    sampleBestSeller = sampleBestSeller[['order_id', 'label']]
    sampleListBestSeller = sampleBestSeller.groupby(['order_id'])['label'].agg(list)
    return sampleListBestSeller

@timer
def dataFit(sampleList):
    te = TransactionEncoder()
    teArray = te.fit(sampleList).transform(sampleList)
    df = pd.DataFrame(teArray, columns=te.columns_)
    return df


@timer
def frequentItemset(df, minSupport, minLift):
    frequentSets = apriori(df, min_support=minSupport, use_colnames=True)
    rule = association_rules(frequentSets, metric='lift', min_threshold=minLift)
    frequentSets['length'] = frequentSets['itemsets'].apply(lambda x: len(x))
    combo = frequentSets[(frequentSets['length'] > 1) & (frequentSets['support'] >= minSupport)]
    return combo, rule




if __name__ == '__main__':
    filePath = os.path.join(os.getcwd(), 'data')
    order, product = readData(filePath)

    sampleList = dataProcess()
    df = dataFit(sampleList)
    combo, rule = frequentItemset(df, 0.05, 1)

    sampleListBestSeller = dataProcessBestSeller(0.95)
    dfBestSeller = dataFit(sampleListBestSeller)
    comboBestSeller, ruleBestSeller = frequentItemset(dfBestSeller, 0.05, 1)
    

    combo.to_csv(os.path.join(filePath, 'combo.csv'), index = False)
    rule.to_csv(os.path.join(filePath, 'rule.csv'), index = False)
    comboBestSeller.to_csv(os.path.join(filePath, 'comboBestSeller.csv'), index = False)
    ruleBestSeller.to_csv(os.path.join(filePath, 'ruleBestSeller.csv'), index = False)
