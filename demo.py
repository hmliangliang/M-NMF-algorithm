import numpy as np
import Kcut
import pandas
import time
import Evaluation
import MNMF
import scipy.io as sci
import math
import sklearn.preprocessing as pre
if __name__=='__main__':
   start=time.time()
   '''thisdata = pandas.read_csv('facebook.csv',sep=',',header=None)
thisdata = np.array(thisdata)
n=max(np.max(thisdata[:,0]),np.max(thisdata[:,1]))+1
data=np.zeros((n,n))
for i in range(thisdata.shape[0]):
   data[thisdata[i,0],thisdata[i,1]]=1
   data[thisdata[i,1],thisdata[i,0]]=1
'''
   label=[]
   thisdata=sci.loadmat('glass.mat')
   thisdata=thisdata['glass']
   label = np.array([thisdata[:,thisdata.shape[1]-1]])-1
   data0=thisdata[:,0:thisdata.shape[1]-1]
   data=np.zeros((data0.shape[0], data0.shape[0]))
   for i in range(data0.shape[0]):
      for j in range(data0.shape[0]):
         data[i,j]=math.exp(-np.linalg.norm(data0[i,:]-data0[j,:],ord=2))
   del data0
   del thisdata
   data=pre.minmax_scale(data)
   for i in range(data.shape[0]):
      for j in range(data.shape[0]):
         if data[i,j]>=0.3:
            data[i,j]=1
         else:
            data[i,j]=0
   result=MNMF.MNMF(data,4)
   FM, ARI, Phi, Hubert, K, RT, precision, recall, F1,NMI,Q=Evaluation.evaluation(data,result,label)
   end = time.time()
   print('此算法在该数据集上的结果为:FM=', FM)
   print('此算法在该数据集上的结果为:ARI=', ARI)
   print('此算法在该数据集上的结果为:Phi=', Phi)
   print('此算法在该数据集上的结果为:Hubert=', Hubert)
   print('此算法在该数据集上的结果为:K=', K)
   print('此算法在该数据集上的结果为:RT=', RT)
   print('此算法在该数据集上的结果为:precision=', precision)
   print('此算法在该数据集上的结果为:recall=',recall)
   print('此算法在该数据集上的结果为:F1=', F1)
   print('此算法在该数据集上的结果为:NMI=', NMI)
   print('此算法在该数据集上的模块度为:Q=',Q)
   print('此算法在该数据集上的运行时间为:', end-start)
