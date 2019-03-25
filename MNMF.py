import numpy as np
import math
def MNMF(data,k):
    '''
    此算法执行的是M-NMF算法，Wang, Xiao, et al. "Community preserving network embedding."
    Thirty-First AAAI Conference on Artificial Intelligence. 2017,203-209.
    输入:  data:图的邻接矩阵n*n  k:为社团的数目   输出:result为聚类的结果矩阵为1*n的向量每一列表示每一个样本的聚类类标签
    '''
    lamda=1000000000
    eta=5
    a=0.5
    b=0.5
    m=100
    N_MAX=100
    H=np.random.rand(data.shape[0],k)#初始化结果矩阵
    e=0#边的数目
    for i in range(data.shape[0]):
        #H[i,np.random.randint(0,k,1)]=1
        e=e+sum(data[i,:])#计算图中边的数目
    e=e/2
    result=[]#初始化结果矩阵
    S1=data#一阶近似
    S2=np.zeros((data.shape[0],data.shape[0]))#初始化二阶矩阵
    B1=np.zeros((data.shape[0],data.shape[0]))#初始化B1矩阵
    for i in range(data.shape[0]):#计算二阶近似矩阵的值
        for j in range(data.shape[1]):
            S2[i,j]=np.dot(S1[i,:],S1[j,:])/(np.linalg.norm(S1[i,:])*np.linalg.norm(S1[j,:])+0.00001)
            B1[i,j]=sum(data[i,:])*sum(data[j,:])/(2*e)
    S=S1+eta*S2#获得计算的近似矩阵
    #初始化迭代求解的矩阵
    U=np.random.rand(data.shape[0],m)
    M=np.random.rand(data.shape[0],m)
    C=np.random.rand(k,m)
    #临时值的保存
    Ut=U
    Mt=M
    Ct=C
    Ht=H
    #开始迭代过程
    f=float("inf")#目标函数的最优值
    ft=0#当前目标函数的值
    for i in range(N_MAX):#更新各个参数
        Mt=M
        M=M*(np.dot(S,U)/np.dot(np.dot(M,U.transpose()),U)+0.0001)#更新M矩阵
        Ut=U
        U=U*((np.dot(S.transpose(),M)+a*np.dot(H,C))/(np.dot(U,(np.dot(M.transpose(),M)+a*np.dot(C.transpose(),C)))+0.0001))#更新U矩阵
        Ct=C
        C=C*(np.dot(H.transpose(),U)/(np.dot(np.dot(C,U.transpose()),U)+0.0001))#更新C矩阵
        deta=2*b*np.dot(B1,H)*2*b*np.dot(B1,H)+16*lamda*np.dot(np.dot(H,H.transpose()),H)*(2*b*np.dot(data,H)+2*a*np.dot(U,C.transpose())+(4*lamda-2*a)*H)
        Ht=H
        H=H*np.sqrt((np.sqrt(deta)-2*b*np.dot(B1,H))/(8*lamda*np.dot(np.dot(H,H.transpose()),H)+0.0001))#更新H矩阵
        ft=math.pow(np.linalg.norm(S-np.dot(M,U.transpose()),'fro'),2)
        if ft<f:#当前求解的结果符合条件,保存结果,继续进行迭代
            f=ft
            Ut=U
            Mt=M
            Ct=C
            Ht=H
        else:#当前求解的结果没有降低损失函数值,上一次迭代结果为最终的结果
            U=Ut
            M=Mt
            C=Ct
            H=Ht
            break
    #H矩阵保存的即是最终的聚类结果,采用最大值归属原则
    result=np.argmax(H,axis=1)
    result=np.array([result])
    return result

