#-*- codeing = utf-8 -*-
#@Time : 2023/9/12 19:47
#@Author :辛永杰
#@File : jiont9.py
#@Sofeware : PyCharm
import torch
import numpy as np
from sklearn.cluster import KMeans
import evaluation
from metrics import cal_clustering_metric, re_newpre
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import hdf5storage
import math
from scipy.linalg import solve_sylvester
import scipy.io as scio
def Adacode(q0,p0):
    views = q0.shape[0]
    pq = 0
    qq = 0
    K = torch.zeros([views,])
    v = torch.zeros([views,])
    u1 = torch.zeros([views,])
    t1 = torch.zeros([views, ])
    ft = 1
    u = 0
    f = 1
    for i in range(views):
        pq = pq + torch.diag(q0[i,:,:].cuda().matmul(p0.cuda().t())).sum()/torch.diag(q0[i,:,:].cuda().matmul(q0[i,:,:].cuda().t())).sum()
        qq = qq + 1/torch.diag(q0[i,:,:].cuda().matmul(q0[i,:,:].cuda().t())).sum()
    for j in range(views):
        K[j] = (torch.diag(q0[j,:,:].cuda().matmul(p0.cuda().t())).sum()/torch.diag(q0[j,:,:].cuda().matmul(q0[j,:,:].cuda().t())).sum())+(1-pq.cuda())/(torch.diag(q0[j,:,:].cuda().matmul(q0[j,:,:].cuda().t())).sum()*qq)
    while abs(f)>1e-10:
        for j in range(views):

            u1[j] = u/(torch.diag(q0[j,:,:].matmul(q0[j,:,:].t())).sum()*qq) - K[j]

            v[j] = 1/(torch.diag(q0[j, :, :].matmul(q0[j, :, :].t())).sum() * qq)
        f = torch.ge(u1,0).mul(u1).sum() - u
        v1 = torch.ge(u1,0).mul(v).sum() - 1
        u = u - f/v1
        ft = ft + 1
        for j1 in range(views):
            t1[j1] = K[j1] - u / (torch.diag(q0[j1, :, :].matmul(q0[j1, :, :].t())).sum() * qq)

        if ft>1000:
            break
    return torch.ge(t1, 0).mul(t1)

def update_E(E,a,p):

    for e1 in range(E.shape[0]):
        for e2 in range(E.shape[2]):
            if torch.square(E[e1,:,e2]).sum().sqrt() > a/p:
                E[e1,:,e2] = (1- ((a/p)/torch.square(E[e1,:,e2]).sum().sqrt()))*E[e1,:,e2]
            else:
                E[e1,:,e2] = 0
    return E

"""def update_E(E,a,p,len_H):
    E_cat = torch.cat((E[0,:,:],E[1,:,:]),0)
    views_num = E.shape[0]
    if views_num>1:
        for e1 in range(2,E.shape[0]):
            E_cat = torch.cat((E_cat,E[e1,:,:]),0)
    #for e1 in range(E.shape[0]):
    for e2 in range(E_cat.shape[1]):
        if torch.square(E_cat[ :, e2]).sum().sqrt() > a / p:
            E_cat[ :, e2] = (1 - ((a / p) / torch.square(E_cat[:, e2]).sum().sqrt())) * E_cat[:, e2]
        else:
            E_cat[ :, e2] = 0
        begin_len = 0
        for e9 in range(views_num):

            E[e9,:,e2] = E_cat[begin_len:(begin_len + len_H), e2]
            begin_len = begin_len + len_H

    return E"""


def update_W(W,L,X,ym2,ym4,H,B,p,dataset,E,Y,S_normal,Weight):
    for e1 in range(len(dataset)):
        D = torch.diag_embed(1/(2*torch.sqrt(torch.square(W[dataset[e1]]).sum(dim=1))))
        W[dataset[e1]] = torch.inverse(p*torch.matmul(X[dataset[e1]].cuda(),X[dataset[e1]].cuda().t()) + ym2*(X[dataset[e1]].cuda().matmul((S_normal.t().cuda().matmul(L.cuda())).matmul(S_normal.cuda()))).matmul(X[dataset[e1]].cuda().t())+ym4*D.cuda()).matmul(p*(torch.matmul(X[dataset[e1]].cuda(),H[dataset[e1]].cuda())).matmul(B[e1,:,:].cuda().t())+p*(X[dataset[e1]].cuda()).matmul(E[e1,:,:].cuda().t())-(X[dataset[e1]].cuda()).matmul(Y[e1,:,:].cuda().t()))
    return W

def update_B(W,X,B,H,dataset,E,Y,p):
    for e1 in range(len(dataset)):
        u,s,v = torch.svd((W[dataset[e1]].cuda().t().matmul(X[dataset[e1]].cuda())+Y[e1,:,:].cuda()/p-E[e1,:,:].cuda()).matmul(H[dataset[e1]].cuda()))
        B[e1,:,:] = u.matmul(v.t())
    return B


#zhen
def sparseGraph(G,dataset,Weight):
    L = torch.zeros([len(dataset),G.shape[1],G.shape[2]])
    L_weight= torch.zeros([ G.shape[1], G.shape[2]])
    I = torch.eye(G.shape[1])

    for g in range(len(dataset)):
        Dv = torch.diag_embed(1/G[g,:,:].cuda().sum(dim=1).sqrt())
        L[g,:,:] = torch.subtract(I.cuda(),(Dv.cuda().matmul((G[g,:,:].cuda()+G[g,:,:].cuda().t())/2)).matmul(Dv.cuda()))
        L_weight = L_weight.cuda() + Weight[g].cuda() * L[g,:,:].cuda()
    return L_weight

"""def update_H(B,W,X,H,dataset,Y,E,S,weight,p,e3,e_H):
    er =1e-10
    I = torch.eye(H.shape[0])
    #w and L
    H_sum = torch.zeros([H.shape[0],H.shape[1]])
    H = torch.ge(H,0).cuda().mul(H.cuda())
    for e2 in range(len(dataset)):# + I.cuda()
        #D_H = torch.diag_embed(1 / (2 * torch.sqrt(torch.square(H[dataset[e2]]).sum(dim=1))))+ e_D*D_H.cuda()

        H_sum = H_sum.cuda() + p * ((X[dataset[e2]].cuda().t()).matmul(W[dataset[e2]].cuda())).matmul(B[e2, :, :].cuda()) + Y[e2, :,:].cuda().t().matmul(B[e2, :, :].cuda()) - p * E[e2, :, :].cuda().t().matmul(B[e2, :, :].cuda())
    H = torch.inverse(2 * e3 *I.cuda()- e3 * (S.cuda()+S.cuda( ).t()) + len(dataset) * p * I.cuda() + 2*e_H*I.cuda()).matmul(2 * e_H * H.cuda()+H_sum)
    return H"""

"""def update_H(B,W,X,H,dataset,Y,E,S,v,p,e3,e_H):
    er =1e-10
    I = torch.eye(H[dataset[0]].shape[0])
    #w and L
    v1 = torch.ge(v, 0).cuda().mul(v.cuda())
    for e2 in range(len(dataset)):# + I.cuda()
        #D_H = torch.diag_embed(1 / (2 * torch.sqrt(torch.square(H[dataset[e2]]).sum(dim=1))))+ e_D*D_H.cuda()
        H[dataset[e2]] = torch.ge(H[dataset[e2]],0).cuda().mul(H[dataset[e2]].cuda())
        H[dataset[e2]] = torch.inverse(2 * e3 *I.cuda()- e3 *  (S.cuda()+S.cuda( ).t()) + p * I.cuda() + 2*e_H*I.cuda()).matmul(2*e_H*H[dataset[e2]].cuda()+p * ((X[dataset[e2]].cuda().t()).matmul(W[dataset[e2]].cuda())).matmul(B[e2, :, :].cuda()) + Y[e2, :,:].cuda().t().matmul(B[e2, :, :].cuda()) - p * E[e2, :, :].cuda().t().matmul(B[e2, :, :].cuda()))
    return H"""

def update_H(B,W,X,H,dataset,Y,E,S,v,p,e3,e_H):
    er =1e-10
    I = torch.eye(H[dataset[0]].shape[0])
    #w and L
    v1 = torch.ge(v, 0).cuda().mul(v.cuda())
    for e2 in range(len(dataset)):# + I.cuda()
        #D_H = torch.diag_embed(1 / (2 * torch.sqrt(torch.square(H[dataset[e2]]).sum(dim=1))))+ e_D*D_H.cuda()
        """H[dataset[e2]] = torch.ge(H[dataset[e2]],0).cuda().mul(H[dataset[e2]].cuda())
        H[dataset[e2]] = torch.inverse(2 * e3 *I.cuda()- e3 *  (S.cuda()+S.cuda( ).t()) + p * I.cuda() + 2*e_H*I.cuda()).matmul(2*e_H*H[dataset[e2]].cuda()+p * ((X[dataset[e2]].cuda().t()).matmul(W[dataset[e2]].cuda())).matmul(B[e2, :, :].cuda()) + Y[e2, :,:].cuda().t().matmul(B[e2, :, :].cuda()) - p * E[e2, :, :].cuda().t().matmul(B[e2, :, :].cuda()))"""
        u, s, v = torch.svd(p*(X[dataset[e2]].cuda().t()).matmul(W[dataset[e2]].cuda()).matmul(B[e2, :, :].cuda())+Y[e2, :,:].cuda().t().matmul(B[e2, :, :].cuda()) - p * E[e2, :, :].cuda().t().matmul(B[e2, :, :].cuda())+2*e_H*v1)
        H[dataset[e2]] = u.matmul(v.t())
    return H

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result

def update_KS(S,dataset):

    S1 =  (S.sum(dim=0))/len(dataset) - S.sum(dim=0).sum()/(len(dataset)*S.shape[1]) +1/S.shape[1]

    vmin = torch.min(S1)
    if vmin <0:
        lambda_m = 0
        ft = 1
        f = 1
        while abs(f)>1e-10:
            S2 = lambda_m - S1
            v2 = torch.ge(S2,0)
            g = v2.sum()/S.shape[1]-1
            f = (torch.ge(S2,0).mul(S2)).sum()/S.shape[1] -lambda_m
            lambda_m = lambda_m - f/g
            ft = ft +1
            if ft >300:
                break
        Sv = torch.ge(-S2, 0).mul(-S2)
    else:
        Sv = S1
    return  Sv

def update_S(S,v,W,X,dataset,ym2,ym1,ym,Weight,S0,e3):
    p = torch.zeros([len(dataset) , v.shape[0], v.shape[0]])

    for t in range(len(dataset)):
        e1= distance(v.cuda().t(),v.cuda().t())
        e2 = Weight[t] * distance((W[dataset[t]].cuda().t().matmul(X[dataset[t]].cuda())).matmul(S0.cuda().t()),(W[dataset[t]].cuda().t().matmul(X[dataset[t]].cuda())).matmul(S0.cuda().t()))
        p[t,:,:] = -(ym2/(2*ym)) * e2.cuda() - (ym1/(2*len(dataset)*ym))* e1.cuda()

    for e5 in range(e2.shape[1]):
        S[e5,:] = update_KS(p[:,e5,:],dataset)
    return S


def main_opt(X,dataset,W,B,S,ym,H,n_cluster,p,ym2,ym4,iters,Y,E,a,e2,Weight,e_H):
    vmax = 1e8
    q = 1e-6#1e-6
    tr_sum_min = 1e-10
    Losslist = []
    while(True):
        ym1 = 1
        Lh,S_normal = reconstruct_L(S)
        W = update_W(W,Lh,X,ym2,ym4,H,B,p,dataset,E,Y,S_normal,Weight)#ym3L ym4D  H ym2 v ym1 s ym
        B = update_B(W,X,B,H,dataset,E,Y,p)
        Weight= reconstruct_Lw(W,X,dataset,H[dataset[0]].shape[0],num_neighbors,n_cluster,Weight,S)

        iter = 1
        L,_ = reconstruct_L(S)
        e, v = torch.linalg.eigh(L)
        sorted, indices1 = torch.sort(e, descending=False)
        v0 = v[:,indices1[:n_cluster]]
        H = update_H(B, W, X, H, dataset,Y,E,S,v0,p,e2,e_H)
        S0 = S_normal

        while (iter < iters):
            S = update_S(S,v0,W,X,dataset,ym2,ym1,ym,Weight,S0,e2)
            L,S_normal = reconstruct_L(S)
            e, v1 = torch.linalg.eigh(L)
            sorted, indices1 = torch.sort(e, descending=False)
            L2 = e[indices1[:n_cluster]].sum()
            v = v1[:,indices1[:n_cluster]]
            if L2<tr_sum_min:
                ym1 = ym1 / 2
            else:
                ym1 = ym1 * 2
                v0 = v
                S0 = S_normal
            iter = iter + 1
        loss1 = []
        for e3 in range(len(dataset)):
            E[e3,:,:] = torch.subtract(W[dataset[e3]].cuda().t().matmul(X[dataset[e3]].cuda()),B[e3,:,:].cuda().matmul(H[dataset[e3]].cuda().t()))+Y[e3,:,:].cuda()/p
        E = update_E(E, a, p)
        #E = update_E(E, a, p,W[dataset[0]].shape[1])
        for y in range(len(dataset)):
            loss1.append(max((torch.subtract(W[dataset[y]].cuda().t().matmul(X[dataset[y]].cuda()),B[y].cuda().matmul(H[dataset[y]].cuda().t()))-E[y,:,:].cuda()).square().sqrt().sum(dim=0)))
            Y[y,:,:] = Y[y,:,:].cuda() + p * (torch.subtract(W[dataset[y]].cuda().t().matmul(X[dataset[y]].cuda()),B[y].cuda().matmul(H[dataset[y]].cuda().t()))-E[y,:,:].cuda())
        print(max(loss1))
        Losslist.append(max(loss1))
        if max(loss1) < q:
            print(max(loss1))
            break
        else:
            p = min(2 * p, vmax)
    return S,Losslist

def reconstruct_S(X,dataset,size,num_neighbors):
    A = torch.zeros([len(dataset), X[dataset[0]].shape[1],X[dataset[0]].shape[1]])
    S = torch.zeros([X[dataset[0]].shape[1],X[dataset[0]].shape[1]])
    for a in range(len(dataset)):
        AA = distance(X[dataset[a]].cuda(),X[dataset[a]].cuda())

        sorted_distances, _ = AA.sort(dim=1)
        top_k = sorted_distances[:, num_neighbors]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

        sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors].cuda(), dim=1)
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))

        T = torch.ge(top_k.cuda() - AA.cuda(),0).mul(top_k.cuda() - AA.cuda())

        A[a,:,:] = torch.div(T.cuda(), num_neighbors * top_k.cuda() - sum_top_k.cuda())
        S = S.cuda() + A[a,:,:].cuda()

    S = S/len(dataset)
    S_sum = torch.t(S.cuda().sum(dim=1).repeat(size, 1))
    S = S.cuda().div(S_sum.cuda())
    return S



def reconstruct_Lw(W,X,dataset,size,num_neighbors,n_cluster,Weight,S):
    I = torch.eye(X[dataset[0]].shape[1])
    A = torch.zeros([len(dataset), X[dataset[0]].shape[1],X[dataset[0]].shape[1]])
    AA = torch.zeros([len(dataset), X[dataset[0]].shape[1], X[dataset[0]].shape[1]])
    A_I = torch.zeros([len(dataset), X[dataset[0]].shape[1], X[dataset[0]].shape[1]])
    L_A_I = torch.zeros([len(dataset), X[dataset[0]].shape[1], X[dataset[0]].shape[1]])
    #Weight = torch.zeros([len(dataset),])
    softmax1 = torch.nn.Softmax(dim=1)
    AA1 = torch.zeros([len(dataset), X[dataset[0]].shape[1], X[dataset[0]].shape[1]])
    for a in range(len(dataset)):
        AA[a,:,:] = distance(W[dataset[a]].cuda().t().matmul(X[dataset[a]].cuda()),W[dataset[a]].cuda().t().matmul(X[dataset[a]].cuda()))
        #A[a,:,:] = softmax1(-AA[a,:,:])
        #Weight[a] = 1/(2*torch.subtract(A[a,:,:],S).square().sum().sqrt())
        sorted_distances, _ = AA[a,:,:].sort(dim=1)
        top_k = sorted_distances[:, num_neighbors]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10
        # n*n
        sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors].cuda(), dim=1)
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))

        T = torch.ge(top_k.cuda() - AA[a,:,:].cuda(),0).mul(top_k.cuda() - AA[a,:,:].cuda())
        # 为正数
        A[a,:,:] = torch.div(T.cuda(), num_neighbors * top_k.cuda() - sum_top_k.cuda())
        #A[a, :, :] = softmax1(-AA[a,:,:])

        AA1[a,:,:] = (A[a,:,:].cuda()+A[a,:,:].cuda().t())/2
        D_I = torch.diag_embed(AA1[a,:,:].cuda().sum(dim=1)) + I.cuda()

        D_A_I = torch.diag_embed(1/(AA1[a, :, :].cuda()+ I.cuda()).sum(dim=1).sqrt())
        A_I[a,:,:] = AA1[a,:,:].cuda()+I.cuda()

        L_A_I[a,:,:] = (D_A_I.cuda().matmul((D_I.cuda()+A_I[a,:,:].cuda())/2)).matmul(D_A_I.cuda())

        e, v = torch.linalg.eigh(L_A_I[a,:,:])
        sorted, indices1 = torch.sort(e, descending=True)
        Weight[a] = e[indices1[:n_cluster]].cuda().sum()

    softmax = torch.nn.Softmax()
    Weights = softmax(Weight)

    #Lw = sparseGraph(A, dataset, Weights)
    return Weights

def reconstruct_L(S):
    I = torch.eye(S.shape[0])
    print(num_neighbors)
    sorted_distances, _ = S.sort(dim=1,descending = True)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(S.shape[0], 1))
    S_xin = torch.ge(S.cuda(),top_k.cuda()).mul(S.cuda())
    S1 = S_xin.cuda() + I.cuda()
    SS = (S+S.t())/2
    P = torch.diag_embed(1/S.cuda().sum(dim=1).sqrt())
    L = torch.subtract(I.cuda(),(P.cuda().matmul(SS.cuda())).matmul(P.cuda()))
    S_normal = S1.cuda()#(P1.cuda().matmul((S1.cuda()+S1.cuda().t())/2)).matmul(P1.cuda())

    return L,S_normal

def load_data(name):
    path = '{}.mat'.format(name)
    data = hdf5storage.loadmat(path)
    print(data)
    # print(data.shape)
    labels = data['labels']
    print(labels.shape)
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['data']
    # print(X.shape)
    print(X[1][0].shape)
    label_np = np.zeros(labels.shape[0], )

    data_np = {}
    for j in range(X.shape[0]):
        data_np["{}".format(j)] = X[j][0]
        print(data_np["{}".format(j)].shape)
    for g in range(label_np.shape[0]):
        label_np[g] = labels[g]
    print(label_np)

    return data_np, label_np

"""def load_data(name):
    path = '{}.mat'.format(name)
    data = scio.loadmat(path)
    print(data)
    # print(data.shape)
    labels = data['Y']
    print(type(labels))
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    # print(X.shape)
    print(X[0][2].shape)
    label_np = np.zeros(labels.shape[0], )

    data_np = {}
    for j in range(X.shape[1]):
        data_np["{}".format(j)] = X[0][j]
        print(data_np["{}".format(j)].shape)
    for g in range(label_np.shape[0]):
        label_np[g] = labels[g]
    print(label_np)

    return data_np, label_np"""
"""def load_data(name):
    listdata = {}
    path = '{}.mat'.format(name)
    #path = 'data/ORL.mat'
    data = scio.loadmat(path)
    print(data)
    #print(data)
    labels = data['gt']
    print(labels)
    #print(labels)
    X = data['X']
    print(X[0][1].shape)
    for i in range(X.shape[1]):
        #X[i][0] /= np.max(X[i][0])
        #X[i][0] = normalize(X[i][0])
        listdata["{}".format(i)] = X[0][i].T
        print(listdata["{}".format(i)])
    #X = X.astype(np.float32)
    listlabel = np.zeros((labels.shape[0]))
    for j in range(labels.shape[0]):
        listlabel[j] = labels[j][0]
    print(listlabel)
    return listdata,listlabel"""
#bbcsport
"""def load_data(name):
    listdata = {}
    path = '{}.mat'.format(name)
    print(path)
    data = scio.loadmat(path)
    print(data)

    labels = data['Y']
    #print(X.shape)
    print(labels.shape)
    listlabel = {}
    Y_label = np.zeros((int(labels.shape[0]), ))

    for i in range(data['X'].shape[1]):
        listdata["{}".format(i)] = data['X'][0][i].toarray()
        print(type(listdata["{}".format(i)]))
        #print((listdata["{}".format(i)]))

    for j in range(labels.shape[0]):
        Y_label[j] = labels[j][0]

    return listdata, Y_label"""

# 3sources2
"""def load_data(name):
    listdata = {}
    path = '{}.mat'.format(name)
    data = scio.loadmat(path)
    print(data)

    labels = data['truelabel']
    print(data['data'].shape[1])
    print(labels[0][1].shape)
    listlabel = {}
    Y_label = np.zeros((int(labels[0][1].shape[1]), ))

    for i in range(data['data'].shape[1]):
        listdata["{}".format(i)] = data['data'][0][i]#X[0][i]
        print(listdata["{}".format(i)].shape)
        print((listdata["{}".format(i)]))
    #print("jie")
    print(labels[0][1].shape)
    print(type(listdata["{}".format(0)]))
    for j in range(labels[0][1].shape[1]):
        Y_label[j] = labels[0][1][0][j]

    return listdata, Y_label"""

def clustering(S,labels,SC=True):
    n_clusters = np.unique(labels).shape[0]
    if SC:
        degree = torch.sum(S, dim=1).pow(-0.5)
        L = (S * degree).t() * degree
        L = L.cpu()
        _, vectors = torch.linalg.eigh(L)
        indicator = vectors[:, -n_clusters:]
        indicator = indicator / (indicator.norm(dim=1) + 10 ** -10).repeat(n_clusters, 1).t()
        indicator = indicator.detach().numpy()
        km = KMeans(n_clusters=n_clusters).fit(indicator)
        prediction = km.predict(indicator)

        acc, nmi = cal_clustering_metric(labels, prediction)

        colors = list(mcolors.CSS4_COLORS.keys())
        predictiondata_max = indicator
        tsne = TSNE(n_components=2, random_state=0)
        result = tsne.fit_transform(predictiondata_max)
        listdata = [t for t in range(np.unique(labels).shape[0])]
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 16,
                 }
        for i in listdata:
            plt.scatter(result[np.sort(prediction) == i, 0], result[np.sort(prediction) == i, 1], s=6,c=colors[int(i) + 10])  # ,label = "%d"%i

        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("100leaves datasets Clustering", fontsize=16)
        #print(type(prediction_max.reshape(prediction_max.shape[0], )))
        #plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0, prop=font1)
        print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
    print(acc, nmi,evaluation.clustering(prediction, labels))
    return acc, nmi,evaluation.clustering(prediction, labels)


"""def update_W(W,L,X,ym2,ym4,H,B,p,dataset,E,Y):

    for e1 in range(len(dataset)):
        I= torch.eye(X[dataset[e1]].shape[0])
        #D = torch.diag_embed(1/(2*torch.sqrt(torch.square(W[dataset[e1]]).sum(dim=1))))
        W[dataset[e1]] = torch.inverse(p*torch.matmul(X[dataset[e1]].cuda(),X[dataset[e1]].cuda().t())+ym2*(X[dataset[e1]].cuda().matmul(L.cuda())).matmul(X[dataset[e1]].cuda().t())+ym4*I.cuda()).matmul(p*(torch.matmul(X[dataset[e1]].cuda(),H[dataset[e1]].cuda())).matmul(B[e1,:,:].cuda().t())+p*(X[dataset[e1]].cuda()).matmul(E[e1,:,:].cuda().t())-(X[dataset[e1]].cuda()).matmul(Y[e1,:,:].cuda().t()))
    return W"""
#ym=1,ym2=0.1,ym4=10.0,e2=1,a=0.1
if __name__ == '__main__':
    p = 1e-3#1e-3,
    #ym=1,ym2=0.1,ym4=0.1,e2=10.0,a=0.1,num_neighbors=10,n1=64,e_H=0.01
    #ym = 1, ym2 = 0.1, ym4 = 0.1, e2 = 10.0, a = 0.1, num_neighbors = 10, n1 = 64, e_H = 0.01
    ymlist = [1]#1e-2,
    #ym2 = 1#1e-1,1,1e1,1e2
    ym2list = [1e-1,1e1]#1e-1,1,1e1
    #ym4 = 1#1e-2,1e-1,1,1e1,1e2
    ym4list = [1e1,1e2]#1e-1,1e1
    alist = [0.1,0.9]
    e2list =[1e1,1e2]#,
    e_Hlist = [1e-2]
    num_neighborslist = [20]#5,10,15,
    n1list = [8]#,16,32,64,128
    iters = 10
    dataset = ["0","1","2","3","4","5"]#
    path = "Caltech101-20"# #"MSRCV1""ORL_mtv"'COIL20-3v''COIL20-3v'"YaleA_3view"1"bbcsport","Caltech101-7"
    [data, labels] = load_data(path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data1 = {}
    for j in range(len(dataset)):
        data1[dataset[j]] = torch.zeros([data[dataset[j]].shape[0],data[dataset[j]].shape[1]])
    for e in dataset:
        data1[e] = torch.Tensor(data[e].astype(float)).to(device).t()
        print(data1[e].shape)
    """for e4 in dataset:
        min_vals, _ = torch.min(data1[e4], dim=1, keepdim=True)
        max_vals, _ = torch.max(data1[e4], dim=1, keepdim=True)

        # 最小-最大缩放，将x的范围缩放到[0, 1]
        data1[e4] = ((data1[e4] - min_vals) / (max_vals - min_vals)).t()"""
    n_clusters = np.unique(labels).shape[0]
    #S = reconstruct_init_S(data1,dataset,data1[dataset[0]].shape[1],num_neighbors)
    Weight = torch.zeros([len(dataset), ])
    for w in range(Weight.shape[0]):
        Weight[w] = 1/len(dataset)
    for ym in ymlist:
        for ym2 in ym2list:
            for ym4 in ym4list:
                for e2 in e2list:
                    for a in alist:
                        for num_neighbors in num_neighborslist:
                            for n1 in n1list:
                                for e_H in e_Hlist:
                                    with open("3sources_jiont.txt", 'a') as f:
                                        f.write(
                                            '-----ym={},ym2={},ym4={},e2={},a={},num_neighbors={},n1={},e_H={}'.format(ym, ym2,
                                                                                                                ym4, e2,
                                                                                                                a,
                                                                                                                num_neighbors,
                                                                                                                n1,e_H))
                                        S = reconstruct_S(data1, dataset, data1[dataset[0]].shape[1], num_neighbors)
                                        t1 = torch.zeros([len(dataset), ])

                                        B = torch.zeros([len(dataset), n1, n_clusters])
                                        #H = torch.rand([data1[dataset[0]].shape[1], n_clusters])
                                        H = {}
                                        W = {}
                                        Y = torch.zeros([len(dataset), n1, data1[dataset[0]].shape[1]])
                                        E = torch.zeros([len(dataset), n1, data1[dataset[0]].shape[1]])
                                        for w1 in range(len(dataset)):
                                            H[dataset[w1]] = torch.rand([data1[dataset[w1]].shape[1], n_clusters])
                                            W[dataset[w1]] = torch.rand([data1[dataset[w1]].shape[0], n1])
                                        B_init = update_B(W, data1, B, H, dataset, E, Y, p)
                                        S, Losslist = main_opt(data1, dataset, W, B_init, S, ym, H, n_clusters, p, ym2,
                                                               ym4, iters, Y, E,
                                                               a, e2, Weight, e_H)  # ym3L ym4D  H ym2 v ym1 s ym
                                        # clustering(S, labels)
                                        f.write("{}\n".format(clustering(S, labels)))
                                        f.close()
                                        """plt.figure()

                                        plt.xlabel('Iter', fontsize=16)  # x轴标签
                                        plt.ylabel('Error', fontsize=16)  # y轴标签
                                        x = range(1, len(Losslist), 1)
                                        plt.xticks(x, fontsize=12)
                                        plt.yticks(fontsize=12)

                                        plt.plot(range(len(Losslist)), Losslist, linewidth=2.5, linestyle="solid",
                                                 marker='.')

                                        plt.legend()
                                        plt.title('BBCsport database', fontsize=16)
                                        plt.show()"""
