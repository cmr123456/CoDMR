# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader  
import torch.optim as optim
import pickle
import random
import numpy as np
import time
import torch 

import scipy.sparse as sp
from scipy.sparse import csr_matrix
import argparse
import os

from itertools import product
from ToolScripts.TimeLogger import log
from ToolScripts.tools import sparse_mx_to_torch_sparse_tensor, sampleLargeGraph, sampleHeterLargeGraph
from ToolScripts.BPRData import BPRData  
import ToolScripts.evaluate as evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model  import MODEL
from args  import make_args
from ToolScripts.TestData import TstData 
modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")
import warnings
import math
from torch.nn.functional import softplus
# import dgl
warnings.filterwarnings('ignore')
isLoadModel = False
LOAD_MODEL_PATH = r"SR-HAN_Yelp_1599990303_hide_dim_8_layer_dim_[8,8,8]_lr_0.05_reg_0.02_topK_10_lambda1_0_lambda2_0"


class Hope():
    def __init__(self, args, data,uiMat,iciMat,icaiMat):
        self.args = args 
        self.trainMat = data ['trn_mat']
        self.testData = data ['tst_mat'] 
        self.iciMat = (iciMat!= 0) * 1#itemMat ( self.itemDistanceMat !=0) * 1#
        self.icaiMat = (icaiMat!= 0) * 1
        self.uiMat = (uiMat != 0) * 1  
        self.userNum, self.itemNum = self.trainMat.shape
        train_coo = self.trainMat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        self.ui_u =  train_u
        self.ui_v =  train_v
        assert np.sum(train_r == 0) == 0
        train_data = np.hstack((train_u.reshape(-1,1),train_v.reshape(-1,1))).tolist()
        testData = self.testData.tocoo()
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, 1, True) #num_negtive samples
        test_dataset      = TstData(testData.tocoo(), self.trainMat)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0) 
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=self.args.testB, shuffle=False,num_workers=0) #test batch=1024000
        train_coo = self.trainMat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        self.train_u=train_u
        self.train_v=train_v
        
       
        # data for plot 
        self.train_losses = []
        self.test_hr = []
        self.test_ndcg = []

        ####Pre-trained user and item profile features
        with open('/raid2/Thirdwork/DiffRec/data/yelp/itm_emb_np.pkl', 'rb') as f:
            item_emb_text = pickle.load(f)
        with open('/raid2/Thirdwork/DiffRec/data/yelp/usr_emb_np.pkl', 'rb') as f:
            user_emb_text = pickle.load(f)
        self. item_emb_text =  torch.Tensor(item_emb_text).cuda()
        self. user_emb_text =  torch.Tensor(user_emb_text).cuda()
        
        
    def prepareModel(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONSEED"] = str(args.seed)
       
        # torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.enabled = True
      
        #metapath encoder model
        self.model = MODEL( self.args,
                           self.userNum,
                           self.itemNum,
                           self.iciMat, self.icaiMat,  self.uiMat, self. item_emb_text,  self. user_emb_text,
                           self.args.hide_dim,
                           self.args.uiLayers,
                           self.args.au_uiLayers).cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.lr)
        # params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
       

    def predictModel(self,user, pos_i, neg_j, isTest=False):#pos_i 234047 32 
        if isTest:
            pred_pos = torch.matmul(user,pos_i.transpose(0,1))#pred_pos = t.sum(user * pos_i, dim=1) 512 234047 
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            
            
            return pred_pos, pred_neg

    def adjust_learning_rate(self):
        # lr = self.lr * (self.args.decay**epoch)
        if self.opt != None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)
                # print(param_group['lr'])

    def getModelName(self):
        title = "SR-HAN" + "_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr +\
        "_hide_dim_" + str(self.args.hide_dim) +\
        "_lr_" + str(self.args.lr) +\
        "_reg_" + str(self.args.reg) +\
        "_topK_" + str(self.args.topk)
        return ModelName

    def saveHistory(self): 
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        ModelName = self.getModelName()

        with open(r'/raid2/Thirdwork/DiffRec/L-DiffRec1/History/' + dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self): 
        ModelName = self.getModelName()
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        savePath = r'/raid2/Thirdwork/DiffRec/L-DiffRec1/Model/' + dataset + r'/' + ModelName + r'.pth'
        params = {
            'model': self.model,
            'epoch': self.curEpoch,
            'args': self.args,
            'opt': self.opt,
            'history':history
            }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def loadModel(self, modelPath):
        checkpoint = t.load(r'/raid2/Thirdwork/DiffRec/L-DiffRec1/Model/' + dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']

        history = checkpoint['history']
        self.train_losses = history['loss']
        self.test_hr = history['hr']
        self.test_ndcg = history['ndcg']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))
        
    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig =0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg
    
   
    def ssl_loss(self, data1, data2,   index):
        ssl_temp =0.65 
        index=t.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
        pos_score  = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = t.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = t.exp(pos_score / ssl_temp)
        all_score  = t.sum(t.exp(all_score / ssl_temp), dim = 1)
        ssl_loss  = (-t.sum(t.log(pos_score / ((all_score))))/(len(index)))
        return ssl_loss
    def trainModel(self,epo):
        epoch_loss = 0
        self.train_loader.dataset.ng_sample() 
        step_num = 0 # count batch num
        epoch_loss_df = 0
        epoch_loss_dftx = 0

        for user, item_i, item_j in self.train_loader: 
            user = user.long().cuda()
            item_i =item_i.long().cuda()
            item_j = item_j.long().cuda()  
            step_num += 1
            userindex=torch.unique(user)
            self.train=True
            indexi = torch.unique(item_i)
            itemindex=torch.unique(torch.cat((item_i,item_j)))
            userindex=torch.unique(user)
            elboloss , self.ui_userEmbed,  self.ui_itemEmbed ,(item1,item2),(user1,user2) = self.model( self.train, user, item_i, item_j, norm=1)
            elboloss, mse,(diffuser,diffuitem) ,(preuser_ii,diffuitem1) = elboloss
            
            ##target predictions
            pred_pos, pred_neg = self.predictModel( self.ui_userEmbed[user],  self.ui_itemEmbed[item_i],  self.ui_itemEmbed[item_j])
            bpr_loss = - nn.LogSigmoid()(pred_pos - pred_neg).sum()  
            regLoss  = (t.norm(self.ui_userEmbed[user])**2 + t.norm( self.ui_itemEmbed[item_i])**2 + t.norm( self.ui_itemEmbed[item_j])**2)  
            epoch_loss += bpr_loss.item()  

            """Construction of prediction loss for denoised and reconstructed features"""
            
            pred_posx, pred_negx = self.predictModel(diffuser[user], diffuitem[item_i],  diffuitem[item_j])
            bpr_loss_diff = - nn.LogSigmoid()(pred_posx - pred_negx).sum()  
            epoch_loss_df += bpr_loss_diff.item()
            regLoss_diff  = (t.norm(diffuser[user])**2 + t.norm(diffuitem[item_i])**2 + t.norm(diffuitem[item_j])**2)  
            loss_diff =  0.95*((bpr_loss_diff + regLoss_diff * self.args.reg ) / self.args.batch) 
            pred_posx, pred_negx = self.predictModel(preuser_ii[user], diffuitem1[item_i],  diffuitem1[item_j])
            bpr_loss_diff = - nn.LogSigmoid()(pred_posx - pred_negx).sum()  
            regLoss_diff  = (t.norm(preuser_ii[user])**2 + t.norm(diffuitem1[item_i])**2 + t.norm(diffuitem1[item_j])**2)  
            loss_diff_ii =  0.95*((bpr_loss_diff + regLoss_diff * self.args.reg ) / self.args.batch) 
            loss_diff = (loss_diff + loss_diff_ii)/2.0

            ###Contrastive Augmentation 
            ssloss = (self.ssl_loss(item1,item2,itemindex) + self.ssl_loss(user1,user2,user))/2.0
            loss = 0.95*((bpr_loss + regLoss * self.args.reg )/self.args.batch) + elboloss * self.args.elbo_w + loss_diff * self.args.di_pre_w  +  mse * self.args.con_fe_w + ssloss * self.args.ssl_reg
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),  max_norm=20, norm_type=2)
            self.opt.step()
        return epoch_loss,epoch_loss_df
    


    def testModel(self,epoch):
        HR=[]
        NDCG=[]
        with t.no_grad():
            self.item_sampled_nodes=np.arange(0,114852)#右开
            self.user_sampled_nodes=np.arange(0,161305)
            self.train=False
            elboloss , self.ui_userEmbed,  self.ui_itemEmbed ,_,_= self.model( self.train, self.item_sampled_nodes, self.item_sampled_nodes, self.item_sampled_nodes, norm=1)

            i=0
            epLoss, epRecall, epNdcg = [0] * 3
            num = self.test_loader.dataset.__len__()
            steps = num // self.args.testB
            for usr, trnMask in  self.test_loader:#512 512 234047 
                i += 1
                usr = usr.long().cuda()
                trnMask = trnMask.cuda()
                allPreds = self.predictModel(self.ui_userEmbed[ usr],  self.ui_itemEmbed, None, isTest=True)
                allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
                _, topLocs = t.topk(allPreds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.test_loader.dataset.tstLocs, usr)
                epRecall += recall
                epNdcg += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
            ret = dict()
            ret['Recall'] = epRecall / num
            ret['NDCG'] = epNdcg / num
            HR  =epRecall / num
            NDCG=epNdcg / num
            return HR,NDCG
       

    def run(self):
        self.prepareModel()
        if isLoadModel:
            self.loadModel(LOAD_MODEL_PATH)
            HR,NDCG = self.testModel()
            log("HR@10=%.4f, NDCG@10=%.4f"%(HR, NDCG))
            return 
        self.curEpoch = 0
        best_hr=-1
        best_ndcg=-1
        best_epoch=-1
        HR_lis=[]
        wait=0
        epochloss=[]
        epochdfloss=[]
        for e in range(args.epochs+1):
            self.curEpoch = e
            # train
            log("**************************************************************")
            epoch_loss ,epoch_loss_df = self.trainModel(e)
            self.train_losses.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f"%(e, args.epochs, epoch_loss))
            epochloss.append(epoch_loss)
            epochdfloss.append(epoch_loss_df)
            # test
            HR, NDCG = self.testModel(e)#,userEmbed, itemEmbed
            self.test_hr.append(HR)
            self.test_ndcg.append(NDCG)
            log("epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f"%(e, args.epochs, HR, NDCG))

            self.adjust_learning_rate()     
            if HR>best_hr:
                best_hr,best_ndcg,best_epoch=HR,NDCG,e
                wait=0
            else:
                wait+=1
                print('wait=%d'%(wait))
            HR_lis.append(HR)
            self.saveHistory()
            if wait==self.args.patience:
                log('Early stop! best epoch = %d'%(best_epoch))
                break
        y1=epochloss
        y2=epochdfloss
        
        x1=range(0,len(y1))
        import matplotlib.pyplot as plt
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, '*-')
        plt.plot(x1, y2, 'o-')
        plt.xlabel('epoch_loss vs. epoches')
        plt.ylabel('epoch_loss')
        plt.show()
        plt.savefig("loss.jpg")
        
        print("*****************************")
        log("best epoch = %d, HR= %.4f, NDCG=%.4f"% (best_epoch,best_hr,best_ndcg)) 
        print("*****************************")   
        print(self.args)
        log("model name : %s"%(self.getModelName()))
if __name__ == '__main__':
    # hyper parameters
    args = make_args()
    print(args)
    dataset = args.dataset

    # train & test data
    with open(r'/raid2/Thirdwork/DiffRec/yelp/dataset.pkl', 'rb') as fs:
        data = pickle.load(fs)
    with open(r'/raid2/Thirdwork/DiffRec/yelp/uiMat.pkl', 'rb') as fs:
        uiMat = pickle.load(fs) 
    with open(r"/raid2/Thirdwork/DiffRec/yelp/ICI.pkl", "rb") as fs:
        iciMat = pickle.load(fs)
    with open(r"/raid2/Thirdwork/DiffRec/yelp/IAI.pkl", "rb") as fs:
        icaiMat = pickle.load(fs)
 

    # model instance
    hope = Hope(args, data,uiMat, iciMat, icaiMat)
    modelName = hope.getModelName()
    print('ModelName = ' + modelName)    
    hope.run()
    
    
    
    
    
    
    
        

    

  

