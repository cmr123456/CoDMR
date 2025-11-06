import torch as t
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import scipy.sparse as sp
import numpy as np
from itertools import product
from torch.nn.functional import softplus
import pdb
# from sklearn import decomposition
from torch.autograd import Variable
import gaussian_diffusioncondit as gd
from conditdenoiser import cdenosier
from Nonconditdenoiser import Nodenoiser

class MODEL(nn.Module):
    def __init__(self,args, userNum, itemNum,iciMat,icaiMat, uiMat,item_emb_text,user_emb_text, hide_dim, uiLayers, au_uiLayers):
        super(MODEL, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.iciMat = iciMat
        self.icaiMat = icaiMat
      
        self.item_emb_text = item_emb_text
        self.user_emb_text = user_emb_text
        self.uiMat = uiMat
        self.hide_dim = hide_dim
        self.uiLayerNums = uiLayers
        self.au_uiLayerNums = au_uiLayers
       

        self.args  = args
        self.steps = args.steps
        self.sampling_steps = args.sampling_steps
        self.ssl_temp = args.ssl_temp

        initializer = nn.init.xavier_uniform_ 

        #user-item graph
        self.gcnLayers = nn.ModuleList()
        for i in range(0, 4):
            self.gcnLayers.append(GCN_layer(hide_dim))

        self.au_gcnLayers = nn.ModuleList()
        for i in range(0, 4):
            self.au_gcnLayers.append(GCN_layer(hide_dim))

        uiadj = self.uiMat[: self.userNum,  self.userNum:]
        uiadj = self.sparse_mx_to_torch_sparse_tensor(uiadj)
        uinorm = ( uiadj.to_dense().sum(1).unsqueeze(1).cuda())
        self.uinorm = uinorm

        uiadj = self.uiMat[: self.userNum,  self.userNum:].T
        uiadj = self.sparse_mx_to_torch_sparse_tensor(uiadj)
        iunorm = ( uiadj.to_dense().sum(1).unsqueeze(1).cuda())
        self.iunorm = iunorm

        device = torch.device("cuda:0" if args.cuda else "cpu")
        if args.mean_type == 'x0':
            mean_type = gd.ModelMeanType.START_X
        elif args.mean_type == 'eps':
            mean_type = gd.ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % args.mean_type)
        self.diffusion = gd.GaussianDiffusion(mean_type,args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)


        device = torch.device("cuda:0" if args.cuda else "cpu")
        self.steps_Non = 6
        if args.mean_typeNon == 'x0':
            mean_type = gd.ModelMeanType.START_X
        elif args.mean_typeNon == 'eps':
            mean_type = gd.ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % args.mean_type)
        self.diffusionNon = gd.GaussianDiffusion(mean_type,args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, self.steps_Non, device).to(device)
        out_dims =  eval(args.out_dims)
        in_dims  =  eval(args.in_dims)[::-1]
        latent_size = in_dims[-1]
        mlp_out_dims = eval(args.mlp_dims) + [latent_size]
        mlp_in_dims = args.hide_dim

        self.cdnmodel  = cdenosier(mlp_in_dims, mlp_out_dims, args.emb_size, time_type="cat", norm=args.norm, act_func=args.mlp_act_func).to(device)
        self.Nonmodel  = Nodenoiser(mlp_in_dims, mlp_out_dims, args.emb_size, time_type="cat", norm=args.norm, act_func=args.mlp_act_func).to(device)
       
        self.item_text_net =nn.Linear(1536, hide_dim, bias=False)
        self.encodecon1 = nn.Sequential(
            nn.Linear(hide_dim, hide_dim),
            nn.ReLU(True),
            nn.Linear(hide_dim, hide_dim))
        self.encodecon2 = nn.Sequential(
            nn.Linear(hide_dim, hide_dim),
            nn.ReLU(True),
            nn.Linear(hide_dim, hide_dim))
       
       
        self.embedding_dict = nn.ModuleDict({
        'uu_emb': torch.nn.Embedding(userNum, hide_dim).cuda(),
        'ii_emb': torch.nn.Embedding(itemNum, hide_dim).cuda(),
        'user_emb': torch.nn.Embedding(userNum , hide_dim).cuda(),
        'item_emb': torch.nn.Embedding(itemNum , hide_dim).cuda(),
        'uinterest_emb' : torch.nn.Embedding(userNum , hide_dim).cuda(),
        })
     
       
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
            'uinterest_emb' : nn.Parameter(initializer(t.empty(userNum, hide_dim//4))),
        })   
        return embedding_dict
  
    
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo() 
   
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def ssl_loss(self, data1, data2,   index,mask):
        index=t.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
        pos_score  = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = t.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = t.exp(pos_score / self.ssl_temp)
        all_score  = t.sum(t.exp(all_score / self.ssl_temp), dim = 1)
        ssl_loss  = (-t.sum(t.log(pos_score / ((all_score)))*mask[index])/(mask.sum()))
        return ssl_loss, all_score

    



    def forward(self,iftraining,user,itemi,itemj,norm = 0):
        item_index=np.arange(0,self.itemNum )#右开
        user_index=np.arange(0,self.userNum)
      
        ui_userembed = self.embedding_dict['user_emb'].weight
        ui_itemembed = self.embedding_dict['item_emb'].weight

        ###
        user_interested = self.embedding_dict['uinterest_emb'].weight
       
        
      
    
       


        #Feature encoding of the target domain
        ui_index = np.array(user_index.tolist() + [ i + self.userNum for i in item_index])
        self.ui_embeddings = t.cat([ui_userembed, ui_itemembed], 0)
        self.all_ui_embeddings = [self.ui_embeddings]
       
        for i in range(self.uiLayerNums):
            layer=self.gcnLayers[i]
            if i == 0:  
                uiEmbeddings0     = layer(self.ui_embeddings, self.uiMat, ui_index)
            else:
                uiEmbeddings0   = layer(uiEmbeddings0,      self.uiMat, ui_index)
            if norm == 1:
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings += [norm_embeddings]
        self.uiEmbedding = t.stack(self.all_ui_embeddings, dim=1)  
        self.uiEmbedding = t.mean(self.uiEmbedding, dim = 1)
        self.ui_userEmbedding, self.ui_itemEmbedding  = t.split((self.uiEmbedding), [self.userNum, self.itemNum])
      

        #Feature encoding of the auxiliary domainc
        
        ##The initial features of products in the auxiliary domain are directly shared
        item_embed0 =  ui_itemembed
        item_embed1 =  ui_itemembed
        #Pretrain user and product profile features
        conditiembed_i = self.item_text_net(self.item_emb_text)
        conditiembed_u = self.item_text_net(self.user_emb_text)
        self.ui_embeddings_text = t.cat([ conditiembed_u,  conditiembed_i], 0)
        self.all_item_embeddings0 = [item_embed0]
        self.all_item_embeddings1 = [item_embed1]
        self.all_ui_embeddings_tx = [self.ui_embeddings_text]
        for i in range(self.au_uiLayerNums):
            layer=self.au_gcnLayers[i]
            if i == 0:  
                uiEmbeddings_tx = layer(self.ui_embeddings_text, self.uiMat, ui_index)
                itemEmbeddings0 = layer(item_embed0, self.iciMat, item_index)
                itemEmbeddings1 = layer(item_embed1, self.icaiMat, item_index)
            
            else:
                uiEmbeddings_tx   = layer(uiEmbeddings_tx, self.uiMat, ui_index)
                itemEmbeddings0 = layer((itemEmbeddings0 ), self.iciMat, item_index)
                itemEmbeddings1 = layer((  itemEmbeddings1 ), self.icaiMat, item_index)
    
            if norm == 1:
                norm_embeddings = F.normalize(uiEmbeddings_tx, p=2, dim=1)
                self.all_ui_embeddings_tx += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings1, p=2, dim=1)
                self.all_item_embeddings1 += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings0 += [norm_embeddings]
        self.uiEmbedding = t.stack(self.all_ui_embeddings_tx, dim=1)  
        self.uiEmbedding_tx = t.mean(self.uiEmbedding, dim = 1)
        self.ui_userEmbedding_tx, self.ui_itemEmbedding_tx  = t.split((self.uiEmbedding_tx), [self.userNum, self.itemNum])
        self.itemEmbedding  = t.stack(self.all_item_embeddings0, dim=1) 
        self.itemEmbedding0 = (t.mean(self.itemEmbedding, dim = 1))
        self.itemEmbedding  = t.stack(self.all_item_embeddings1, dim=1)
        self.itemEmbedding1 =  (t.mean(self.itemEmbedding, dim = 1))
      

        sampling_steps = self.sampling_steps
        sampling_noise = False
      
        elboii  = 0
        elboNonii = 0
        elbo_txi = 0
        elbo_txu = 0
        elboNon_txi = 0
        elboNon_txu = 0
        mse = 0
        
        ##train
        if iftraining == True:
            batch_latent = torch.zeros_like(self.ui_itemEmbedding)
            batch_latenti  = torch.zeros_like(self.ui_itemEmbedding)
            batch_latentNon = torch.zeros_like(self.ui_itemEmbedding)
            batch_latentiNon = torch.zeros_like(self.ui_itemEmbedding)
            batch_latentuNon = torch.zeros_like(self.ui_userEmbedding)
            batch_latentu = torch.zeros_like(self.ui_userEmbedding)
            
            userindex = torch.unique(user)
            itemindex=torch.unique(torch.cat((itemi,itemj))) 



            """Extraction of conditional features"""
        
            #Target domain features(self.ui_userEmbedding) as conditional features 
            conditionembed_ui2 = self.encodecon2(self.ui_userEmbedding.detach())
            #Target domain features(self.ui_itemEmbedding) as conditional features 
            conditionembed_ui = self.encodecon1(self.ui_itemEmbedding.detach())
            conditionembed = conditionembed_ui[itemindex]

            ##Auxiliary domain features to be denoised
            strartembed = (self.itemEmbedding0 + self.itemEmbedding1)/2.0
            strartembed =  strartembed[itemindex]

            """ The first stage: unconditional denoising of item-item domain"""

            terms = self.diffusion.training_losses(self.Nonmodel,iftraining,(strartembed.detach(), conditionembed), self.args.reweight)
            elboNonii =  terms["loss"].mean()
            batch_latent_reconNonindex = terms["pred_xstart"]
            batch_latentNon[itemindex]= batch_latent_reconNonindex
            batch_latent_reconNon_ii = batch_latentNon

            
            """ The second stage: conditional diffusion denoising"""

          

            #####The conditional denoising object of the auxiliary domain is the combination of unconditional denoising and original features
            batch_latent_reconNonindex =  ( batch_latent_reconNonindex.detach() + strartembed.detach())/2.0
            terms = self.diffusion.training_losses(self.cdnmodel,iftraining,( batch_latent_reconNonindex, conditionembed_ui[itemindex].detach()),self.args.reweight)
            elboii =  terms["loss"].mean()
            batch_latent_reconidx = terms["pred_xstart"]
            batch_latent[ itemindex]= batch_latent_reconidx
            batch_latent_recon_ii = batch_latent


            """Diffusion denoising process of the textual domain"""
            
            
            """ The first stage: unconditional denoising of textual domain"""
            #item
            terms1 = self.diffusion.training_losses(self.Nonmodel,iftraining,(self.ui_itemEmbedding_tx[itemindex].detach(),  conditionembed), self.args.reweight)
            elboNon_txi =  terms1["loss"].mean()
            batch_latent_reconiNonindex = terms1["pred_xstart"]
            batch_latentiNon[ itemindex]= batch_latent_reconiNonindex
            batch_latent_reconiNon = batch_latentiNon

            #user 
            terms1 = self.diffusion.training_losses(self.Nonmodel,iftraining,(self.ui_userEmbedding_tx[userindex].detach(),  conditionembed), self.args.reweight)
            elboNon_txu =  terms1["loss"].mean()
            batch_latent_reconuNonindex = terms1["pred_xstart"]
            batch_latentuNon[userindex ]= batch_latent_reconuNonindex
            batch_latent_reconuNon = batch_latentuNon
            


    
            
            ##### The conditional denoising object of the auxiliary domain is the combination of unconditional denoising and original features
            batch_latent_reconiNonindex =  (batch_latent_reconiNonindex.detach() +  self.ui_itemEmbedding_tx[itemindex].detach())/2.0
            terms = self.diffusion.training_losses(self.cdnmodel,iftraining,(batch_latent_reconiNonindex, conditionembed_ui[itemindex].detach()), self.args.reweight)
            elbo_txi =  terms["loss"].mean()
            batch_latent_reconidx = terms["pred_xstart"]
            batch_latenti[ itemindex]= batch_latent_reconidx
            batch_latent_reconi = batch_latenti

           
            ##### The conditional denoising object 
            batch_latent_reconuNonindex =  (batch_latent_reconuNonindex.detach() +  self.ui_userEmbedding_tx[userindex].detach())/2.0
            terms = self.diffusion.training_losses(self.cdnmodel,iftraining,(batch_latent_reconuNonindex, conditionembed_ui2[userindex].detach()), self.args.reweight)
            elbo_txu =  terms["loss"].mean()
            batch_latent_reconidx = terms["pred_xstart"]
            batch_latentu[ userindex ]= batch_latent_reconidx
            batch_latent_reconu = batch_latentu

        
            """Structure similarity constraint loss of conditional features"""

            uiadj = self.uiMat[: self.userNum,  self.userNum:]
            uiadj = self.sparse_mx_to_torch_sparse_tensor(uiadj)
            conditiembed1 = conditionembed_ui
            userembed = torch.spmm( (uiadj).cuda(),conditiembed1)/self.uinorm
            mse1 = ((userembed - (self.ui_userEmbedding))**2).sum(1)

            uiadj = self.uiMat[: self.userNum,  self.userNum:].T
            uiadj = self.sparse_mx_to_torch_sparse_tensor(uiadj)
            conditiembed2 = (conditionembed_ui2)
            itemembed = torch.spmm((uiadj).cuda(),conditiembed2)/self.iunorm
            mse2 = ((itemembed - (self.ui_itemEmbedding))**2).sum(1)
            mse = (mse2.mean() + mse1.mean())

        if iftraining == False:
            strartembed = (self.itemEmbedding0 + self.itemEmbedding1)/2.0
            ##Extraction of conditional features
            conditionembed_ui = self.encodecon1(self.ui_itemEmbedding.detach())
            conditionembed_ui2 =self.encodecon2(self.ui_userEmbedding.detach())
            #The first stage: unconditional denoising of auxiliary domain
            batch_latent_reconNon_ii  = self.diffusion.p_sample(self.Nonmodel, iftraining, (strartembed, conditionembed_ui), sampling_steps, sampling_noise)
            batch_latent_reconiNon = self.diffusion.p_sample(self.Nonmodel, iftraining, (self.ui_itemEmbedding_tx, conditionembed_ui), sampling_steps, sampling_noise)
            batch_latent_reconuNon = self.diffusion.p_sample(self.Nonmodel, iftraining, (self.ui_userEmbedding_tx, conditionembed_ui2), sampling_steps, sampling_noise)
           
            #The second stage: conditional diffusion denoising
            batch_latent_reconNon_ii = ( batch_latent_reconNon_ii + strartembed)/2.0
            batch_latent_recon_ii = self.diffusion.p_sample(self.cdnmodel, iftraining,(  batch_latent_reconNon_ii, conditionembed_ui), sampling_steps, sampling_noise)
           
            batch_latent_reconiNon = (batch_latent_reconiNon + self.ui_itemEmbedding_tx)/2.0
            batch_latent_reconi = self.diffusion.p_sample(self.cdnmodel, iftraining, ( batch_latent_reconiNon, conditionembed_ui), sampling_steps, sampling_noise)
        
            batch_latent_reconuNon= (batch_latent_reconuNon + self.ui_userEmbedding_tx)/2.0
            batch_latent_reconu = self.diffusion.p_sample(self.cdnmodel, iftraining, ( batch_latent_reconuNon, conditionembed_ui2), sampling_steps, sampling_noise)

        ##Enable the model to learn the ability to reconstruct data from noise
        elbo = (elbo_txi + elboNon_txi + elbo_txu + elboNon_txu) + ( elboii +  elboNonii)


        # Denoised and reconstructed auxiliary domain features
        reitemedtx = batch_latent_reconi * 0.5 + batch_latent_reconiNon * 0.5
        reuseredtx = batch_latent_reconu * 0.5 + batch_latent_reconuNon * 0.5
        reitemedii = batch_latent_recon_ii * 0.5 + batch_latent_reconNon_ii * 0.5

        # Added to the target domain features
        recouserembed = (user_interested + reuseredtx)/2.0
        recoitemembed = (reitemedtx + reitemedii)/2.0
        reitemembed   = (reitemedtx)
        reuserembed   = (user_interested + reuseredtx)/2.0
        return (elbo,mse,(reuseredtx,reitemedtx), (user_interested,reitemedii)), (user_interested + reuseredtx).detach()/2.0*0.25  + self.ui_userEmbedding_tx*0.25 + self.ui_userEmbedding*0.5,((reitemedtx.detach()*0.25 + reitemedii.detach()*0.25)/2.0 + self.ui_itemEmbedding_tx*0.25 + self.ui_itemEmbedding*0.5),((recoitemembed,self.ui_itemEmbedding)),((recouserembed ),self.ui_userEmbedding)


class GCN_layer(nn.Module):
    def __init__(self,hide_dim):
        super(GCN_layer, self).__init__()
        self.hide_dim = hide_dim
        self.weight = nn.Parameter(torch.FloatTensor(self.hide_dim,self.hide_dim))
        nn.init.xavier_normal_(self.weight.data)
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()    

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features=features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return (new_features)

        
      



