import json
import math
import pickle
import random
import logging
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix
import scipy.sparse as sp
save = True

# random seed
seed = 2022
random.seed(seed)
np.random.seed(seed)

# data_path = 'amazon'
data_path = 'F:/cmr/Thirdwork/code/RLMRec-main/RLMRec-main/data/data/amazon'
review_data_path = 'C:\\Users\\cmr\\dataset\\amazon\\origin\\Electronics.json'
meta_data_path = data_path + '/origin/meta_Books.json'
meta_data_path = "F:/cmr/Thirdwork/code/RLMRec-main/RLMRec-main/data/data/meta_Electronics.json/meta_Electronics.json"
amazon_item_map_path = "F:\\cmr\\Thirdwork\\code\\RLMRec-main\\RLMRec-main\\data\\data\\amazonelectronic\\filter_data\\remap_dict\\item_org2new_iid_dict.pkl"
# metas_data_path = "F:\\cmr\\Thirdwork\\code\\RLMRec-main\\RLMRec-main\\data\\data\\steam\\origin\\steam_metas.json"
# steam_item_map_path = "F:/cmr/Thirdwork/codeRLMRec-main/RLMRec-main/data/data/amazonelectronic/filter_data/steam/remap_dict/item_org2remap_dict.pkl"#'F:/cmr/Thirdwork/code/RLMRec-main/RLMRec-main/data/mapper/steam_item.json'
# f = open('F:/cmr/Thirdwork/code/RLMRec-main/RLMRec-main/data/data/steam/data/user_text.pkl', 'rb') 
# data = pickle.load(f)
# print(1)
# ####descrIBtion
# item_org2remap = []
# file = open(amazon_item_map_path, 'rb')
# item_org2remap_dict = pickle.load(file)
# metas = []
# publisher_org2remap_dict = {}
# with open(meta_data_path, 'r', encoding='utf-8') as fp:
#     for line in tqdm(fp.readlines(), desc='读取electronic文件'):
#         metas.append(line)
# cnt=0
# ii_iids, ii_sids = [], []
# ib_iids, ib_bids = [], []
# ic_iids, ic_cids = [], []
# cate_org2remap_dict = {}
# cate_org2remap_dict_inv = {}
# brand_org2remap_dict = {}
# brand_org2remap_dict_inv = {}
# for i in tqdm(range(len(metas)), desc='处理metas数据'):

#     data = json.loads(json.dumps(eval(metas[i].replace('\n', ''))))
#     org_item_id = data['asin']
#     if org_item_id not in item_org2remap_dict:
#         continue
  
    
#     iid = item_org2remap_dict[org_item_id]

#     ## categories
#     if 'categories' in data:    
#         categories = data['category']
#         assert type(categories) == list
#         for cate_org_id in categories:
#             if cate_org_id not in cate_org2remap_dict:
#                 new_cate_id = len(cate_org2remap_dict)
#                 cate_org2remap_dict[cate_org_id] = new_cate_id
#                 cate_org2remap_dict_inv[new_cate_id] = cate_org_id
#             cid = cate_org2remap_dict[cate_org_id]
#             ic_iids.append(iid)
#             ic_cids.append(cid)
#     ## brand
#     if 'brand' in data:
#         brand_org_id = data['brand']
#         assert type(brand_org_id) == str
#         if brand_org_id not in brand_org2remap_dict:
#             new_brand_id = len(brand_org2remap_dict)
#             brand_org2remap_dict[brand_org_id] = new_brand_id
#             brand_org2remap_dict_inv[new_brand_id] = brand_org_id
#         bid = brand_org2remap_dict[brand_org_id]
#         ib_iids.append(iid)
#         ib_bids.append(bid)
#     ## also buy items
#     if 'also_buy' in data:
#         also_buy_items = data['also_buy']
#         assert type(also_buy_items) == list
#         for co_buy_item in also_buy_items:
#             if co_buy_item not in item_org2remap_dict:
#                 continue
#             sid = item_org2remap_dict[co_buy_item]
#             ii_iids.append(iid)
#             ii_sids.append(sid)

# # final coalesce
# n_item = len(item_org2remap_dict)
# n_brand = len(brand_org2remap_dict)
# n_category = len(cate_org2remap_dict)
# ii_matrix = coo_matrix((np.ones(len(ii_iids)), (ii_iids, ii_sids)), shape=[n_item, n_item]).tocsr().tocoo()
# ii_matrix = (ii_matrix + ii_matrix.transpose()).tocsr().tocoo()  # important
# ib_matrix = coo_matrix((np.ones(len(ib_iids)), (ib_iids, ib_bids)), shape=[n_item, n_brand]).tocsr().tocoo()
# ic_matrix = coo_matrix((np.ones(len(ic_iids)), (ic_iids, ic_cids)), shape=[n_item, n_category]).tocsr().tocoo()

# ii_iids = ii_matrix.row
# ii_sids = ii_matrix.col
# ib_iids = ib_matrix.row
# ib_bids = ib_matrix.col
# ic_iids = ic_matrix.row
# ic_cids = ic_matrix.col

# ii_matrix = coo_matrix((np.ones(len(ii_iids)), (ii_iids, ii_sids)), shape=[n_item, n_item]).tocsr().tocoo()
# ib_matrix = coo_matrix((np.ones(len(ib_iids)), (ib_iids, ib_bids)), shape=[n_item, n_brand]).tocsr().tocoo()
# ic_matrix = coo_matrix((np.ones(len(ic_iids)), (ic_iids, ic_cids)), shape=[n_item, n_category]).tocsr().tocoo()
# assert len(np.where(ii_matrix.data == 1)[0]) == len(ii_iids)
# assert len(np.where(ib_matrix.data == 1)[0]) == len(ib_iids)
# assert len(np.where(ic_matrix.data == 1)[0]) == len(ic_iids)

# ### save    
# data_path = 'F:\\cmr\\Thirdwork\\code\\RLMRec-main\\RLMRec-main\\data\\data\\amazonelectronic\\filter_data' 
# with open("{}/II.csv".format(data_path), 'wb') as fs:
#     pickle.dump(ii_matrix .tocsr(),fs)
# with open("{}/IB.csv".format(data_path), 'wb') as fs:
#     pickle.dump(ib_matrix.tocsr(),fs)
# with open("{}/IC.csv".format(data_path), 'wb') as fs:
#     pickle.dump(ic_matrix.tocsr(),fs)
# # ICI
# data_path = 'C:\\Users\\cmr\\dataset\\steam\\data' 
data_path = 'F:\\cmr\\Thirdwork\\code\\RLMRec-main\\RLMRec-main\\data\\data\\amazonelectronic\\filter_data' 

with open("{}/II.csv".format(data_path), 'rb') as fs:
    IImat = pickle.load(fs)
itemNum =   IImat.shape[0]#33168 5508 33141
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))
for i in tqdm(range(itemNum)):#每个电影只可能属于一个国家
    itemType = np.where(IImat[i].toarray()!=0)[1]#7433 62
    for j in itemType:
        AList  = np.where(IImat[:,j ].toarray()!=0)[0]
        AList2 = np.random.choice( AList, size=int( AList.size * 0.03), replace=False)# 0.005 8754942 .0.001 1688452
        AList2 = AList2.tolist()
        tmp = [i]*len(AList2)
        ItemDistance_mat[tmp, AList2] = 1.0
        ItemDistance_mat[AList2, tmp] = 1.0  
##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()#22265
with open('./II.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)

print("Done")


with open("{}/IB.csv".format(data_path), 'rb') as fs:
    IBmat = pickle.load(fs)
itemNum =   IBmat.shape[0]#33168 1180 146058 
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))
for i in tqdm(range(itemNum)):#每个电影只可能属于一个国家
    itemType = np.where(IBmat[i].toarray()!=0)[1]#7433 62
    for j in itemType:
        AList = np.where(IBmat[:,j ].toarray()!=0)[0]
        AList2 =  np.random.choice( AList, size=int(  AList.size*0.08  ), replace=False)#0.05 5314912 
        AList2 = AList2.tolist()
        tmp = [i]*len(AList2)
        ItemDistance_mat[tmp, AList2] = 1.0
        ItemDistance_mat[AList2, tmp] = 1.0
##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()#45755
with open('./IBI.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)

print("Done")


with open("{}/IC.csv".format(data_path), 'rb') as fs:
    ICmat = pickle.load(fs)
itemNum =   ICmat.shape[0]#33168 1180 146058 
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))
for i in tqdm(range(itemNum)):#每个电影只可能属于一个国家
    itemType = np.where(ICmat[i].toarray()!=0)[1]#7433 62
    for j in itemType:
        AList = np.where(ICmat[:,j ].toarray()!=0)[0]
        AList2 =  np.random.choice( AList, size=int(  AList.size  ), replace=False)#0.05 5314912 
        AList2 = AList2.tolist()
        tmp = [i]*len(AList2)
        ItemDistance_mat[tmp, AList2] = 1.0
        ItemDistance_mat[AList2, tmp] = 1.0
##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()#45755
with open('./ICI.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)

print("Done")








# log file
logger = logging.getLogger('data_logger')
# logger.setLevel(logging.INFO)
# logfile = logging.FileHandler('sparse/{}/log/process.log'.format(data_path), 'a', encoding='utf-8')
# logfile.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(message)s')
# logfile.setFormatter(formatter)
# logger.addHandler(logfile)

# define the hyper-parameters
core_k = 10 # 10 (这个值用来初步过滤交互矩阵，保证每个用户/商品的degree最小值为core_k)
base_score = 4 # 4 (这个值表示用户打分至少为多少可以当作一次交互)
logger.info("Core-k {} and Base score {}".format(core_k, base_score))
print("Core-k {} and Base score {}".format(core_k, base_score))

# read review data
reviews = []
with open(review_data_path, 'r', encoding='utf-8') as fp:
    for line in tqdm(fp.readlines(), desc='读取review文件'):
        reviews.append(line)

# load review data and delete data with score less than 4
uids = []
iids = []
user_org2remap_dict = {}
item_org2remap_dict = {}
user_org2remap_dict_inv = {}
item_org2remap_dict_inv = {}
for i in tqdm(range(len(reviews)), desc='处理review数据'):
    data = json.loads(reviews[i])
    score = data['overall']
    if score < base_score:
        continue
    user_org_id = data['reviewerID']
    item_org_id = data['asin']
    if user_org_id not in user_org2remap_dict:
        new_uid = len(user_org2remap_dict)
        user_org2remap_dict[user_org_id] = new_uid
        user_org2remap_dict_inv[new_uid] = user_org_id
    if item_org_id not in item_org2remap_dict:
        new_iid = len(item_org2remap_dict)
        item_org2remap_dict[item_org_id] = new_iid
        item_org2remap_dict_inv[new_iid] = item_org_id
    user_id = user_org2remap_dict[user_org_id]
    item_id = item_org2remap_dict[item_org_id]
    uids.append(user_id)
    iids.append(item_id)

# Filter data to ensure k-core
## coalese and check
n_user = len(user_org2remap_dict)
n_item = len(item_org2remap_dict)
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
uids = ui_matrix.nonzero()[0].tolist()
iids = ui_matrix.nonzero()[1].tolist()
assert n_user == max(uids) + 1 and n_item == max(iids) + 1
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()

# 这段注释的代码，用于稀疏化数据集，就是只选取部分degree比较小的用户，不一定要使用
# # sparse
# sparse_max_degree = 200
# user_degree = ui_matrix.sum(axis=1).squeeze().tolist()[0]
# new_uids = []
# new_iids = []
# user_set = set()
# for i in tqdm(range(len(uids)), desc='稀疏化'):
#     uid = uids[i]
#     iid = iids[i]
#     if user_degree[uid] <= sparse_max_degree:
#         new_uids.append(uid)
#         new_iids.append(iid)
#         user_set.add(uid)
#
# # random choose
# max_choose_user = 1000000
# uids = new_uids.copy()
# iids = new_iids.copy()
# user_list = list(user_set)
# np.random.shuffle(user_list)
# is_choosed = np.zeros(max(user_list) + 1)
# is_choosed[user_list[:max_choose_user]] = 1
# new_uids = []
# new_iids = []
# for i in tqdm(range(len(uids)), desc='随机选择'):
#     uid = uids[i]
#     iid = iids[i]
#     if is_choosed[uid] == 1:
#         new_uids.append(uid)
#         new_iids.append(iid)
# uids = new_uids.copy()
# iids = new_iids.copy()
# ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
# uids = ui_matrix.nonzero()[0].tolist()
# iids = ui_matrix.nonzero()[1].tolist()
# ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()

## remove users/items for k-core
edge_list = []
for i in tqdm(range(len(uids)), desc='准备networkx的edges'):
    uid = uids[i]
    iid = iids[i] + n_user # node_id
    assert uid < n_user
    assert iids[i] >= 0
    edge_list.append((uid, iid))
G = nx.Graph(edge_list)
edge_list = list(nx.k_core(G, k=core_k).edges())
new_uids = []
new_iids = []
for i in tqdm(range(len(edge_list)), desc='k-core完成后收集edges'):
    assert edge_list[i][0] != edge_list[i][1]
    assert min(edge_list[i][0], edge_list[i][1]) >= 0
    assert max(edge_list[i][0], edge_list[i][1]) <= n_item + n_user - 1
    uid = min(edge_list[i][0], edge_list[i][1])
    iid = max(edge_list[i][0], edge_list[i][1])
    new_uids.append(uid)
    new_iids.append(iid - n_user)

## remap ids
new_user_org2remap_dict = {}
new_user_org2remap_dict_inv = {}
new_item_org2remap_dict = {}
new_item_org2remap_dict_inv = {}
remap_uids = []
remap_iids = []
for i in tqdm(range(len(new_uids)), desc='映射数据中'):
    org_user_id = user_org2remap_dict_inv[new_uids[i]]
    org_item_id = item_org2remap_dict_inv[new_iids[i]]
    if org_user_id not in new_user_org2remap_dict:
        new_uid = len(new_user_org2remap_dict)
        new_user_org2remap_dict[org_user_id] = new_uid
        new_user_org2remap_dict_inv[new_uid] = org_user_id
    if org_item_id not in new_item_org2remap_dict:
        new_iid = len(new_item_org2remap_dict)
        new_item_org2remap_dict[org_item_id] = new_iid
        new_item_org2remap_dict_inv[new_iid] = org_item_id
    user_id = new_user_org2remap_dict[org_user_id]
    item_id = new_item_org2remap_dict[org_item_id]
    remap_uids.append(user_id)
    remap_iids.append(item_id)
user_org2remap_dict = new_user_org2remap_dict
item_org2remap_dict = new_item_org2remap_dict
user_org2remap_dict_inv = new_user_org2remap_dict_inv
item_org2remap_dict_inv = new_item_org2remap_dict_inv
uids = remap_uids
iids = remap_iids
## calculate degree
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[len(user_org2remap_dict), len(item_org2remap_dict)]).tocsr().tocoo()
user_degree = ui_matrix.sum(axis=1).A[:, 0]
item_degree = ui_matrix.sum(axis=0).A[0, :]
assert np.where(user_degree < core_k)[0].size == 0 and np.where(item_degree < core_k)[0].size == 0
assert len(user_org2remap_dict) == max(uids) + 1 and len(item_org2remap_dict) == max(iids) + 1
logger.info("After first {}-core, users {} / {} and items {} / {}".format(core_k, len(user_org2remap_dict), n_user, len(item_org2remap_dict), n_item))
print("After first {}-core, users {} / {} and items {} / {}".format(core_k, len(user_org2remap_dict), n_user, len(item_org2remap_dict), n_item))
"""  """
# read meta data
metas = []
with open(meta_data_path, 'r', encoding='utf-8') as fp:
    for line in tqdm(fp.readlines(), desc='读取meta文件'):
        metas.append(line)

# load meta data and record I-I, I-Br and I-Ca relation
ii_iids, ii_sids = [], []
ib_iids, ib_bids = [], []
ic_iids, ic_cids = [], []
cate_org2remap_dict = {}
cate_org2remap_dict_inv = {}
brand_org2remap_dict = {}
brand_org2remap_dict_inv = {}
for i in tqdm(range(len(metas)), desc='处理meta数据'):
    data = json.loads(metas[i])
    org_item_id = data['asin']
    if org_item_id not in item_org2remap_dict:
        continue
    iid = item_org2remap_dict[org_item_id]
    ## categories
    categories = data['category']
    assert type(categories) == list
    for cate_org_id in categories:
        if cate_org_id not in cate_org2remap_dict:
            new_cate_id = len(cate_org2remap_dict)
            cate_org2remap_dict[cate_org_id] = new_cate_id
            cate_org2remap_dict_inv[new_cate_id] = cate_org_id
        cid = cate_org2remap_dict[cate_org_id]
        ic_iids.append(iid)
        ic_cids.append(cid)
    ## brand
    brand_org_id = data['brand']
    assert type(brand_org_id) == str
    if brand_org_id not in brand_org2remap_dict:
        new_brand_id = len(brand_org2remap_dict)
        brand_org2remap_dict[brand_org_id] = new_brand_id
        brand_org2remap_dict_inv[new_brand_id] = brand_org_id
    bid = brand_org2remap_dict[brand_org_id]
    ib_iids.append(iid)
    ib_bids.append(bid)
    ## also buy items
    also_buy_items = data['also_buy']
    assert type(also_buy_items) == list
    for co_buy_item in also_buy_items:
        if co_buy_item not in item_org2remap_dict:
            continue
        sid = item_org2remap_dict[co_buy_item]
        ii_iids.append(iid)
        ii_sids.append(sid)

# final coalesce
n_user = len(user_org2remap_dict)
n_item = len(item_org2remap_dict)
n_brand = len(brand_org2remap_dict)
n_category = len(cate_org2remap_dict)

ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
ii_matrix = coo_matrix((np.ones(len(ii_iids)), (ii_iids, ii_sids)), shape=[n_item, n_item]).tocsr().tocoo()
ii_matrix = (ii_matrix + ii_matrix.transpose()).tocsr().tocoo()  # important
ib_matrix = coo_matrix((np.ones(len(ib_iids)), (ib_iids, ib_bids)), shape=[n_item, n_brand]).tocsr().tocoo()
ic_matrix = coo_matrix((np.ones(len(ic_iids)), (ic_iids, ic_cids)), shape=[n_item, n_category]).tocsr().tocoo()

ui_uids = ui_matrix.row
ui_iids = ui_matrix.col
ii_iids = ii_matrix.row
ii_sids = ii_matrix.col
ib_iids = ib_matrix.row
ib_bids = ib_matrix.col
ic_iids = ic_matrix.row
ic_cids = ic_matrix.col

ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
ii_matrix = coo_matrix((np.ones(len(ii_iids)), (ii_iids, ii_sids)), shape=[n_item, n_item]).tocsr().tocoo()
ib_matrix = coo_matrix((np.ones(len(ib_iids)), (ib_iids, ib_bids)), shape=[n_item, n_brand]).tocsr().tocoo()
ic_matrix = coo_matrix((np.ones(len(ic_iids)), (ic_iids, ic_cids)), shape=[n_item, n_category]).tocsr().tocoo()
assert len(np.where(ui_matrix.data == 1)[0]) == len(ui_uids)
assert len(np.where(ii_matrix.data == 1)[0]) == len(ii_iids)
assert len(np.where(ib_matrix.data == 1)[0]) == len(ib_iids)
assert len(np.where(ic_matrix.data == 1)[0]) == len(ic_iids)

user_ui_degree = ui_matrix.sum(axis=1).A[:, 0]
item_ui_degree = ui_matrix.sum(axis=0).A[0, :]
item_ii_degree = ii_matrix.sum(axis=1).A[:, 0]
item_ib_degree = ib_matrix.sum(axis=1).A[:, 0]
brand_ib_degree = ib_matrix.sum(axis=0).A[0, :]
item_ic_degree = ic_matrix.sum(axis=1).A[:, 0]
category_ic_degree = ic_matrix.sum(axis=0).A[0, :]

# check remap dict
for org_id in user_org2remap_dict.keys():
    assert user_org2remap_dict_inv[user_org2remap_dict[org_id]] == org_id
for remap_id in user_org2remap_dict_inv.keys():
    assert user_org2remap_dict[user_org2remap_dict_inv[remap_id]] == remap_id

for org_id in item_org2remap_dict.keys():
    assert item_org2remap_dict_inv[item_org2remap_dict[org_id]] == org_id
for remap_id in item_org2remap_dict_inv.keys():
    assert item_org2remap_dict[item_org2remap_dict_inv[remap_id]] == remap_id

for org_id in cate_org2remap_dict.keys():
    assert cate_org2remap_dict_inv[cate_org2remap_dict[org_id]] == org_id
for remap_id in cate_org2remap_dict_inv.keys():
    assert cate_org2remap_dict[cate_org2remap_dict_inv[remap_id]] == remap_id

for org_id in brand_org2remap_dict.keys():
    assert brand_org2remap_dict_inv[brand_org2remap_dict[org_id]] == org_id
for remap_id in brand_org2remap_dict_inv.keys():
    assert brand_org2remap_dict[brand_org2remap_dict_inv[remap_id]] == remap_id


# statistics
logger.info("Final user-item relations {}, sparsity {}. User degree {}/{}/{}, Item degree {}/{}/{}".
      format(len(ui_uids), len(ui_uids) / (n_user * n_item), np.min(user_ui_degree), np.average(user_ui_degree), np.max(user_ui_degree), np.min(item_ui_degree), np.average(item_ui_degree), np.max(item_ui_degree)))
logger.info("Final item-item relations {}, sparsity {}. Item degree {}/{}/{}".
      format(len(ii_iids), len(ii_iids) / (n_item * n_item), np.min(item_ii_degree), np.average(item_ii_degree), np.max(item_ii_degree)))
logger.info("Final item-brand relations {}, sparsity {}. Item degree {}/{}/{}, Brand degree {}/{}/{}".
      format(len(ib_iids), len(ib_iids) / (n_item * n_brand), np.min(item_ib_degree), np.average(item_ib_degree), np.max(item_ib_degree), np.min(brand_ib_degree), np.average(brand_ib_degree), np.max(brand_ib_degree)))
logger.info("Final item-category relations {}, sparsity {}. Item degree {}/{}/{}, Category degree {}/{}/{}".
      format(len(ic_iids), len(ic_iids) / (n_item * n_category), np.min(item_ic_degree), np.average(item_ic_degree), np.max(item_ic_degree), np.min(category_ic_degree), np.average(category_ic_degree), np.max(category_ic_degree)))

print("Final user-item relations {}, sparsity {}. User degree {}/{}/{}, Item degree {}/{}/{}".
      format(len(ui_uids), len(ui_uids) / (n_user * n_item), np.min(user_ui_degree), np.average(user_ui_degree), np.max(user_ui_degree), np.min(item_ui_degree), np.average(item_ui_degree), np.max(item_ui_degree)))
print("Final item-item relations {}, sparsity {}. Item degree {}/{}/{}".
      format(len(ii_iids), len(ii_iids) / (n_item * n_item), np.min(item_ii_degree), np.average(item_ii_degree), np.max(item_ii_degree)))
print("Final item-brand relations {}, sparsity {}. Item degree {}/{}/{}, Brand degree {}/{}/{}".
      format(len(ib_iids), len(ib_iids) / (n_item * n_brand), np.min(item_ib_degree), np.average(item_ib_degree), np.max(item_ib_degree), np.min(brand_ib_degree), np.average(brand_ib_degree), np.max(brand_ib_degree)))
print("Final item-category relations {}, sparsity {}. Item degree {}/{}/{}, Category degree {}/{}/{}".
      format(len(ic_iids), len(ic_iids) / (n_item * n_category), np.min(item_ic_degree), np.average(item_ic_degree), np.max(item_ic_degree), np.min(category_ic_degree), np.average(category_ic_degree), np.max(category_ic_degree)))

# save as pickle file
## remap_dict
if save:
    with open("sparse/{}/remap_dict/user_org2remap_dict.pkl".format(data_path), 'wb') as fs:
        pickle.dump(user_org2remap_dict, fs)
    with open("sparse/{}/remap_dict/user_org2remap_dict_inv.pkl".format(data_path), 'wb') as fs:
        pickle.dump(user_org2remap_dict_inv, fs)

    with open("sparse/{}/remap_dict/item_org2remap_dict.pkl".format(data_path), 'wb') as fs:
        pickle.dump(item_org2remap_dict, fs)
    with open("sparse/{}/remap_dict/item_org2remap_dict_inv.pkl".format(data_path), 'wb') as fs:
        pickle.dump(item_org2remap_dict_inv, fs)

    with open("sparse/{}/remap_dict/cate_org2remap_dict.pkl".format(data_path), 'wb') as fs:
        pickle.dump(cate_org2remap_dict, fs)
    with open("sparse/{}/remap_dict/cate_org2remap_dict_inv.pkl".format(data_path), 'wb') as fs:
        pickle.dump(cate_org2remap_dict_inv, fs)

    with open("sparse/{}/remap_dict/brand_org2remap_dict.pkl".format(data_path), 'wb') as fs:
        pickle.dump(brand_org2remap_dict, fs)
    with open("sparse/{}/remap_dict/brand_org2remap_dict_inv.pkl".format(data_path), 'wb') as fs:
        pickle.dump(brand_org2remap_dict_inv, fs)

## dataset
dataset = {}
dataset['node_types'] = ['user', 'item', 'brand', 'category']
dataset['relations'] = ['user_item', 'item_item', 'item_brand', 'item_category']
dataset['user_num'] = n_user
dataset['item_num'] = n_item
dataset['brand_num'] = n_brand
dataset['category_num'] = n_category
dataset['item_item'] = ii_matrix
dataset['item_brand'] = ib_matrix
dataset['item_category'] = ic_matrix
logger.info("Statistics user_num {}, item_num {}, brand_num {} and category_num {}".format(n_user, n_item, n_brand, n_category))
print("Statistics user_num {}, item_num {}, brand_num {} and category_num {}".format(n_user, n_item, n_brand, n_category))

### split 8:1:1
length = len(ui_uids)
indices = np.random.permutation(length)
#### train_set
trn_len = int(length * 0.8)
trn_indices = indices[:trn_len]
trn_row = ui_uids[trn_indices].tolist()
trn_col = ui_iids[trn_indices].tolist()
trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
logger.info("First split: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))
print("First split: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))

# 这种方法是对于训练集里面度为0的用户/商品，直接从验证/测试集中把交互取回来
# # add zero degree's interactions
# for i in tqdm(range(len(uids)), desc='add interactions'):
#     uid = ui_uids[i]
#     iid = ui_iids[i]
#     if trn_user_degree[uid] < 1 or trn_item_degree[iid] < 1:
#         trn_row.append(uid)
#         trn_col.append(iid)
#
# trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
# trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
# trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
# assert np.where(trn_user_degree < 1)[0].size <= 0 and np.where(trn_item_degree < 1)[0].size <= 0
# logger.info("After add: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))
# print("After add: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))

# 这种方法是继续随机split，直到满足要求
while np.where(trn_user_degree < math.ceil(core_k / 4))[0].size > 0 and np.where(trn_item_degree < math.ceil(core_k / 4))[0].size > 0:
    length = len(ui_uids)
    indices = np.random.permutation(length)
    trn_len = int(length * 0.8)
    trn_indices = indices[:trn_len]
    trn_row = ui_uids[trn_indices].tolist()
    trn_col = ui_iids[trn_indices].tolist()
    trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
    trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
    trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
    logger.info("Split again: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))
    print("Split again: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))

trn_row = trn_matrix.row
trn_col = trn_matrix.col
trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
#### validation/test set
res_mat = (ui_matrix - trn_matrix).tocsr().tocoo()
res_ui_uids = res_mat.row
res_ui_iids = res_mat.col
assert len(res_ui_iids) + len(trn_row) == len(ui_uids)
assert len(np.where(res_mat.data == 1)[0]) == len(res_ui_uids)
length = len(res_ui_uids)
indices = np.random.permutation(length)

val_len = int(length * 0.5)
val_indices = indices[:val_len]
val_row = res_ui_uids[val_indices].tolist()
val_col = res_ui_iids[val_indices].tolist()
val_matrix = coo_matrix((np.ones(len(val_row)), (val_row, val_col)), shape=[n_user, n_item]).tocsr().tocoo()
val_row = val_matrix.row
val_col = val_matrix.col
val_matrix = coo_matrix((np.ones(len(val_row)), (val_row, val_col)), shape=[n_user, n_item]).tocsr().tocoo()

tst_indices = indices[val_len:]
tst_row = res_ui_uids[tst_indices].tolist()
tst_col = res_ui_iids[tst_indices].tolist()
tst_matrix = coo_matrix((np.ones(len(tst_row)), (tst_row, tst_col)), shape=[n_user, n_item]).tocsr().tocoo()
tst_row = tst_matrix.row
tst_col = tst_matrix.col
tst_matrix = coo_matrix((np.ones(len(tst_row)), (tst_row, tst_col)), shape=[n_user, n_item]).tocsr().tocoo()

assert len(ui_uids) == (len(val_row) + len(tst_row) + len(trn_row))
assert max([max(ui_uids), max(trn_row), max(val_row), max(tst_row)]) == n_user - 1
assert max(trn_row) == n_user - 1
assert max([max(ui_iids), max(ii_iids), max(ii_sids), max(ic_iids), max(ib_iids), max(trn_col), max(val_col), max(tst_col)]) == n_item - 1
assert max(trn_col) == n_item - 1
assert max([max(ib_bids)]) == n_brand - 1
assert max([max(ic_cids)]) == n_category - 1

trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
logger.info("Split result: trn user degree {}/{}/{}, trn item degree {}/{}/{}".
      format(np.min(trn_user_degree), np.average(trn_user_degree), np.max(trn_user_degree), np.min(trn_item_degree), np.average(trn_item_degree), np.max(trn_item_degree)))
logger.info("Split result: trn inter {}, val inter {} and tst inter {}".format(len(trn_row), len(val_row), len(tst_row)))
logger.info("Split result: trn inter / tst inter {}".format(len(trn_row) / len(tst_row)))
logger.info("Split result: trn inter / val inter {}".format(len(trn_row) / len(val_row)))
print("Split result: trn user degree {}/{}/{}, trn item degree {}/{}/{}".
      format(np.min(trn_user_degree), np.average(trn_user_degree), np.max(trn_user_degree), np.min(trn_item_degree), np.average(trn_item_degree), np.max(trn_item_degree)))
print("Split result: trn inter {}, val inter {} and tst inter {}".format(len(trn_row), len(val_row), len(tst_row)))
print("Split result: trn inter / tst inter {}".format(len(trn_row) / len(tst_row)))
print("Split result: trn inter / val inter {}".format(len(trn_row) / len(val_row)))

dataset['train'] = trn_matrix
dataset['test'] = tst_matrix
dataset['val'] = val_matrix

if save:
    with open("sparse/{}/dataset.pkl".format(data_path), 'wb') as fs:
        pickle.dump(dataset, fs)
