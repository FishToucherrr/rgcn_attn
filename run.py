import os
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import dgl.function as fn
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
from model import EntityClassify
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time


from torch.utils.data import DataLoader, Dataset

# 1. 划分训练集和测试集
def get_dataset_splits(data_dir):
    file_paths = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_paths.append(os.path.join(subdir, file))

    train_paths, test_paths = train_test_split(file_paths, test_size=0.2, random_state=42)
    return train_paths, test_paths

def compute_edge_norm(g):
    # 计算每个目标节点的入度
    in_degrees = g.in_degrees().float()

    # 为了避免除以零，将入度为零的节点的入度设为1
    in_degrees[in_degrees == 0] = 1

    # 计算每条边的归一化权重
    edge_norm = 1.0 / in_degrees[g.edges()[1]]

    return edge_norm
# 2. 读取.pkl文件
def load_data(file_paths, edge_types,node_types):
    graphs = []

    for path in file_paths:
        with open(path, 'rb') as f:
            DGL = pickle.load(f)
            graph = DGL["dgl"]
            ntypes = set(graph.ndata['node_type'].tolist())

            etypes = set(graph.edata['edge_type'].tolist())
            edge_types |= etypes
            node_types |= ntypes

            label = 1 if 'attack' in path else 0
            
            DGL["dgl"] = graph
            DGL["loop_label"] = label
            DGL["edge_norm"] = compute_edge_norm(graph)
            
            # print(DGL["edge_norm"].shape,graph.number_of_edges())
            graphs.append(DGL)
            

    return graphs

# 3. 使用PyTorch的R-GCN模型


class DGLDataset(Dataset):
    def __init__(self, DGLGraphs):
        print("len of DGLGraphs:%d" % len(DGLGraphs))
        self.DGLGraphs = DGLGraphs

    def __len__(self):
        return len(self.DGLGraphs)

    def __getitem__(self, idx):
        return self.DGLGraphs[idx]


class SimpleBatch:
    def __init__(self, data):
        self.data = data

    def pin_memory(self):
        return self.data

def collate_wrapper(batch):
    return SimpleBatch(batch)



class Classifier(object):
    def __init__(self,edge_types,node_types,device,train_graphs,test_graphs):
        self.num_rels = len(edge_types)
        self.num_hidden_layers = 1
        self.activation = F.relu
        self.input_dim =32
        self.n_hidden = 16
        self.g_dim = 0
        self.num_workers = 0
        
        self.batch_size = 10
        self.epochs = 6
        self.lr = 0.01
        self.l2norm = float(0)
        self.train_graphs = train_graphs
        self.test_graphs = test_graphs
        self.out_dim = 2
        self.device = device
        
        self.num_ntypes = len(node_types)
        
        self.model = EntityClassify( 
                                self.num_ntypes,
                                self.input_dim,
                                self.n_hidden,
                                self.g_dim,
                                self.out_dim,
                                self.num_rels,
                                num_bases=-1,
                                num_hidden_layers=self.num_hidden_layers,
                                dropout=0.25,
                                use_self_loop=True)
        self.model = self.model.to(self.device)
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2norm)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=10, 
                                                                    verbose=True, min_lr=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()
    

    def iterate(self,dataloader,epoch=None):
        print("epoch :",epoch)
        total_loss = 0.0
        y = []
        y_hat = []

        for batch_idx, Graphs in enumerate(dataloader): # every batch
            if batch_idx % 10 == 0:
                print("-- %s on %d-st batch" % ("train", batch_idx))
            self.optimizer.zero_grad()
            batch_loss = 0.0

            for DGL in Graphs.pin_memory():
                if DGL is None: # failed building DGL
                    continue
                g = DGL["dgl"].to(self.device)
                edge_type = DGL["edge_type"]
                edge_type = edge_type.to(self.device)
                edge_norm = DGL["edge_norm"].unsqueeze(1)
                edge_norm = edge_norm.to(self.device)
                # train_idx = DGL["train_idx"]
                sketchs = []
                for sketch in DGL["sketchs"]:
                    sketch = torch.tensor(sketch).float().to(self.device)
                    sketchs.append(sketch)

                
                labels = F.one_hot(torch.tensor(DGL["loop_label"]).to(torch.long), 2).float().to(self.device)
                labels = torch.unsqueeze(labels, dim=0)
                
                logits_raw = self.model(g,edge_type, edge_norm,sketchs)
                # print(labels)
                # print(logits_raw)
                # logits = torch.reshape(logits, (1, 2))
                loss = self.criterion(logits_raw, labels)
                batch_loss += loss

                logits = logits_raw.detach().cpu().argmax(dim=1)
                labels = labels.detach().cpu()

                _y = DGL["loop_label"]
                _y_hat = logits.item()

                y.append(_y)
                y_hat.append(_y_hat)
                
            if epoch!= None:

                batch_loss.backward()
                self.optimizer.step()

                batch_loss = batch_loss.detach().cpu()
                total_loss += float(batch_loss.item())


        report = classification_report(
            y, y_hat
        )

        print(report)
    
    
    def train(self):
        self.model.train()
        train_dataset = DGLDataset(self.train_graphs)
        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, 
                                    collate_fn=collate_wrapper,
                                    pin_memory=False, shuffle=True)
        for epoch in range(1, self.epochs+1):
            self.iterate(train_dataloader,epoch)
            
    def test(self): 
        self.model.eval()
        test_dataset = DGLDataset(self.test_graphs)
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, 
                                    collate_fn=collate_wrapper,
                                    pin_memory=False, shuffle=True)
        with torch.no_grad():
            print("testing model:")
            self.iterate(test_dataloader)

        


if __name__ == '__main__':
    data_dir = 'pkl/'
    train_paths, test_paths = get_dataset_splits(data_dir)
    edge_types = set()
    node_types = set()
    train_graphs = load_data(train_paths,edge_types,node_types)
    test_graphs = load_data(test_paths,edge_types,node_types)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    node_type_dict = {r: i for i, r in enumerate(node_types)}
    edge_type_dict = {r: i for i, r in enumerate(edge_types)}
    
    num_ntypes = len(node_types)
    
    for DGL in train_graphs:
        DGL["edge_type"] = torch.tensor([edge_type_dict[int(r.item())] for r in DGL["dgl"].edata['edge_type']], dtype=torch.long)
        nt = torch.tensor([node_type_dict[int(r.item())] for r in DGL["dgl"].ndata['node_type']], dtype=torch.long)
        DGL["node_type"] = F.one_hot(nt.to(torch.long), num_ntypes).float()

        
    for DGL in test_graphs:
        DGL["edge_type"] = torch.tensor([edge_type_dict[int(r.item())] for r in DGL["dgl"].edata['edge_type']], dtype=torch.long)
        nt = torch.tensor([node_type_dict[int(r.item())] for r in DGL["dgl"].ndata['node_type']], dtype=torch.long)
        DGL["node_type"] = F.one_hot(nt.to(torch.long), num_ntypes).float()
    classifier = Classifier(edge_types,node_types,device,train_graphs,test_graphs)
    classifier.train()
    classifier.test()
    



    # precision, recall, f1 = evaluate(model, test_graphs, test_labels, batch_size)
    # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
