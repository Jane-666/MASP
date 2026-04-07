import torch
import random
import numpy as np
import argparse
import pickle
import dgl    
from scipy.sparse import csr_matrix
from datetime import datetime
from utils import *
from models.node_classification import NC

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='MASP')

    parser.add_argument('--embedder', nargs='?', default='MASP')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.002, help="Learning rate")
    parser.add_argument('--dim', type=int, default=128, help="Dimension size.")
    parser.add_argument('--num_aspects', type=int, default=5, help="Number of aspects")
    parser.add_argument('--isInit', action = 'store_true', default=True , help="Warm-up")
    parser.add_argument('--reg_coef', type=float, default=0.00001)

    parser.add_argument('--batch_size', type=int, default=9000)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--dropout', type=int, default=0.2) #0.3
    parser.add_argument('--iter_max', type=int, default=300)
    parser.add_argument('--tau_gumbel', type=float, default=0.7)
    parser.add_argument('--Is_hard', action='store_true', default=True)
    parser.add_argument('--gnn', type=str, default='GraphSAGE')# GraphSAGE
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--data', type=str, default='mydata3', choices=['mydata4','mydata3','shijie','mydata', 'dblp'])

    parser.add_argument('--scalar-edge-weights', action='store_true', default=True,
                        help='')

    parser.add_argument('--fusion-type', type=str, default='gated_with_cnn',
                        choices=['mlp', 'multihead', 'transformer', 'gated','gated_with_cnn',
                                 'hierarchical', 'adaptive_gumbel'],
                        help='')

    parser.add_argument('--fusion-heads', type=int, default=4,
                        help='')
    parser.add_argument('--inner-heads', type=int, default=2,
                        help='')
    parser.add_argument('--cross-heads', type=int, default=4,
                        help='')

    parser.add_argument('--initial-tau', type=float, default=1.0,
                        help='Gumbel-Softmax')
    parser.add_argument('--min-tau', type=float, default=0.1,
                        help='Gumbel-Softmax')
    parser.add_argument('--anneal-rate', type=float, default=0.99,
                        help='Gumbel')
    parser.add_argument('--gumbel-hard', action='store_true', default=True,
                        help='hard Gumbel-Softmax')


    num_labels_default = 6
    args, unknown = parser.parse_known_args()

    # Set defaults based on the data argument
    if args.data == 'mydata3':
        num_nodes_default = 11145 
        user_node_default =  8467
        num_labels_default = 6 
    elif args.data == 'mydata4':
        num_nodes_default = 11524  
        user_node_default = 9000 
        num_labels_default = 6
    elif args.data == 'shijie':
        num_nodes_default = 8942 
        user_node_default = 9000 
        num_labels_default = 5
    elif args.data == 'dblp':
        num_nodes_default = 26128
        user_node_default = 4057
        num_labels_default = 4

    parser.add_argument('--num_nodes', type=int, default=num_nodes_default)
    parser.add_argument('--user_node', type=int, default=user_node_default)
    parser.add_argument('--num_labels', type=int, default=num_labels_default)
    parser.add_argument('--graph', type=str, default='graph_0')
    parser.add_argument('--use-feature',type=bool, default=True)

    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'sum', 'max', 'attention'],
                        help='Pooling method for graph convolution')
    return parser.parse_known_args()

def get_node_features(args):
    node_features = []
    with open(f'./data/{args.data}/node.dat', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:  
                features = list(map(float, parts[3].split(',')))
                node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float32).to(args.device)
    return node_features

def main():
    args, unknown = parse_args()
    print(args.data)
    print("Start Learning [{}]".format( datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    graph = dgl.load_graphs(f'./data/{args.data}/graph_0_fixed.bin')[0][0].to(args.device)
    graph = dgl.add_self_loop(graph)
    adj_matrix = graph.adjacency_matrix()

    labels = torch.tensor(np.load(f'./data/{args.data}/labels.npy')).to(args.device).long()
    train = torch.load( f"./data/{args.data}/train_dataset_{args.ratio}.pt")
    # print(train.shape)
    val = torch.load( f"./data/{args.data}/val_dataset_{args.ratio}.pt")
    test = torch.load( f"./data/{args.data}/test_dataset_{args.ratio}.pt")
    node_features = get_node_features(args)
    embedder = NC(args,train,val,test,labels,graph,adj_matrix,node_features)

    embedder.training(args)
if __name__ == '__main__':
    main()
