import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append('..')
from MF2Vec.embedder import embedder
import numpy as np
import torch.nn.init as init
import dgl
from dgl.nn.pytorch import GraphConv
from datetime import datetime
from sklearn.metrics import f1_score,roc_auc_score

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    num_classes = logits.shape[1]
    class_accuracy = {}
    class_f1 = {}

    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        if class_mask.sum() > 0:  
            class_acc = (prediction[class_mask] == labels[class_mask]).sum() / class_mask.sum()
            class_accuracy[class_id] = class_acc

            class_f1[class_id] = f1_score(labels, prediction, average=None)[class_id]

    return accuracy, micro_f1, macro_f1, class_accuracy, class_f1


def evaluate(model, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        z, logits = model()

    prob = F.softmax(logits[mask], 1)
    prob = prob.cpu().detach().numpy()
    accuracy, micro_f1, macro_f1, class_accuracy, class_f1 = score(logits[mask], labels[mask])
    labels_cpu = labels[mask].cpu()
    auc = roc_auc_score(labels_cpu, prob, multi_class='ovr')

    return loss_func(logits[mask], labels[mask]), accuracy, micro_f1, macro_f1, auc, z

def load_data_to_gpu(batch):
    return [tensor.long().cuda() for tensor in batch]

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class NC(embedder):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, labels, G, adj_matrix, node_features=None):
        embedder.__init__(self, args)
        self.clip_max = torch.FloatTensor([1.0]).cuda()
        self.train_dataset = train_dataset.tensors
        self.val_dataset = val_dataset.tensors
        self.test_dataset = test_dataset.tensors
        self.labels = labels
        self.g = G
        self.user_node = args.user_node
        self.adj_matrix = adj_matrix
        self.node_features = node_features

    def train_DW(self, args):
        model_DW = modeler_warm(args, self.adj_matrix,feat=self.node_features).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model_DW.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)
        loss_fcn = nn.CrossEntropyLoss()
        best = 0
        print("[{}] Start warm-up".format(currentTime()))
        for epoch in range(0, 150):
            self.batch_loss = 0

            uids = load_data_to_gpu(self.train_dataset)
            optimizer.zero_grad()
            z, logits = model_DW()

            loss = loss_fcn(logits[uids], self.labels[uids])
            loss.backward()
            optimizer.step()

            model_DW.center_embedding.weight.data.div_(
                torch.max(torch.norm(model_DW.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(
                    model_DW.center_embedding.weight.data))
            test_loss, val_acc, val_micro_f1, val_macro_f1, val_auc, z = evaluate(model_DW, self.labels,
                                                                                  load_data_to_gpu(self.val_dataset),
                                                                                  loss_fcn)
            test_loss, test_acc, test_micro_f1, test_macro_f1, test_auc, z = evaluate(model_DW, self.labels,
                                                                                      load_data_to_gpu(
                                                                                          self.test_dataset), loss_fcn)
            print('Epoch {:d} | Train Loss {:.4f} |Val ACC {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val AUC f1 {:.4f}'.format(
                    epoch + 1, loss.item(), test_acc, test_micro_f1, test_macro_f1, test_auc))
            if val_macro_f1 > best:
                best = val_macro_f1
                cnt_wait = 0

            else:
                cnt_wait += 1
            if cnt_wait == args.patience and epoch > 50 or cnt_wait == args.patience + 50:
                print('Early stopping!')
                break
        return model_DW.get_embeds()

    def training(self, args):
        pretrained_embed = self.train_DW(args) if self.isInit else None
        result = []
        self.args = args

        model_Nc = modeler_Nc(self.args, self.g, self.node_features ,pretrained_embed,use_features=args.use_feature).to(self.device)
        training_time = []
        print('#Parameters:', sum(p.numel() for p in model_Nc.parameters()))
        parameters = filter(lambda p: p.requires_grad, model_Nc.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=args.reg_coef)
        loss_fcn = nn.CrossEntropyLoss()
        best = 0
        min_loss = 100
        print("[{}] Start training Nc".format(currentTime()))
        self.batch_loss = 0
        for epoch in range(0, self.iter_max):
            t0 = datetime.now()

            uids = load_data_to_gpu(self.train_dataset)
            optimizer.zero_grad()
            z, logits = model_Nc()

            loss = loss_fcn(logits[uids], self.labels[uids])
            loss.backward()
            optimizer.step()
            training_time.append((datetime.now() - t0))
            train_loss, train_acc, _, _, _, _ = evaluate(model_Nc, self.labels, load_data_to_gpu(self.train_dataset),
                                                         loss_fcn)
            valloss, _, _, aa, _, z = evaluate(model_Nc, self.labels, load_data_to_gpu(self.val_dataset), loss_fcn)
            val_loss, val_acc, val_micro_f1, val_macro_f1, val_auc, z = evaluate(model_Nc, self.labels,
                                                                                 load_data_to_gpu(self.test_dataset),
                                                                                 loss_fcn)
            if min_loss > valloss:
                min_loss = valloss
                cnt_wait = 0
                result = [val_loss, val_acc, val_micro_f1, val_macro_f1, val_auc]
                best_z = z
            else:
                cnt_wait += 1
            if cnt_wait == args.patience and epoch > 50 or cnt_wait == args.patience + 50:
                print('Early stopping!')
                break

            print(
                'Epoch {:d} |tran_loss {:.4f}| train ACC {:.4f} Test Loss {:.4f} | Test ACC {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test AUC f1 {:.4f}'.format(
                    epoch + 1, train_loss, train_acc, val_loss, val_acc, val_micro_f1, val_macro_f1, val_auc))
        print("Total time: ", np.sum(training_time))
        print('Best model Loss {} |  Test ACC {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test AUC f1 {:.4f}'.format(
                *result))

        try:
            self.visualize_tsne(best_z, self.labels, self.test_dataset[0], args.data)
        except Exception as e:
            print(f"t-SNE: {e}")

    def visualize_tsne(self, embeddings, labels, test_indices, data_name, max_points=8467):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
        import matplotlib
        import os
        from datetime import datetime

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        embeddings_np = embeddings.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        total_nodes = len(embeddings_np)

        if total_nodes > max_points:
            np.random.seed(42)
            sampled_idx = np.random.choice(total_nodes, max_points, replace=False)
        else:
            sampled_idx = np.arange(total_nodes)

        sample_embeddings = embeddings_np[sampled_idx]
        sample_labels = labels_np[sampled_idx]

        tsne_params = {
            'n_components': 2,
            'random_state': 42,
            'perplexity': min(30, len(sample_embeddings) - 1),
            'init': 'random'
        }
        try:
            tsne_params['max_iter'] = 1000
            tsne = TSNE(**tsne_params)
            _ = tsne.get_params()
        except TypeError as e:
             if 'max_iter' in str(e):
                tsne_params.pop('max_iter', None)
                tsne_params['n_iter'] = 1000
            if 'learning_rate' in str(e):
                tsne_params.pop('learning_rate', None)

            tsne = TSNE(**tsne_params)

        print(f"TSNE: {tsne.get_params()}")

        embeddings_2d = tsne.fit_transform(sample_embeddings)

        custom_colors = [
            '#FF0000',  
            '#0000FF',  
            '#00FF00',  
            '#FFFF00',  
            '#800080',  
            '#00FFFF' 
        ]
        color_names = ["red", "blue", "green", "yello", "purple", "Cyan"]

        unique_labels = np.unique(sample_labels)
        unique_labels.sort()  

        if len(unique_labels) > len(custom_colors):
            print(f"warn：{len(unique_labels)}，{len(custom_colors)}")

            color_indices = np.arange(len(unique_labels)) % len(custom_colors)
        else:
            color_indices = np.arange(len(unique_labels))

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, label in enumerate(unique_labels):
            mask = sample_labels == label
            color_idx = color_indices[i]

            color = custom_colors[color_idx]
            color_name = color_names[color_idx] if color_idx < len(color_names) else f"颜色{color_idx + 1}"

            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                color=color,  
                label=f' {label} ({color_name})',
                s=20,  
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )

        ax.set_title(f'{data_name} - t-SNE ({len(sampled_idx)} nodes)', fontsize=14)
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.legend(title='class', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(False) 

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)

        filename = f'{save_dir}/tsne_all_nodes_{data_name}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ : {filename}")

        data_filename = f'{save_dir}/tsne_all_nodes_data_{data_name}_{timestamp}.npz'
        np.savez(
            data_filename,
            embeddings_2d=embeddings_2d,
            labels=sample_labels,
            indices=sampled_idx,
            embeddings=sample_embeddings
        )
        print(f"✓ : {data_filename}")

        return embeddings_2d

class modeler_warm(nn.Module):
    def __init__(self, args, adjacency_matrix, actual_num_classes=None,feat=None):
        super(modeler_warm, self).__init__()
        self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
        self.gcn1 = GraphConv(args.dim, args.dim)
        self.gcn2 = GraphConv(args.dim, args.dim)
        self.gcn3 = GraphConv(args.dim, args.dim)
        self.gcn4 = GraphConv(args.dim, args.dim)
        self.feat = feat
        self.fc = nn.Linear(self.feat.shape[1],args.dim)

        self.bn = nn.BatchNorm1d(args.dim)
        self.adjacency_matrix = adjacency_matrix
        self.init_weights()
        self.dropout = nn.Dropout(0.5)

        if actual_num_classes is not None:
            self.linear = nn.Linear(args.dim, actual_num_classes)
        else:
            self.linear = nn.Linear(args.dim, args.num_labels)

    def forward(self):
        x = self.fc(self.feat)

        indices = self.adjacency_matrix.indices()
        src, dst = indices[0], indices[1]

        graph = dgl.graph((src, dst))
        graph = graph.to(x.device)  

        x = self.gcn1(graph, x)
        x = F.elu(self.bn(self.dropout(x)))
        x = self.gcn2(graph, x)
        x = F.elu(self.bn(self.dropout(x)))
        x = self.gcn3(graph, x)

        return x, self.linear(x)

    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.1)

    # def get_embeds(self):
    #     self.eval()
    #     with torch.no_grad():
    #         x = self.fc(self.feat)
    #         indices = self.adjacency_matrix.indices()
    #         src, dst = indices[0], indices[1]

    #         graph = dgl.graph((src, dst))
    #         graph = graph.to(x.device)

    #         x = self.gcn1(graph, x)
    #         x = self.gcn2(graph, x)
    #         x = self.gcn3(graph, x)
    #     return x.detach.cpu()
        
    def get_embeds(self):
        with torch.no_grad():
            return self.center_embedding.weight.data.cpu()


class AdaptiveFeatureFusion(nn.Module):

    def __init__(self, feature_dim, embed_dim, fusion_type='gate'):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type

        if fusion_type == 'gate':

            self.feature_gate = nn.Sequential(
                nn.Linear(feature_dim + embed_dim, embed_dim),
                nn.Sigmoid()
            )
            self.feature_transform = nn.Linear(feature_dim, embed_dim)

        elif fusion_type == 'attention':

            self.feature_attention = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True
            )
            self.feature_proj = nn.Linear(feature_dim, embed_dim)
            self.embed_proj = nn.Linear(embed_dim, embed_dim)

        elif fusion_type == 'concat':

            self.concat_proj = nn.Linear(feature_dim + embed_dim, embed_dim)

    def forward(self, features, embeddings):
        if self.fusion_type == 'gate':
            feat_proj = self.feature_transform(features)
            gate_input = torch.cat([feat_proj, embeddings], dim=-1)
            gate = self.feature_gate(gate_input)
            return gate * feat_proj + (1 - gate) * embeddings

        elif self.fusion_type == 'attention':

            feat_proj = self.feature_proj(features).unsqueeze(1)
            embed_proj = self.embed_proj(embeddings).unsqueeze(1)

            attended, _ = self.feature_attention(embed_proj, feat_proj, feat_proj)
            return attended.squeeze(1)

        elif self.fusion_type == 'concat':
            combined = torch.cat([features, embeddings], dim=-1)
            return self.concat_proj(combined)

class modeler_Nc(nn.Module):
    def __init__(self, args, g,features=None, pretrained_embed=None,use_features=False):
        super(modeler_Nc, self).__init__()
        self.num_aspects = args.num_aspects
        self.num_nodes = args.num_nodes
        self.dim = args.dim
        self.args = args
        self.use_features = use_features
        self.device = torch.device("cuda:0")
        self.pretrained_embed = pretrained_embed
        self.features = features

        self.base_aspect_embedding = nn.Embedding(self.num_nodes, self.dim)
        nn.init.normal_(self.base_aspect_embedding.weight, mean=0.0, std=0.1)

        self.aspect_transform_layers = nn.ModuleList([
            self._create_aspect_transform(args.dim)
            for _ in range(self.num_aspects)
        ])

        if True:
            self.aspect_biases = nn.Parameter(
                torch.zeros(self.num_aspects, self.dim)
            )

        if self.use_features:
            self.feat_to_embed = AdaptiveFeatureFusion(feature_dim=self.features.shape[1], embed_dim=self.dim, fusion_type='concat')

        if pretrained_embed is not None:
            self.center_embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        else:
            self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
            self.init_weights()

        self.g = g
        self.dropout_rate = args.dropout
        fusion_type = args.fusion_type
        if fusion_type == 'mlp':
            self.edge_fusion = EdgeWeightFusionMLP(
                num_aspects=self.num_aspects,
                dim=self.dim,
                dropout=args.dropout
            )
            self.return_attention = False

        elif fusion_type == 'transformer':
            num_heads = getattr(args, 'fusion_heads', 4)
            self.edge_fusion = EdgeWeightFusionTransformer(
                num_aspects=self.num_aspects,
                dim=self.dim,
                num_heads=num_heads,
                dropout=args.dropout
            )
            self.return_attention = False

        elif fusion_type == 'gated':
            self.edge_fusion = EdgeWeightFusionGated(
                num_aspects=self.num_aspects,
                dim=self.dim,
                dropout=args.dropout
            )
            self.return_attention = False
        elif fusion_type == 'gated_with_cnn':
            print("ImproEdgeWeightFusionGated")
            self.edge_fusion = ImprovedEdgeWeightFusionCNN(
                num_aspects=self.num_aspects,
                dim=self.dim,
                dropout=args.dropout
            )
            self.return_attention = False
        elif fusion_type == 'adaptive_gumbel':
            initial_tau = getattr(args, 'initial_tau', 1.0)
            min_tau = getattr(args, 'min_tau', 0.1)
            anneal_rate = getattr(args, 'anneal_rate', 0.99)
            hard = getattr(args, 'gumbel_hard', True)

            self.edge_fusion = AdaptiveGumbelFusion(
                num_aspects=self.num_aspects,
                dim=self.dim,
                initial_tau=initial_tau,
                min_tau=min_tau,
                anneal_rate=anneal_rate,
                hard=hard,
                dropout=args.dropout
            )
            self.return_attention = True

        else:
            raise ValueError(f": {fusion_type}")

        self.GCN = GCN(self.dim, self.dropout_rate)
        self.GAT = GAT(self.dim, self.dropout_rate)
        self.GraphSAGE = GraphSAGE(self.dim, self.dropout_rate)

        self.gnn = args.gnn
        self.linear = nn.Linear(self.dim, args.num_labels, bias=True)
        self.classifier = SimpleEnhancedLinear(self.dim, args.num_labels)

    def _create_aspect_transform(self, dim, transform_type='linear'):
        if transform_type == 'linear':
            return nn.Linear(dim, dim, bias=False)
        elif transform_type == 'mlp':
            return nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            )
        elif transform_type == 'residual':
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        else:
            return nn.Identity()

    def init_weights(self, pretrained_embed=None):
        nn.init.normal_(self.edge_aspect_embedding.weight.data, mean=0.0, std=0.1)
        if pretrained_embed is None:
            nn.init.normal_(self.center_embedding.weight.data, mean=0.0, std=0.1)

    def _compute_edge_weights(self):
        base_emb = self.base_aspect_embedding.weight

        edge_type_indices = self.g.edata['w']
        base_edge_emb = base_emb[edge_type_indices]

        edge_weights = []
        for k in range(self.num_aspects):
            transform_layer = self.aspect_transform_layers[k]

            aspect_edge_emb = transform_layer(base_edge_emb)

            if hasattr(self, 'aspect_biases'):
                aspect_edge_emb = aspect_edge_emb + self.aspect_biases[k]

            edge_weights.append(aspect_edge_emb)

        edge_weight = torch.stack(edge_weights, dim=1)

        if self.return_attention:
            fused_edge_weight, attention_weights = self.edge_fusion(edge_weight)
            self.attention_weights = attention_weights
        else:
            fused_edge_weight = self.edge_fusion(edge_weight)

        scalar_edge_weight = torch.norm(fused_edge_weight, dim=1)
        scalar_edge_weight = torch.sigmoid(scalar_edge_weight) 

        return scalar_edge_weight

    def reset_fusion_step(self):
        if hasattr(self.edge_fusion, 'reset_step'):
            self.edge_fusion.reset_step()

    def forward(self):
        edge_weight = self._compute_edge_weights()
        node_weight = self.center_embedding.weight.data
        embedding = self.center_embedding.weight.data
        node_weight = self.features.to(self.device)

        if self.use_features:
            node_weight = self.feat_to_embed(node_weight,embedding)

        if self.gnn == 'GCN':
            self.h = self.GCN(self.g, node_weight, edge_weight).view(-1, self.dim)
        elif self.gnn == 'GAT':
            self.h = self.GAT(self.g, node_weight, edge_weight).view(-1, self.dim)
        elif self.gnn == 'GraphSAGE':
            self.h = self.GraphSAGE(self.g, node_weight, edge_weight).view(-1, self.dim)

        combined = self.h + node_weight
        combined = F.normalize(combined, p=2, dim=1)
        return self.h, self.classifier(combined)

class SimpleEnhancedLinear(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(self.norm(x))

class GCN(nn.Module):
    def __init__(self, in_dim, dropout_rate):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.gcn1 = GraphConv(in_dim, in_dim, weight=True)
        self.gcn2 = GraphConv(in_dim, in_dim, weight=True)
        self.gcn3 = GraphConv(in_dim, in_dim, weight=True)
        self.bn1 = nn.BatchNorm1d(in_dim)

        self.activate = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = nn.Linear(in_dim, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GraphConv) and m.weight is not None:
                init.normal_(m.weight.data, mean=0.0, std=0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, graph, node_weight, edge_weight):
        h = self.gcn1(graph, node_weight, edge_weight=edge_weight)
        h = self.dropout(self.bn1(h))
        h = self.activate(h)

        h = self.gcn2(graph, h, edge_weight=edge_weight)
        h = self.dropout(self.bn1(h))
        h = self.activate(h)

        h = self.gcn3(graph, h, edge_weight=edge_weight)
        h = self.bn1(h)
        return h


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


class GAT(nn.Module):
    def __init__(self, in_dim, dropout_rate):
        super(GAT, self).__init__()
        self.in_dim = in_dim

        self.gat1 = dglnn.GATConv(in_dim, in_dim, 1)
        self.gat2 = dglnn.GATConv(in_dim, in_dim, 1)
        self.gat3 = dglnn.GATConv(in_dim, in_dim, 1)

        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.activate = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph, node_weight, edge_weight):

        h = self.gat1(graph, node_weight)

        if h.dim() > 2:
            h = h.mean(dim=1) 
        h = self.bn1(self.dropout(h))
        h = self.activate(h)

        h = self.gat2(graph, h)
        if h.dim() > 2:
            h = h.mean(dim=1)
        h = self.bn2(self.dropout(h))
        h = self.activate(h)

        h = self.gat3(graph, h)
        if h.dim() > 2:
            h = h.mean(dim=1)
        h = self.bn3(self.dropout(h))

        return h

from dgl.nn.pytorch import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = 3    
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attn = nn.Linear(in_dim, 1, bias=True)
        for i in range(self.num_layers):
            self.convs.append(SAGEConv(in_dim, in_dim, 'pool'))
            self.bns.append(nn.BatchNorm1d(in_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
        self.activate = nn.ELU()

    def forward(self, graph, node_weight, edge_weight):
        x = node_weight
        for i in range(self.num_layers):
            x = self.convs[i](graph, x, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = self.activate(x)
            x = self.dropouts[i](x)
        return x


class EdgeWeightFusionMLP(nn.Module):

    def __init__(self, num_aspects, dim, dropout=0.1):
        super().__init__()
        self.num_aspects = num_aspects
        self.dim = dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * num_aspects, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        self.aspect_weights = nn.Parameter(torch.ones(num_aspects) / num_aspects)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_weights):
        batch_size = edge_weights.shape[0]

        flattened = edge_weights.view(batch_size, -1)
        fused = self.fusion_mlp(flattened)

        aspect_weights = F.softmax(self.aspect_weights, dim=0)
        weighted_avg = torch.sum(
            edge_weights * aspect_weights.view(1, self.num_aspects, 1),
            dim=1
        )

        output = fused + weighted_avg
        return self.dropout(output)

class EdgeWeightFusionTransformer(nn.Module):

    def __init__(self, num_aspects, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_aspects = num_aspects
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** 0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_weights):
        batch_size = edge_weights.shape[0]

        Q = self.query(edge_weights.mean(dim=1)).unsqueeze(1) 
        K = self.key(edge_weights) 
        V = self.value(edge_weights)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V).squeeze(1)  

        output = self.layer_norm(attended + edge_weights.mean(dim=1))
        return self.dropout(output)

class ImprovedEdgeWeightFusionCNN(nn.Module):

    def __init__(self, num_aspects, dim, dropout=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=dim,  
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim  
        )

        self.attention = nn.Sequential(
            nn.Linear(num_aspects * dim, num_aspects),
            nn.Softmax(dim=-1)
        )

        self.gate = nn.Sequential(
            nn.Linear(num_aspects * dim, num_aspects),
            nn.Softmax(dim=-1)
        )

        self.alpha = nn.Parameter(torch.tensor(0.8))  

    def forward(self, edge_weights):
        batch_size = edge_weights.shape[0]

        cnn_input = edge_weights.transpose(1, 2).contiguous()  
        cnn_out = self.conv1d(cnn_input).transpose(1, 2)  

        cnn_out = cnn_out.contiguous()

        flattened = edge_weights.reshape(batch_size, -1)  
        gates = self.gate(flattened)
        gated = torch.sum(edge_weights * gates.unsqueeze(-1), dim=1)

        flattened_cnn = cnn_out.reshape(batch_size,-1) 
        cnn_gates = self.attention(flattened_cnn)
        cnn_gated = torch.sum(cnn_out * cnn_gates.unsqueeze(-1), dim=1)

        alpha = torch.sigmoid(self.alpha)
        output = alpha * gated + (1 - alpha) * cnn_gated

        return output

class EdgeWeightFusionGated(nn.Module):

    def __init__(self, num_aspects, dim, dropout=0.1):
        super().__init__()
        self.num_aspects = num_aspects
        self.dim = dim

        self.gate = nn.Sequential(
            nn.Linear(dim * num_aspects, num_aspects * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_aspects * 2, num_aspects),
            nn.Softmax(dim=-1)
        )

        self.transform = nn.Sequential(
            nn.Linear(dim * num_aspects, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_weights):
        batch_size = edge_weights.shape[0]
        flattened = edge_weights.view(batch_size, -1)

        gates = self.gate(flattened)

        gated_weights = torch.sum(edge_weights * gates.unsqueeze(-1), dim=1)

        transformed = self.transform(flattened)

        output = transformed + gated_weights
        return self.dropout(output)

import math
class AdaptiveGumbelFusion(nn.Module):

    def __init__(self, num_aspects, dim, initial_tau=1.0, min_tau=0.1,
                 anneal_rate=0.99, hard=True, dropout=0.1):
        super().__init__()
        self.num_aspects = num_aspects
        self.dim = dim
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.anneal_rate = anneal_rate
        self.hard = hard

        self.log_tau = nn.Parameter(torch.tensor(math.log(initial_tau)))

        self.attention_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(dim * num_aspects, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('step', torch.tensor(0))

    def forward(self, edge_weights):
        
        batch_size = edge_weights.shape[0]

        attention_scores = self.attention_proj(edge_weights).squeeze(-1) 

        if self.training:
            learned_tau = torch.exp(self.log_tau)

            step_annealed_tau = max(self.initial_tau * (self.anneal_rate ** self.step), self.min_tau)

            tau = min(learned_tau, step_annealed_tau)

            self.step += 1
        else:
            tau = self.min_tau

        attention_weights = F.gumbel_softmax(
            attention_scores,
            tau=tau,
            hard=self.hard
        )  

        attended = torch.sum(edge_weights * attention_weights.unsqueeze(-1), dim=1)

        flattened = edge_weights.view(batch_size, -1) 
        transformed = self.feature_transform(flattened) 

        output = self.layer_norm(attended + transformed)

        return self.dropout(output), attention_weights

    def get_temperature(self):
        return torch.exp(self.log_tau).item()

    def reset_step(self):
        self.step = torch.tensor(0)
