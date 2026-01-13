import numpy as np

import dgl
import torch
import numpy as np
from utils import node_classification_split  # 假设文档7已导入
mydata = 'mydata3'
node_nums = 11145
user_lenth = 8467  # 病人节点数

def create_mydata_graph():
    # 手动构建边列表
    src_nodes = []
    dst_nodes = []
    edge_features = []

    # 从link.dat读取边信息
    with open('./data/'+mydata+'/link.dat', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                src = int(parts[0])
                dst = int(parts[1])
                edge_type = int(parts[2])
                src_nodes.append(src)
                dst_nodes.append(dst)
                edge_features.append(edge_type)
    # 创建DGL图
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
    g.edata['w'] = torch.tensor(edge_features, dtype=torch.long)

    # 保存图
    dgl.save_graphs('./data/'+mydata+'/graph_0.bin', [g])
    print("图文件生成完成！")

def validate_graph_data():
    # 加载图数据
    graph = dgl.load_graphs('data/'+mydata+'/graph_0.bin')[0][0]

    print("=== 图数据验证 ===")
    print(f"节点数: {graph.number_of_nodes()}")
    print(f"边数: {graph.number_of_edges()}")

    # 检查边数据
    if 'w' in graph.edata:
        edge_types = graph.edata['w']
        print(f"边类型数据形状: {edge_types.shape}")
        print(f"边类型取值范围: {edge_types.min()} 到 {edge_types.max()}")
        print(f"边类型唯一值: {torch.unique(edge_types)}")

        # 检查是否有越界索引
        if edge_types.max() >= graph.number_of_nodes():
            print(f"错误: 边类型索引越界! 最大索引{edge_types.max()} >= 节点数{graph.number_of_nodes()}")
            # 修复越界索引
            fixed_edge_types = edge_types % graph.number_of_nodes()
            graph.edata['w'] = fixed_edge_types
            print("已修复边类型索引")
    else:
        print("警告: 图中没有边类型数据'w'")
        # 添加默认边类型
        graph.edata['w'] = torch.zeros(graph.number_of_edges(), dtype=torch.long)
        print("已添加默认边类型数据")

    # 保存修复后的图
    dgl.save_graphs('data/'+mydata+'/graph_0_fixed.bin', [graph])
    print("修复后的图已保存")

if __name__ == '__main__':
    create_mydata_graph()

    # 总节点数11523，病人节点9000，标签类别6
    labels = np.zeros(node_nums, dtype=int)  # 非病人节点标签为0
    lab = 0
    with open(f'data\{mydata}\label.dat', 'r', encoding='utf-8') as f:
        for line in f:
            th = line.split('\t')
            node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), int(th[3])
            labels[node_id] = node_label
            lab = max(lab, node_label+1)
    # print(labels[:])
    print('最大的标签',lab)
    np.save(f'data\{mydata}\labels.npy', labels)
    # for i in range(11523):
    #     print(labels[i])


    node_classification_split(mydata, user_lenth)
    validate_graph_data()
    node_classification_split(mydata, user_lenth,0.8)
