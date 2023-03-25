import glob
import networkx as nx
import community as community_louvain
import itertools
from collections import defaultdict
import numpy as np
import time
from scipy.sparse import lil_matrix, csr_matrix
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import dgl
import pickle
import torch
import mmh3


def hash_feature(value, num_buckets):

    value = str(value).encode('utf-8')
    hash_val = mmh3.hash(value, seed=42)
    feature = np.zeros(num_buckets)
    feature[hash_val % num_buckets] = 1
    return feature

def graph_sketch(graph, node_buckets=8, edge_buckets=6, connection_buckets=6):
    node_type_sketch = np.zeros(node_buckets)
    edge_type_sketch = np.zeros(edge_buckets)
    connection_sketch = np.zeros(connection_buckets)

    for node, attributes in graph.nodes(data=True):
        node_type = attributes['node_type']
        node_type_feature = hash_feature(node_type, node_buckets)
        node_type_sketch += node_type_feature

    for u, v, attributes in graph.edges(data=True):
        edge_type = attributes['edge_type']
        edge_type_feature = hash_feature(edge_type, edge_buckets)
        edge_type_sketch += edge_type_feature

        connection_feature = hash_feature(str(u) + "_" + str(v), connection_buckets)
        connection_sketch += connection_feature

    node_type_sketch /= np.sum(node_type_sketch)
    edge_type_sketch /= np.sum(edge_type_sketch)
    connection_sketch /= np.sum(connection_sketch)

    graph_sketch = np.concatenate([node_type_sketch, edge_type_sketch, connection_sketch])

    return graph_sketch

def save_as_dgl(G,sketchs, id):
    # 创建一个空的 DGL 图
    DGL = {}
    graph = dgl.from_networkx(G, node_attrs=['node_type'], edge_attrs=['edge_type'])

    DGL["dgl"] = graph
    DGL["sketchs"] = sketchs
    # print(DGL.ndata["node_type"])
    # print(DGL.edata["edge_type"])

    if id >= 300 and id < 400:
        with open("pkl/attack/" + str(id) + ".pkl", "wb") as f:
            pickle.dump(DGL, f)
    else:
        with open("pkl/benign/" + str(id) + ".pkl", "wb") as f:
            pickle.dump(DGL, f)

# with open("graph.pkl", "rb") as f:
#     loaded_g = pickle.load(f)


def compress_nodes_using_louvain(G):
    partition = community_louvain.best_partition(G.to_undirected(),resolution=12)
    compressed_graph = nx.DiGraph()

    # 添加压缩后的节点到新图
    for node, community in partition.items():
        compressed_graph.add_node(community, node_type=G.nodes[node]['node_type'])

    # 添加压缩后的边到新图
    for u, v, edge_data in G.edges(data=True):
        u_community = partition[u]
        v_community = partition[v]
        edge_type = edge_data['edge_type']

        # 如果两个社区之间没有边，添加一条新边
        if not compressed_graph.has_edge(u_community, v_community):
            compressed_graph.add_edge(u_community, v_community, edge_type=edge_type)


    return compressed_graph




def directed_jaccard_similarity(graph, node1, node2):
    successors1 = set(graph.successors(node1))
    successors2 = set(graph.successors(node2))
    predecessors1 = set(graph.predecessors(node1))
    predecessors2 = set(graph.predecessors(node2))

    intersection_successors = successors1.intersection(successors2)
    intersection_predecessors = predecessors1.intersection(predecessors2)

    union_successors = successors1.union(successors2)
    union_predecessors = predecessors1.union(predecessors2)

    if len(union_successors) == 0 or len(union_predecessors) == 0:
        return 0
    else:
        return (len(intersection_successors) / len(union_successors)) * (len(intersection_predecessors) / len(union_predecessors))

def aggregate_edges_by_directed_jaccard_similarity(graph, target_edge_count):
    aggregated_graph = graph.copy()

    edge_similarity = []
    for edge in graph.edges(data=True):
        u, v, data = edge
        similarity = directed_jaccard_similarity(graph, u, v)
        edge_similarity.append((u, v, similarity))

    # 根据相似度对边进行排序
    edge_similarity.sort(key=lambda x: x[2], reverse=True)

    # 删除边直到达到目标边数
    edges_to_remove = edge_similarity[:len(edge_similarity) - target_edge_count]
    for u, v, similarity in edges_to_remove:
        aggregated_graph.remove_edge(u, v)

    return aggregated_graph

def compress_edges_by_network_flow(G, target_num_edges):
    start_time = time.time()
    H = G.copy()

    num_edges = H.number_of_edges()

    # 计算每个节点的度
    degrees = dict(H.degree())

    # 计算每条边的权重并将权重赋给边
    for u, v in H.edges():
        w = 1 / (degrees[u] + degrees[v] + 1)
        H[u][v]['weight'] = w
        
    nodes = list(H.nodes())
    source_node = nodes[0]
    target_node = nodes[1]
    
    # 计算最大流并找到割边
    while num_edges > target_num_edges:
        flow_value, flow_dict = nx.maximum_flow(H, source_node, target_node, capacity='weight')
        cut_nodes = set()
        visited_nodes = set()

        # 使用深度优先搜索找到割边
        def dfs(u, visited):
            visited.add(u)
            for v, flow in flow_dict[u].items():
                residual_capacity = H[u][v]['weight'] - flow
                if residual_capacity > 0 and v not in visited:
                    dfs(v, visited)

        dfs(source_node, visited_nodes)
        cut_nodes = visited_nodes

        # 找到割边并从图中删除
        edges_to_remove = set()
        for u, v in H.edges():
            if u in cut_nodes and v not in cut_nodes:
                edges_to_remove.add((u, v))

        for u, v in edges_to_remove:
            H.remove_edge(u, v)

        num_edges = H.number_of_edges()

    end_time = time.time()
    print('Total time: {:.2f} seconds'.format(end_time - start_time))
    return H



def task(file):
    G = nx.DiGraph()

    graph_id = int(file.split('/')[-1].split('.')[0])
    print(graph_id)
    node_types = []
    edge_types = []
    sketchs = []
    with open(file, "r") as f:
        edge_num = 0
        line_count = 0
        line_sum = 0
        for _ in enumerate(file):
            line_sum += 1
            
        for line in f:
            line_count += 1
            edge_num += 1
            line = line.strip().split("\t")
            source_node_id = int(line[0])
            source_node_type = ord(line[1])-ord('a')
            dest_node_id = int(line[2])
            dest_node_type = ord(line[3])-ord('a')
            edge_type = ord(line[4])-ord('a')

            if source_node_type not in node_types:
                node_types.append(source_node_type)
            if dest_node_type not in node_types:
                node_types.append(dest_node_type)
            if edge_type not in edge_types:
                edge_types.append(edge_type)

            G.add_node(source_node_id, node_type=source_node_type)
            G.add_node(dest_node_id, node_type=dest_node_type)
            G.add_edge(source_node_id, dest_node_id, edge_type=edge_type)

            if G.number_of_nodes() >= 8000:
                print("compressing nodes ..")
                G = compress_nodes_using_louvain(G)
                print("compress done.")
                # print(G.number_of_nodes())
            if edge_num >= 30000:
                if G.number_of_edges() < 15000:
                    edge_num = 0
                    continue
                
                print("compressing edges ..")
                G = aggregate_edges_by_directed_jaccard_similarity(G,10000)
                # G = compress_edges_by_network_flow(G,10000)
                print("compress done ..")
                edge_num = G.number_of_edges()
                print(edge_num)
            
            if line_count == round(0.25 * line_sum):
                sketchs.append(graph_sketch(G))
            elif line_count == round(0.5 * line_sum):
                sketchs.append(graph_sketch(G))
            elif line_count == round(0.75 * line_sum):
                sketchs.append(graph_sketch(G))

                
        if G.number_of_edges() > 12000:       
            print("compressing edges ..")
            G = aggregate_edges_by_directed_jaccard_similarity(G,10000)
            # G = compress_edges_by_network_flow(G,10000)
            print("compress done ..")
            
    sketchs.append(graph_sketch(G))
    save_as_dgl(G,sketchs,graph_id)
    # print(len(node_types),len(edge_types))

def batch_task(task_list):
    for arg in task_list:
        task(arg)
    
def main():

    attack_files = glob.glob("raw/attack/*.tsv")
    benign_files = glob.glob("raw/benign/*.tsv")

    
    for file in attack_files:
        task(file)
    for file in benign_files:
        task(file)
    # executor = ProcessPoolExecutor(max_workers=1)
    # batch_size = 1
    # tasks = set()
    # try:
    #     # for i in range(0, len(attack_files), batch_size):
    #     #     batch = attack_files[i : i+batch_size]
    #     #     tasks.add(executor.submit(batch_task, batch))
            
    #     for i in range(0, len(benign_files), batch_size):
    #         batch = benign_files[i : i+batch_size]
    #         tasks.add(executor.submit(batch_task, batch))
    # except BrokenProcessPool:
    #     print("################## BrokenProcessPool ERROR!!! ################## ")

    # job_count = len(tasks)
    # for future in as_completed(tasks):
    #     try:
    #         future.result()
    #     except BrokenProcessPool:
    #         print("##################  BrokenProcessPool ERROR!!! ################## ")
    #     job_count -= 1
    #     print("=============================================")
    #     print("== One Job Done, Remaining Job Count: %s ===" % (job_count))
    #     print("=============================================")


if __name__ == "__main__":
    main()



