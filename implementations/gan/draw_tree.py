import argparse
import torch
import os
import sys

import numpy as np
from tqdm import tqdm

import graphviz 
import math

import pickle

from tree import tree
sys.modules['tree'] = tree


parser = argparse.ArgumentParser()
parser.add_argument("--draw_group_size", type=int, default=0, help="divide samples into groups of this size, by default no grouping")
parser.add_argument("--tree_path", type=str, default= "tree/acl5_10k-23-acc-27-bytes-1592431760.25.pkl", help="path to decision tree")
parser.add_argument("--input_file", type=str, help="input file")
parser.add_argument("--output_folder", type=str, help="output folder")
parser.add_argument("--generate_video", default=False, action="store_true", help="generate video")

opt = parser.parse_args()

with open(opt.tree_path, 'rb') as f:
    t = pickle.load(f)


class PacketClassificationBlackbox():
    def __init__(self, decision_tree):
        self._decision_tree = decision_tree

    @staticmethod
    def stat(results):
        print("min: {}, 25%: {}, 50%: {}, 75%: {}, max: {}".format(np.quantile(results, 0), np.quantile(results, 0.25), np.quantile(results, 0.5), np.quantile(results, 0.75), np.quantile(results, 1)))

    def query(self, field_dict_inputs):
        packets = self._binary_to_int(field_dict_inputs)
        times = self._get_classification_time(packets, self._decision_tree)
        PacketClassificationBlackbox.stat(times)
        return times
    
    def test_path(self, field_dict_inputs):
        packets = self._binary_to_int(field_dict_inputs)
        paths = []
        for packet in packets:
            path = self._decision_tree.match_and_get_path(packet)
            paths.append(path)
        return paths

    def _get_classification_time(self, packets, tree):

        depths = []

        for packet in packets:
            depth = tree.match(packet)
        
            depths.append(depth)

        return np.asarray(depths)

    def _count_unique_path(self, packets, tree):
        # group result by depth and count unique path in each group of depth
        pass

    def _binary_to_int(self, inputs):
        packets = []
        for p in inputs:
            packets.append((self._b2i(p['src_ip']), self._b2i(p['dst_ip']), 
                self._b2i(p['src_port']), self._b2i(p['dst_port']),
                self._b2i(p['proto'])))
        return packets

    def _b2i(self, b_list):
        return int("".join(str(x) for x in b_list), 2) 
    
box = PacketClassificationBlackbox(t)

def np_inputs_to_field_dict_inputs(np_inputs):
    fd_inputs = []
    for np_input in np_inputs:
        fd_input = {}
        fd_input['src_ip'] = np_input[0:32]
        fd_input['dst_ip'] = np_input[32:32 + 32]
        fd_input['src_port'] = np_input[32 + 32:32 + 32 + 16]
        fd_input['dst_port'] = np_input[32 + 32 + 16:32 + 32 + 16 + 16]
        fd_input['proto'] = np_input[32 + 32 + 16 + 16:]
        fd_inputs.append(fd_input)
    return fd_inputs


def gen_graph_torch(data2):
    data2 = np_inputs_to_field_dict_inputs(data2)
    paths2 = box.test_path(data2)

    from collections import defaultdict
    nodes_set = t.get_all_nodes()
    edge_pairs = {}
    for e in t.get_all_edges():
        edge_pairs[e] = 0
    for p in paths2:
        p = p.split(',')
        for i in range(len(p) - 1):
            edge_pairs[(int(p[i]), int(p[i+1]))] += 1

    dot = graphviz.Digraph(comment='Decision tree', graph_attr=[("ranksep","3"), ("ratio", "auto"), ("nodesep", "5"), ("root", "0")])

    for n in nodes_set:
        dot.node(str(n), str(n)) 

    for e in edge_pairs.keys():
        dot.edge(str(e[0]), str(e[1]), penwidth=str(max(1,math.log(edge_pairs[e] + 1))), color="green" if edge_pairs[e] > 0 else "red")

    # dot.render(output_file, view=False, format='svg' ,engine='twopi') 
    return dot
    
def gen_graph_by_group(data_path, output_folder, group_size):
    os.makedirs(output_folder, exist_ok=True)
    data = torch.load(data_path)
    data = data.reshape(-1, 103, 2).argmax(-1).numpy()
    if group_size > 0:
        data = data[:data.shape[0] - data.shape[0] % group_size]
        print("data shape: ", data.shape)
        data_groups = np.split(data, data.shape[0] // group_size)
    else:
        data_groups = [data]
    for idx, data_group in tqdm(enumerate(data_groups), total=len(data_groups)):
        output_file = output_folder + "/" + str(idx)
        dot = gen_graph_torch(data_group)
        dot.render(output_file, view=False, format='svg', engine='twopi')
        dot.render(output_file, view=False, format='jpeg', engine='twopi')
        
if __name__ == "__main__":
    gen_graph_by_group(opt.input_file, opt.output_folder, opt.draw_group_size)