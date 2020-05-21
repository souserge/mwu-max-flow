import numpy as np
from argparse import ArgumentParser
from functools import reduce
import math
from pprint import PrettyPrinter
import json


def get_path(prev, edge_dict, node):
    prev_node = prev[node]
    if prev_node is None:
        return []
    else:
        path = get_path(prev, edge_dict, prev_node)
        edge_idx = edge_dict.get((prev_node, node))
        return path + [edge_idx]


def find_shortest_path(net, edge_dict, edge_w, source, sink):
    """
    Uses the Bellman–Ford algorithm to find the shortest path from source to sink in O(V*E)
    """
    num_vertices = len(net)
    dist = {key: math.inf for key in net.keys()}
    prev = {key: None for key in net.keys()}

    dist[source] = 0

    for _ in range(num_vertices - 1):
        for u in net:
            for v in net[u].keys():
                w = edge_w[edge_dict.get((u, v))]
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u

    return get_path(prev, edge_dict, sink), dist[sink]


def mwu_init(edges, delta):
    return np.full((edges.shape[0]), delta)


def mwu_update_weights(weights, capacities, path, min_cap, eps):
    M_path = min_cap/capacities[path]
    weights[path] = weights[path]*(1 + eps)**(M_path)

    return weights


def net_to_edges(net):
    edge_list = reduce(
        lambda E, u: E + [(u[0], v, c) for v, c in u[1].items()],
        net.items(),
        [])

    edges = np.array([[x[0], x[1]] for x in edge_list])
    capacities = np.array([x[2] for x in edge_list])
    return edges, capacities


def get_flow_net(flow, edge_dict):
    flow_net = {}
    for key in edge_dict:
        flow_net[key] = flow[edge_dict[key]]

    return flow_net


def max_flow(net, source, sink, eps=.5):

    edges, capacities = net_to_edges(net)
    edge_dict = {(e[0], e[1]): idx for idx, e in enumerate(edges)}

    delta = (1+eps)*((1+eps)*len(edge_dict))**(-1/eps)
    weights = mwu_init(edges, delta)

    flow_val = 0
    flow = np.zeros_like(weights)

    while not np.any(weights >= 1):
        edge_w = weights/capacities
        min_path, min_path_dist = find_shortest_path(
            net, edge_dict, edge_w, source, sink)

        min_cap = np.min(capacities[min_path])
        weights = mwu_update_weights(
            weights, capacities, min_path, min_cap, eps)

        flow[min_path] += min_cap

    flow /= math.log((1+eps)/delta, 1+eps)
    flow_val = np.sum(flow[edges[:, 0] == source])
    flow_net = get_flow_net(flow, edge_dict)

    return (flow_net, flow_val)


def main():
    pp = PrettyPrinter()

    parser = ArgumentParser(
        description='The max flow algorithm using the MWU method')

    parser.add_argument('--eps', type=float, default=.1,
                        help='the epsilon parameter of the MWU method')

    parser.add_argument('--file', type=str,
                        help='Number of breeds to be selected', default='network.json')

    args = parser.parse_args()
    if args.eps < 0 or args.eps > 0.5:
        raise ValueError('Argument --eps must be within 0 and 1/2')

    with open(args.file, 'r') as inp:
        flow_net = json.load(inp)

        flow, flow_val = max_flow(
            flow_net['net'], flow_net['source'], flow_net['sink'], eps=args.eps)

        print('(Approximately) optimal flow value: {0:.2f}'.format(flow_val))
        print('Flow:')
        pp.pprint(flow)


if __name__ == '__main__':
    main()
