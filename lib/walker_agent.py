""" Agent architectures and training methods for optimal greedy graph traversal """
import torch, torch.nn as nn
from collections import namedtuple, defaultdict
from .nn_utils import GraphConvolutionBlock
from .nn_gat_utils import SpGAT,GAT
from .nn_gsg_utils import GraphSage
import numpy as np

class BaseWalkerAgent(nn.Module):
    # State is arbitrary information that agent needs to compute about its graph. Immutable.
    State = namedtuple("AgentState", ['vertices', 'edges'])

    def prepare_state(self, graph, **kwargs):
        """ Pre-computes graph representation for further use """
        return self.State(vertices=graph.vertices, edges=graph.edges)

    def get_query_vectors(self, queries, *, state, device='cuda', **kwargs):
        """ Return vector representation for queries """
        return queries.to(device=device)

    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """ Return vector representation for vertices """
        return state.vertices[vertex_ids, :].to(device=device)


class SimpleWalkerAgent(BaseWalkerAgent):
    def __init__(self, vertex_size, hidden_size, output_size, activation=nn.ELU(),
                 residual=False, project_query=True):
        """
        An agent that projects both vertex and query to a space with given dimensionality
        :param vertex_size: input size of graph vertices and queries
        :param hidden_size: intermediate layer size
        :param output_size: output dimension size
        """
        super().__init__()
        assert (not residual) or (output_size == vertex_size), 'residual can only be used if output size == vertex size'
        assert (project_query or (vertex_size == output_size)), 'need to project query to output size'
        self.residual, self.project_query = residual, project_query
        self.network = nn.Sequential(
            nn.Linear(vertex_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
        )
        if project_query:
            self.query_proj = nn.Linear(vertex_size, output_size,
                                        bias=output_size != vertex_size)  # use bias only in a dimension reduction case

    def get_query_vectors(self, queries, *, state, device='cuda', **kwargs):
        """ Return vector representation for queries """
        q = query_vec = queries.to(device=device)
        if self.project_query:
            query_vec = self.query_proj(q)
            if self.residual:
                query_vec += q
        return query_vec

    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """
        :param vertex_ids: indices of vertices to embed [batch_size]
        :return: vector representation for vertices shape: [batch_size, output_size]
        """
        vertices = state.vertices[vertex_ids, :].to(device=device)
        vectors = self.network(vertices)
        if self.residual:
            vectors += vertices
        return vectors


class GATWalkerAgent(SimpleWalkerAgent):
    State = namedtuple("AgentState", ['vertices', 'hidden'])

    def __init__(self, nfeat, nhid, hidden_size, nclass, dropout, alpha, nheads, activation=nn.ELU(),residual=False, project_query=True, **kwargs):
        """ Walker agent with graph-attention network """
        super().__init__(nfeat,  hidden_size, nclass)
        assert (not residual) or (nclass == nfeat), 'residual can only be used if output size == vertex size'
        assert (project_query or (nfeat == nclass)), 'need to project query to output size'
        self.residual, self.project_query = residual, project_query
        self.block = SpGAT(nfeat, nhid, nclass, dropout, alpha, nheads)
        if project_query:
            self.query_proj = nn.Linear(nfeat, nclass,
                                        bias=nclass != nfeat) 

    def prepare_state(self, graph, device='cuda', **kwargs):
        """ Pre-computes graph representation for further use in edge prediction """
        adj_edges = torch.tensor([(from_i, to_i) for from_i in graph.edges
                                  for to_i in list(graph.edges[from_i]) + [from_i]], device=device)
        # d = torch.tensor([(len(graph.edges[from_i]) + 1) for from_i in graph.edges], device=device)
        # d = 1. / d.type(torch.float32)
        # adj_values = d[adj_edges[:, 0]]
        # adj = torch.sparse_coo_tensor(adj_edges.t(), adj_values, dtype=torch.float32, device=device)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        # idx_map = {j: i for i, j in enumerate(idx)}
        # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # # build symmetric adjacency matrix
        # adj = adj + adj.T.mulstiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj = torch.FloatTensor(np.array(adj.todense()))
        
        # print(adj_edges)
        hidden = graph.vertices.to(device)
        hidden = self.block(hidden,adj_edges)
        return self.State(vertices=graph.vertices, hidden=hidden)
    
    
    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """
        :param vertex_ids: indices of vertices to embed [batch_size]
        :return: vector representation for vertices shape: [batch_size, output_size]
        """
        hidden = state.hidden[vertex_ids, :].to(device=device)
        vectors = self.network(hidden)
        if self.residual:
            vectors += state.vertices[vertex_ids, :].to(device=device)
        return vectors
    
class GCNWalkerAgent(SimpleWalkerAgent):
    State = namedtuple("AgentState", ['vertices', 'hidden'])

    def __init__(self, vertex_size, conv_hid_size, hidden_size, output_size, activation=nn.ELU(),
                 convolution_blocks=3, residual_conv=True, normalize_out=True, **kwargs):
        """ Walker agent with graph-convolutional network """
        super().__init__(vertex_size, hidden_size, output_size, activation=activation, **kwargs)
        self.blocks = [
            GraphConvolutionBlock(vertex_size, hid_size=conv_hid_size, out_size=vertex_size,
                                  residual=residual_conv, normalize_out=normalize_out, activation=activation)
            for i in range(convolution_blocks)
        ]
        for i, block in enumerate(self.blocks):
            self.add_module('block%i' % i, block)

    def prepare_state(self, graph, device='cuda', **kwargs):
        """ Pre-computes graph representation for further use in edge prediction """
        adj_edges = torch.tensor([(from_i, to_i) for from_i in graph.edges
                                  for to_i in list(graph.edges[from_i]) + [from_i]], device=device)
        d = torch.tensor([(len(graph.edges[from_i]) + 1) for from_i in graph.edges], device=device)
        d = 1. / d.type(torch.float32)
        adj_values = d[adj_edges[:, 0]]

        adj = torch.sparse_coo_tensor(adj_edges.t(), adj_values, dtype=torch.float32, device=device)
        hidden = graph.vertices.to(device)
        for block in self.blocks:
            hidden = block(hidden, adj)
        return self.State(vertices=graph.vertices, hidden=hidden)

    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """
        :param vertex_ids: indices of vertices to embed [batch_size]
        :return: vector representation for vertices shape: [batch_size, output_size]
        """
        hidden = state.hidden[vertex_ids, :].to(device=device)
        vectors = self.network(hidden)
        if self.residual:
            vectors += state.vertices[vertex_ids, :].to(device=device)
        return vectors


class NoProjAgent(BaseWalkerAgent):
    """ Agent wrapper that does not compute query projection at inference time """
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

        matrix = self.agent.query_proj.weight.t()
        if self.agent.residual:
            matrix = matrix + torch.eye(matrix.shape[0]).to(device=matrix.device)
        self.query_additive = torch.mm(self.agent.query_proj.bias[None, :], matrix.inverse())

    def prepare_state(self, graph, **kwargs):
        return self.agent.prepare_state(graph, **kwargs)

    def get_query_vectors(self, queries, *, state, device='cuda', **kwargs):
        """ Return vector representation for queries """
        return queries.to(device=device) + self.query_additive

    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """ Return vector representation for vertices """
        vectors = self.agent.get_vertex_vectors(vertex_ids, state=state, device=device, **kwargs)
        matrix = self.agent.query_proj.weight.t()
        if self.agent.residual:
            matrix = matrix + torch.eye(matrix.shape[0]).to(device=matrix.device)
        return vectors.mm(matrix.t())


class PrecomputedWalkerAgent(BaseWalkerAgent):
    """ Agent walker that uses pre-computed edge vectors """
    State = namedtuple("AgentState", ['vertex_vectors'])

    def __init__(self, agent, batch_size=None):
        """ :type agent: BaseWalkerAgent """
        super().__init__()
        self.agent = agent
        self.batch_size = batch_size

    def prepare_state(self, graph, batch_size=None, state=None, **kwargs):
        vertex_ids = list(graph.edges.keys())
        vertex_vectors = []
        batch_size = batch_size or self.batch_size or len(vertex_ids)
        with torch.no_grad():
            state = state or self.agent.prepare_state(graph, **kwargs)
            for batch_start in range(0, len(vertex_ids), batch_size):
                batch_vectors = self.agent.get_vertex_vectors(vertex_ids[batch_start: batch_start + batch_size],
                                                              state=state, **kwargs)
                vertex_vectors.extend(batch_vectors.data.cpu().numpy().tolist())

        assert len(vertex_vectors) == len(vertex_ids)

        return self.State(vertex_vectors=dict(zip(vertex_ids, vertex_vectors)))

    def get_query_vectors(self, queries, **kwargs):
        return self.agent.get_query_vectors(queries, **kwargs)

    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """ Return vector representation for vertices """
        return [state.vertex_vectors[i] for i in zip(vertex_ids)]

# TODO: Add init and prepare_state for graphsage 
class GSGWalkerAgent(SimpleWalkerAgent):
    State = namedtuple("AgentState", ['vertices', 'hidden'])
    def __init__(self, nfeat, nhid, hidden_size, nclass, activation=nn.ELU(),residual=False, project_query=True, **kwargs):
        """ Walker agent with graph-attention network """
        super().__init__(nfeat,  hidden_size, nclass)
        assert (not residual) or (nclass == nfeat), 'residual can only be used if output size == vertex size'
        assert (project_query or (nfeat == nclass)), 'need to project query to output size'
        self.residual, self.project_query = residual, project_query
        self.block=GraphSage(nfeat,[nhid,nclass],[20,20])
        if project_query:
            self.query_proj = nn.Linear(nfeat, nclass,
                                        bias=nclass != nfeat) 
    
    #TODO get node_feature_list
    def prepare_state(self, graph, device='cuda', **kwargs):
        """ Pre-computes graph representation for further use in edge prediction """
        # adj_edges = torch.tensor([(from_i, to_i) for from_i in graph.edges
        #                           for to_i in list(graph.edges[from_i]) + [from_i]], device=device)

        # batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
        # batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        # batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
        # batch_train_logits = model(batch_sampling_x)
        hidden = graph.vertices.to(device)
        node_feature_list=[]
        hidden = self.block(node_feature_list)
        # hidden = self.block(hidden,adj_edges)
        return self.State(vertices=graph.vertices, hidden=hidden)
    
    
    def get_vertex_vectors(self, vertex_ids, *, state, device='cuda', **kwargs):
        """
        :param vertex_ids: indices of vertices to embed [batch_size]
        :return: vector representation for vertices shape: [batch_size, output_size]
        """
        hidden = state.hidden[vertex_ids, :].to(device=device)
        vectors = self.network(hidden)
        if self.residual:
            vectors += state.vertices[vertex_ids, :].to(device=device)
        return vectors