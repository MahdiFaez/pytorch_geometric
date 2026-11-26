# Explaining a GNN on an OWL ontology with an OWL class expression.
#
# Requirements: `pip install rdflib ontolearn`
#
# The family benchmark ontology is converted to an `OWLData` object, an R-GCN
# is trained to recognize aunts, and its predictions are explained twice:
# by `GNNExplainer` (a feature mask) and by `OntolearnExplainer` (a concept).
import json
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.data import OWLData, download_url, extract_zip
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer, OntolearnExplainer
from torch_geometric.nn import RGCNConv

url = 'https://files.dice-research.org/projects/Ontolearn/KGs.zip'
lp_url = ('https://raw.githubusercontent.com/dice-group/Ontolearn/master/'
          'examples/synthetic_problems.json')

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Ontolearn')
owl_path = osp.join(root, 'KGs', 'Family',
                    'family-benchmark_rich_background.owl')
if not osp.exists(owl_path):
    extract_zip(download_url(url, root), root)
with open(download_url(lp_url, root)) as f:
    problem = json.load(f)['problems']['Aunt']

# OWL -> PyG: Individuals become nodes, object properties edge types (plus
# their inverses), and asserted OWL classes multi-hot node features:
data = OWLData.from_owl(owl_path)
data.set_labels_from_examples(problem['positive_examples'],
                              problem['negative_examples'])
print(data)

# Split the labeled individuals into training and test examples:
labeled = data.train_mask.nonzero().view(-1)
perm = labeled[torch.randperm(labeled.numel(),
                              generator=torch.Generator().manual_seed(0))]
num_train = int(0.7 * perm.numel())
train_index, test_index = perm[:num_train], perm[num_train:]


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        return self.conv2(x, edge_index, edge_type)


torch.manual_seed(12345)
model = RGCN(data.num_features, 32, 2, int(data.edge_type.max()) + 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.cross_entropy(out[train_index], data.y[train_index])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
test_acc = (pred[test_index] == data.y[test_index]).float().mean()
print(f'GNN test accuracy: {test_acc:.4f}')

model_config = dict(
    mode='multiclass_classification',
    task_level='node',
    return_type='raw',
)
node_index = int(test_index[data.y[test_index] == 1][0])

# 1. A mask-based explanation works on `OWLData` like on any `Data` object:
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=100),
    explanation_type='model',
    node_mask_type='attributes',
    model_config=model_config,
)
explanation = explainer(data.x, data.edge_index, index=node_index,
                        edge_type=data.edge_type)
importance = explanation.node_mask[node_index]
top_features = importance.topk(3).indices.tolist()
names = [data.schema.feature_names[j].split('#')[-1] for j in top_features]
print(f'GNNExplainer: most important features of node {node_index}: {names}')

# 2. A concept-based explanation of the GNN's predictions, learned on the
# training examples:
explainer = Explainer(
    model=model,
    algorithm=OntolearnExplainer('EvoLearner', data=data,
                                 example_index=train_index, max_runtime=30),
    explanation_type='model',
    node_mask_type='object',
    model_config=model_config,
)
explanation = explainer(data.x, data.edge_index, index=node_index,
                        edge_type=data.edge_type)
print(f'OntolearnExplainer: {explanation.concept} '
      f'(accuracy on training examples: {explanation.concept_accuracy:.4f})')

# How faithfully does the concept describe the GNN on unseen individuals?
covered = explanation.node_mask.view(-1).bool()
fidelity = (covered[test_index] == (pred[test_index] == 1)).float().mean()
print(f'Concept fidelity on test examples: {fidelity:.4f}')
