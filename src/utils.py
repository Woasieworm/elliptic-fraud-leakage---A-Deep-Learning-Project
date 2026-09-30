import numpy as np
import torch
import random
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

def build_full_graph ( df , edges, feature_cols:list ) :
  node_ids = df[ "txId" ].values
  id_to_idx = { int( nid ) : i for i , nid in enumerate( node_ids ) }

  valid_edges = edges[
    edges[ "txId1" ].isin( id_to_idx ) &
    edges[ "txId2" ].isin( id_to_idx )
    ].copy()

  src = valid_edges[ "txId1" ].map( id_to_idx ).to_numpy()
  dst = valid_edges[ "txId2" ].map( id_to_idx ).to_numpy()

  # Store both directions so message passing is effectively undirected.
  edge_index = torch.tensor(
    np.vstack( [
      np.concatenate( [ src , dst ] ) ,
      np.concatenate( [ dst , src ] )
    ] ) ,
    dtype = torch.long
  )

  return Data(
    x = torch.tensor( df[ feature_cols ].values , dtype = torch.float32 ) ,
    y = torch.tensor( df[ "label" ].values , dtype = torch.long ) ,
    time_step = torch.tensor( df[ "time_step" ].values , dtype = torch.long ) ,
    edge_index = edge_index ,
    tx_id = torch.tensor( df[ "txId" ].values , dtype = torch.long ) ,
    num_nodes = len( df )
  )




def scale_features(data, train_mask):
    d = data.clone()
    scaler = StandardScaler()

    train_idx = train_mask.cpu().numpy()
    all_x = d.x.cpu().numpy()

    scaler.fit(all_x[train_idx])
    d.x = torch.tensor(
        scaler.transform(all_x),
        dtype=torch.float32
    )
    d.scaler = scaler
    return d


def random_split(data, seed=42, train_ratio=0.70, val_ratio=0.10):
    d = data.clone()

    labelled_idx = (d.y != -1).nonzero(as_tuple=True)[0].cpu().numpy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(labelled_idx)

    n_train = int(len(perm) * train_ratio)
    n_val = int(len(perm) * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    d.train_mask = torch.zeros(d.num_nodes, dtype=torch.bool)
    d.val_mask = torch.zeros(d.num_nodes, dtype=torch.bool)
    d.test_mask = torch.zeros(d.num_nodes, dtype=torch.bool)

    d.train_mask[train_idx] = True
    d.val_mask[val_idx] = True
    d.test_mask[test_idx] = True

    return scale_features(d, d.train_mask)


def temporal_split(data, train_end=34, val_end=39):
    d = data.clone()

    labelled = d.y != -1
    t = d.time_step

    d.train_mask = labelled & (t <= train_end)
    d.val_mask = labelled & (t > train_end) & (t <= val_end)
    d.test_mask = labelled & (t > val_end)

    return scale_features(d, d.train_mask)


def causal_edge_index(data, max_time):
    # An edge is usable at time t only if both endpoints
    # are observable by time t.
    src, dst = data.edge_index
    node_time = data.time_step

    keep = (
        (node_time[src] <= max_time) &
        (node_time[dst] <= max_time)
    )

    return data.edge_index[:, keep]


def seed_everything ( seeds: int ) :
  random.seed( seeds )
  np.random.seed( seeds )
  torch.manual_seed( seeds )
  if torch.cuda.is_available() :
    torch.cuda.manual_seed_all( seeds )
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False




def get_device():
  if torch.cuda.is_available() and torch.cuda.get_device_capability()[ 0 ] >= 7 :
    return torch.device( "cuda" )

  return torch.device( "cpu" )



# ============================================================
# 5. EVALUATION HELPERS
# ============================================================

def evaluate_predictions(y_true, y_pred, model_name, split_name, seed):
    return {
        "seed": seed,
        "model": model_name,
        "split": split_name,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def build_split(seed, split_name, data:Data):
    if split_name == "random":
        return random_split(data, seed=seed)
    return temporal_split(data)
