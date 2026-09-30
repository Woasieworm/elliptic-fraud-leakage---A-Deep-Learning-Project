from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as f
from sklearn.metrics import f1_score

from src.utils import get_device, seed_everything, causal_edge_index


DEVICE = get_device()


'''
  IMPLEMENTATION FOR GCN

  GCN is the first graph neural network in the experiment.

  Random split
  The conventional random experiment uses the complete graph.

  Temporal split
  The model is trained using graph information available through time step 34. Validation and test predictions are evaluated using causal graphs at their respective time steps.
'''


class GCN( nn.Module ) :
  def __init__ ( self , in_channels , hidden = 128 , out = 2 , dropout = 0.50 ) :
    super().__init__()

    self.conv1 = GCNConv( in_channels , hidden )
    self.conv2 = GCNConv( hidden , out )
    self.dropout = dropout

  def forward ( self , x , edge_index ) :
    x = self.conv1( x , edge_index )
    x = f.relu( x )
    x = f.dropout( x , p = self.dropout , training = self.training )
    return self.conv2( x , edge_index )



class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden=128, out=2, dropout=0.50):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def temporal_gnn_predictions ( model , data , mask , start_time , end_time ) :
  model.eval()

  predictions = [ ]
  truths = [ ]

  with torch.no_grad() :
    x = data.x.to( DEVICE )

    for t in range( start_time , end_time + 1 ) :
      node_mask = mask & (data.time_step == t)

      if not node_mask.any() :
        continue

      edge_index_t = causal_edge_index( data , t ).to( DEVICE )
      logits = model( x , edge_index_t )

      pred = logits[ node_mask.to( DEVICE ) ].argmax( 1 ).cpu().numpy()
      true = data.y[ node_mask ].cpu().numpy()

      predictions.extend( pred )
      truths.extend( true )

  return np.asarray( truths ) , np.asarray( predictions )


def train_gnn ( model_class , data , seed , split_name , epochs = 150 ) :
  seed_everything( seed )

  model = model_class(
    in_channels = data.x.shape[ 1 ] ,
    hidden = 128
  ).to( DEVICE )

  d = data.clone().to( DEVICE )

  y_train = d.y[ d.train_mask ]

  class_weight = torch.tensor(
    [
      1.0 ,
      ((y_train == 0).sum() / (y_train == 1).sum()).item()
    ] ,
    dtype = torch.float32 ,
    device = DEVICE
  )

  optimizer = torch.optim.Adam(
    model.parameters() ,
    lr = 1e-3 ,
    weight_decay = 5e-4
  )

  best_val_f1 = -1
  best_state = None

  if split_name == "temporal" :
    train_end = int( d.time_step[ d.train_mask ].max().item() )
    val_end = int( d.time_step[ d.val_mask ].max().item() )
    train_edges = causal_edge_index( d , train_end ).to( DEVICE )
  else :
    train_edges = d.edge_index.to( DEVICE )

  for epoch in range( epochs ) :
    model.train()
    optimizer.zero_grad()

    logits = model( d.x , train_edges )

    loss = f.cross_entropy(
      logits[ d.train_mask ] ,
      d.y[ d.train_mask ] ,
      weight = class_weight
    )

    loss.backward()
    optimizer.step()

    # Validation
    if split_name == "temporal" :
      val_true , val_pred = temporal_gnn_predictions(
        model ,
        d ,
        d.val_mask ,
        train_end + 1 ,
        val_end
      )
    else :
      model.eval()
      with torch.no_grad() :
        val_logits = model( d.x , d.edge_index )
        val_pred = val_logits[ d.val_mask ].argmax( 1 ).cpu().numpy()
        val_true = d.y[ d.val_mask ].cpu().numpy()

    val_f1 = f1_score(
      val_true ,
      val_pred ,
      pos_label = 1 ,
      zero_division = 0
    )

    if val_f1 > best_val_f1 :
      best_val_f1 = val_f1
      best_state = {
        k : v.detach().cpu().clone()
        for k , v in model.state_dict().items()
      }

  model.load_state_dict( best_state )
  model.eval()

  # Final test
  if split_name == "temporal" :
    test_start = val_end + 1
    test_end = int( d.time_step[ d.test_mask ].max().item() )

    test_true , test_pred = temporal_gnn_predictions(
      model ,
      d ,
      d.test_mask ,
      test_start ,
      test_end
    )
  else :
    with torch.no_grad() :
      test_logits = model( d.x , d.edge_index )
      test_pred = test_logits[ d.test_mask ].argmax( 1 ).cpu().numpy()
      test_true = d.y[ d.test_mask ].cpu().numpy()

  return model , test_pred , test_true