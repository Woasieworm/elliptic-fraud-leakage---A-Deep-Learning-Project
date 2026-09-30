import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import degree,add_self_loops
import torch.nn.functional as f
from sklearn.metrics import  f1_score
from src.utils import get_device, seed_everything

DEVICE = get_device()

class EvolveGCNOLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.W0 = nn.Parameter(
            torch.empty(in_channels, in_channels)
        )
        nn.init.xavier_uniform_(self.W0)

        self.gru = nn.GRUCell(
            in_channels,
            in_channels
        )

        self.W = None

    def reset_state(self):
        # Start every sequence from the learned initial matrix.
        self.W = self.W0.clone()

    def forward(self, x, edge_index):
        if self.W is None:
            self.reset_state()

        # Graph-level summary used to evolve the weight matrix.
        summary = x.mean(dim=0)

        # Update every row of W using the same graph summary.
        rows = []
        for row in self.W:
            rows.append(self.gru(summary.unsqueeze(0), row.unsqueeze(0)).squeeze(0))

        self.W = torch.stack(rows, dim=0)

        x = x @ self.W.T

        # GCN-style propagation.
        edge_index, _ = add_self_loops(
            edge_index,
            num_nodes=x.size(0)
        )

        src, dst = edge_index

        deg = degree(
            dst,
            x.size(0),
            dtype=x.dtype
        )

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        out = torch.zeros_like(x)
        out.index_add_(
            0,
            dst,
            norm.unsqueeze(1) * x[src]
        )

        return out


class EvolveGCN( nn.Module ) :
  def __init__ ( self , in_channels , hidden = 64 , out = 2 , dropout = 0.50 ) :
    super().__init__()

    self.conv1 = EvolveGCNOLayer( in_channels )
    self.lin1 = nn.Linear( in_channels , hidden )

    self.conv2 = EvolveGCNOLayer( hidden )
    self.lin2 = nn.Linear( hidden , out )

    self.dropout = dropout

  def reset_state ( self ) :
    self.conv1.reset_state()
    self.conv2.reset_state()

  def forward ( self , x , edge_index ) :
    x = self.conv1( x , edge_index )
    x = f.relu( self.lin1( x ) )
    x = f.dropout(
      x ,
      p = self.dropout ,
      training = self.training
    )

    x = self.conv2( x , edge_index )
    return self.lin2( x )




def build_snapshots ( data ) :
  # ========================================================
  # This is the custom snapshot-building component.
  # Each snapshot contains nodes from one time step and
  # edges whose endpoints are in that same snapshot.
  # ========================================================

  snapshots = [ ]

  for t in sorted( data.time_step.unique().cpu().tolist() ) :
    node_mask = data.time_step == t
    node_idx = node_mask.nonzero( as_tuple = True )[ 0 ]

    if len( node_idx ) == 0 :
      continue

    global_to_local = torch.full(
      (data.num_nodes ,) ,
      -1 ,
      dtype = torch.long
    )
    global_to_local[ node_idx ] = torch.arange(
      len( node_idx )
    )

    src , dst = data.edge_index
    edge_mask = node_mask[ src ] & node_mask[ dst ]

    local_edges = global_to_local[
      data.edge_index[ : , edge_mask ]
    ]

    snapshots.append( {
      "time" : int( t ) ,
      "node_idx" : node_idx ,
      "x" : data.x[ node_idx ] ,
      "y" : data.y[ node_idx ] ,
      "edge_index" : local_edges ,
      "train_mask" : data.train_mask[ node_idx ] ,
      "val_mask" : data.val_mask[ node_idx ] ,
      "test_mask" : data.test_mask[ node_idx ]
    } )

  return snapshots






def train_evolvegcn ( data , seed , split_name , epochs = 100 ) :
  seed_everything( seed )

  model = EvolveGCN(
    in_channels = data.x.shape[ 1 ] ,
    hidden = 64
  ).to( DEVICE )

  snapshots = build_snapshots( data )

  y_train = data.y[ data.train_mask ]

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

  train_end = int( data.time_step[ data.train_mask ].max() )
  val_start = int( data.time_step[ data.val_mask ].min() )
  val_end = int( data.time_step[ data.val_mask ].max() )

  if split_name == "random" :
    training_times = set( int( s[ "time" ] ) for s in snapshots )
  else :
    training_times = set( range( 1 , train_end + 1 ) )

  best_val_f1 = -1
  best_state = None

  for epoch in range( epochs ) :
    # ----------------------------------------------------
    # TRAIN SEQUENCE
    # ----------------------------------------------------
    model.train()
    model.reset_state()
    optimizer.zero_grad()

    train_losses = [ ]

    for snap in snapshots :
      t = snap[ "time" ]

      if t not in training_times :
        continue

      x = snap[ "x" ].to( DEVICE )
      edge_index = snap[ "edge_index" ].to( DEVICE )
      y = snap[ "y" ].to( DEVICE )
      train_mask = snap[ "train_mask" ].to( DEVICE )

      if not train_mask.any() :
        # Still evolve the model state through the snapshot.
        _ = model( x , edge_index )
        continue

      logits = model( x , edge_index )

      loss = f.cross_entropy(
        logits[ train_mask ] ,
        y[ train_mask ] ,
        weight = class_weight
      )

      train_losses.append( loss )

    if train_losses :
      total_loss = torch.stack( train_losses ).mean()
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_( model.parameters() , max_norm = 1.0 )
      
      optimizer.step()

    # ----------------------------------------------------
    # VALIDATION SEQUENCE
    # ----------------------------------------------------
    model.eval()
    model.reset_state()

    val_true = [ ]
    val_pred = [ ]

    with torch.no_grad() :
      for snap in snapshots :
        t = snap[ "time" ]

        # For temporal evaluation, the recurrent state must
        # see the past before it predicts the future.
        if split_name == "temporal" :
          if t > val_end :
            break
          if t < val_start :
            x = snap[ "x" ].to( DEVICE )
            ei = snap[ "edge_index" ].to( DEVICE )
            _ = model( x , ei )
            continue
        else :
          # Random pipeline uses the complete sequence.
          pass

        x = snap[ "x" ].to( DEVICE )
        edge_index = snap[ "edge_index" ].to( DEVICE )
        mask = snap[ "val_mask" ].to( DEVICE )

        logits = model( x , edge_index )

        if mask.any() :
          val_pred.extend(
            logits[ mask ].argmax( 1 ).cpu().numpy()
          )
          val_true.extend(
            snap[ "y" ][ mask.cpu() ].numpy()
          )

    val_f1 = (
      f1_score(
        val_true ,
        val_pred ,
        pos_label = 1 ,
        zero_division = 0
      )
      if val_true else 0.0
    )

    if val_f1 > best_val_f1 :
      best_val_f1 = val_f1
      best_state = {
        k : v.detach().cpu().clone()
        for k , v in model.state_dict().items()
      }

  model.load_state_dict( best_state )
  model.eval()

  # --------------------------------------------------------
  # FINAL TEST SEQUENCE
  # --------------------------------------------------------
  model.reset_state()

  test_true = [ ]
  test_pred = [ ]

  with torch.no_grad() :
    for snap in snapshots :
      t = snap[ "time" ]

      # In the temporal experiment we must advance through
      # all observed history before making test predictions.
      if split_name == "temporal" and t < 40 :
        x = snap[ "x" ].to( DEVICE )
        ei = snap[ "edge_index" ].to( DEVICE )
        _ = model( x , ei )
        continue

      mask = snap[ "test_mask" ].to( DEVICE )

      x = snap[ "x" ].to( DEVICE )
      ei = snap[ "edge_index" ].to( DEVICE )

      logits = model( x , ei )

      if mask.any() :
        test_pred.extend(
          logits[ mask ].argmax( 1 ).cpu().numpy()
        )
        test_true.extend(
          snap[ "y" ][ mask.cpu() ].numpy()
        )

  return model , np.asarray( test_pred ) , np.asarray( test_true )