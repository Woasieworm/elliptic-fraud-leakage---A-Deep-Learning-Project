import torch.nn as nn
import torch.nn.functional as f
import torch
from src.utils import seed_everything, get_device
from sklearn.metrics import f1_score


DEVICE = get_device()


'''
  IMPLEMENTATION FOR MLP
  
  The MLP is a non-graph neural-network baseline.
  
  It has access to the transaction features but not to transaction relationships.
'''

class MLP( nn.Module ) :
  def __init__ ( self , in_channels , hidden = 128 , out = 2 , dropout = 0.30 ) :
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear( in_channels , hidden ) ,
      nn.BatchNorm1d( hidden ) ,
      nn.ReLU() ,
      nn.Dropout( dropout ) ,

      nn.Linear( hidden , hidden ) ,
      nn.BatchNorm1d( hidden ) ,
      nn.ReLU() ,
      nn.Dropout( dropout ) ,

      nn.Linear( hidden , out )
    )

  def forward ( self , x ) :
    return self.net( x )


def train_mlp ( data , seed , split_name , epochs = 150 ) :
  seed_everything( seed )

  model = MLP( data.x.shape[ 1 ] ).to( DEVICE )
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

  for epoch in range( epochs ) :
    model.train()
    optimizer.zero_grad()

    logits = model( d.x )

    loss = f.cross_entropy(
      logits[ d.train_mask ] ,
      d.y[ d.train_mask ] ,
      weight = class_weight
    )

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad() :
      val_pred = logits[ d.val_mask ].argmax( 1 ).cpu().numpy()
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

  with torch.no_grad() :
    test_pred = model( d.x )[ d.test_mask ].argmax( 1 ).cpu().numpy()

  return model , test_pred , d.y[ d.test_mask ].cpu().numpy()


def predict_mlp_all_nodes ( model , data ) :
  model.eval()
  with torch.no_grad() :
    return model( data.x.to( DEVICE ) ).argmax( 1 ).cpu().numpy()