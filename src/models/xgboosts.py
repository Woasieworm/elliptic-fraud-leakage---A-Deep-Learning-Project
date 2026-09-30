import xgboost as xgb



'''
  IMPLEMENTATION FOR XGBOOST
  
  XGBoost is the tree-based machine-learning baseline.
  
  The validation set is used during fitting, while the final reported metrics are calculated only on the test set.
'''

def train_xgboost ( data , seed,split_name ) :
  x = data.x.numpy()
  y = data.y.numpy()
  
  train = data.train_mask.numpy()
  valuate = data.val_mask.numpy()
  test = data.test_mask.numpy()
  
  x_train , y_train = x[ train ] , y[ train ]
  x_val , y_val = x[ valuate ] , y[ valuate ]
  x_test , y_test = x[ test ] , y[ test ]
  
  neg = (y_train == 0).sum()
  pos = (y_train == 1).sum()
  scale_pos_weight = neg / max( pos , 1 )

  
  

  model = xgb.XGBClassifier(
    n_estimators = 300 ,
    max_depth = 6 ,
    learning_rate = 0.05 ,
    subsample = 0.9 ,
    colsample_bytree = 0.9 ,
    scale_pos_weight = scale_pos_weight ,
    eval_metric = "logloss" ,
    random_state = seed ,
    verbosity = 0 ,
    n_jobs = 1 ,
    tree_method = "hist"
  )

  model.fit(
    x_train ,
    y_train ,
    eval_set = [ (x_val , y_val) ] ,
    verbose = False
  )
  
  pred = model.predict( x_test )
  
  return model, pred , y_test


def predict_xgb_all_nodes ( model , data ) :
  return model.predict( data.x.numpy() )