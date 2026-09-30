from sklearn.linear_model import LogisticRegression


'''
  IMPLEMENTATION FOR LOGISTIC REGRESSION
  
  It uses only the 165 transaction features and does not use graph edges.
'''

def train_logistic_regression ( data , seed , split_name ) :
  x = data.x.numpy()
  y = data.y.numpy()

  train = data.train_mask.numpy()
  test = data.test_mask.numpy()

  model = LogisticRegression(
    max_iter = 1000 ,
    class_weight = "balanced" ,
    random_state = seed
  )

  model.fit( x[ train ] , y[ train ] )

  pred = model.predict( x[ test ] )

  return model , pred , y[ test ]


def predict_lr_all_nodes ( model , data ) :
  return model.predict( data.x.numpy() )