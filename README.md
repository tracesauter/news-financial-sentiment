# News Financial Sentiment Project

## Authors

- Georgia Appel
- George Pinel
- Nicolas Parga
- Trace Sauter

## Key Files in Repo:

### Key Files for Conv 1D End 2 End Neural Network
- *news-financial-sentiment/conv_1d_neural_network/conv_1d_headline_preprocess_nn_architecture.py*
  - contains the PyTorch code for the preprocessing layer architecture that was pretrained on synthetic data. this contains the learned embeddings from tokens to a higher dimensional representation, convolutional layers, adaptive average pooling, and a set of fully connected layers
- *news-financial-sentiment/conv_1d_neural_network/pretrain.py*
  - contains the code for pretraining a multitask neural network that starts with the preprocessing layer and then proceeds to process data through two separate heads - one for predicting logits for a masked token and the other for predicting logits for finbert sentiment associated with a headline. this pretraining occurs on synthetic training data and is useful for getting the preprocessing layer to start identifying patterns that are relevant to financial text so that we can use transfer learning to load the learned weights for the preprocessing layer into our actual neural network that learns from our dataset
- *news-financial-sentiment/conv_1d_neural_network/tokenizer.py*
  - contains code for byte pair encoding. this essentially creates the mapping that converts text into tokens
- *news-financial-sentiment/conv_1d_neural_network/train.py*
  - contains code that trains the actual neural network for prediction on our data. handles transfer learning using pretrained weights for the preprocessing layer. utilizes some interesting logic to handle training that processes a variable number of variable-length headlines down to a single output per date in our dataset, and incorporates interesting logic for processing mini-batches of only headlines from a single day, but aggregating losses across mini-batches over a full batch before backpropagating gradients.
- *news-financial-sentiment/conv_1d_neural_network/assess_performance.py*
  - code that loads the learned weights for our neural network that were learned on specific chunks of our data and then evaluates the neural network on all splits of the data so we can look at in sample and out of sample performance

### Key Files for State Space Modeling Approach
- *news-financial-sentiment/ssm1_gbm_ou_multiplier/hierarchical_1_gbm_params.py*
  - fit Geometric Brownian Motion parameters to data using Maximum Likelihood Estimation as a starting point for fitting the more complex model.
- *news-financial-sentiment/ssm1_gbm_ou_multiplier/index.py*
  - implements the Kalman Filter and EM algorithm approach for optimizing parameters of the state space model. performs this "training" procedure on different splits of the data, and then writes inferred latent states (based on using the Kalman Filter with the optimized parameters) on the in sample chunk as well as the next time period (out of sample) chunk of data so that we can use those in a subsequent model to predict our target
- *news-financial-sentiment/ssm1_gbm_ou_multiplier/predict_target.py*
  - uses xgboost to make predictions on the target variable using latent states inferred from the Kalman Filter exercise as regressors to explore whether we were able to infer mean reverting states that have forward looking predictive power.
