import pandas as pd
import numpy as np
import torch

kaggle_headline_data = pd.read_csv('kaggle_data/kaggle_headlines_with_sentiment_and_derived_market_features_and_targets.csv')
synthetic_data = pd.read_csv("synthetic_data/consolidated_data/consolidated_with_sentiment.txt", sep="|")
print(kaggle_headline_data.head())
print(kaggle_headline_data.columns)
characters_to_replace_with_dash = ['–', '—']
characters_to_replace_with_single_quote = ['`', '‘', '’', '•', '…']
characters_to_replace_with_double_quote = ['“', '”', '″']
characters_to_replace_with_delimiter = ['[', ']', '{', '}', '(', ')']
characters_to_replace_with_comparison = ['<', '>', '=', '≠', '≤', '≥']
nonstandard_characters = ['`', '¥', '®', 'ا', 'ج', 'ح', 'ع', 'ف', 'ل', 'م', 'ن', 'ي', 'अ', 'आ', 'इ', 'ए', 'औ', 'क', 'ख', 'ग', 'च', 'ज', 'ट', 'ड', 'त', 'थ', 'द', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'स', 'ह', 'ा', 'ि', 'ी', 'ो', '–', '—', '‘', '’', '“', '”', '•', '…', '″', '€', '™']

# pretreat the 'Title' data to replace the prescribed characters
for char in characters_to_replace_with_dash:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '-', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '-', regex=False)
for char in characters_to_replace_with_single_quote:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, "'", regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, "'", regex=False)
for char in characters_to_replace_with_double_quote:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '"', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '"', regex=False)
for char in nonstandard_characters:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<UNK>', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '<UNK>', regex=False)
for char in characters_to_replace_with_delimiter:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<DELIM>', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '<DELIM>', regex=False)
for char in characters_to_replace_with_comparison:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<COMPAR>', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '<COMPAR>', regex=False)

kaggle_headline_data['Date'] = pd.to_datetime(kaggle_headline_data['Date'])
bad_nans = kaggle_headline_data[[
    'SP500_Adj_Close',
    'SP500_1_ago',
    'SP500_2_ago',
    'SP500_3_ago',
    'SP500_1_ahead',
    'SP500_2_ahead',
    'SP500_3_ahead',
    'SP500_1_day_return_lag_1',
    'SP500_1_day_return_lag_2',
    'SP500_1_day_return_lag_3',
]].isna().any(axis=1)
kaggle_headline_data = kaggle_headline_data[~bad_nans].reset_index(drop=True)

kaggle_headline_data = kaggle_headline_data.sort_values(by='Date', ascending=True).reset_index(drop=True)

date_56_pctile = kaggle_headline_data['Date'].quantile(0.56)
date_80_pctile = kaggle_headline_data['Date'].quantile(0.80)

holdout_data = kaggle_headline_data[kaggle_headline_data['Date'] > date_80_pctile].reset_index(drop=True)
test_data = kaggle_headline_data[(kaggle_headline_data['Date'] > date_56_pctile) & (kaggle_headline_data['Date'] <= date_80_pctile)].reset_index(drop=True)
train_data = kaggle_headline_data[kaggle_headline_data['Date'] <= date_56_pctile].reset_index(drop=True)

train_x = train_data[[
    'Date',
    'Title',
    'SP500_Adj_Close',
    'SP500_1_day_return_forward',
]]
test_x = test_data[[
    'Date',
    'Title',
    'SP500_Adj_Close',
    'SP500_1_day_return_forward',
]]

train_x = train_x[['Date','SP500_Adj_Close','SP500_1_day_return_forward']].copy().drop_duplicates().reset_index(drop=True)
test_x = test_x[['Date','SP500_Adj_Close','SP500_1_day_return_forward']].copy().drop_duplicates().reset_index(drop=True)

train_x['time_in_days'] = (train_x['Date'] - train_x['Date'].min()).dt.days
train_x['days_elapsed'] = train_x['time_in_days'].diff()
train_x['log_SP500_Adj_Close'] = np.log(train_x['SP500_Adj_Close'])

# initial guesses for mu and sigma from hierarchical approach - fit gbm with MLE
mu_guess = torch.tensor(0.0029898293480198263, dtype=torch.float32, requires_grad=True)
sigma_guess = torch.tensor(0.0735540618142129, dtype=torch.float32, requires_grad=True)
theta_guess = torch.tensor(0.05, dtype=torch.float32, requires_grad=True)
alpha_guess = torch.tensor(0.001, dtype=torch.float32, requires_grad=True)
rho_guess = torch.tensor(-0.1, dtype=torch.float32, requires_grad=True)

initial_y_guess = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# initial covariance matrix - if real x is lower than initial_ln_x_guess, y is greater than 0.
# if real x is higher than initial_ln_x_guess, y is less than 0.
initial_ln_x_guess_variance = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
initial_y_guess_variance = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
initial_ln_x_y_guess_covariance = torch.tensor(-0.15, dtype=torch.float32, requires_grad=True)
observation_matrix = torch.tensor([[1.0, 1.0]], dtype=torch.float32)  # shape (1,2)
observation_noise_variance = torch.tensor([[0.01]], dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([mu_guess, sigma_guess, theta_guess, alpha_guess, rho_guess, initial_y_guess, initial_ln_x_guess_variance, initial_y_guess_variance, initial_ln_x_y_guess_covariance, observation_noise_variance], lr=0.01)

for epoch in range(100):
    initial_ln_x_guess = torch.tensor(train_x['log_SP500_Adj_Close'].iloc[0] - initial_y_guess, dtype=torch.float32)
    state = torch.tensor([[initial_ln_x_guess.item()], [initial_y_guess.item()]], dtype=torch.float32)  # shape (2,1)
    state_covariance_matrix = torch.tensor([
        [initial_ln_x_guess_variance, initial_ln_x_y_guess_covariance],
        [initial_ln_x_y_guess_covariance, initial_y_guess_variance]
    ], dtype=torch.float32)  # shape (2,2)

    # create variables for nll loss accumulation - we'll take gradients wrt these later
    nll_observations = []
    nll_latents = []

    # kalman filter forward pass
    for time_idx in range(1, len(train_x)):
        # extrapolate from previous state
        delta_t = train_x['days_elapsed'].iloc[time_idx]
        F_t = torch.tensor([
            [1.0, 0.0],
            [0.0, torch.exp(-theta_guess * delta_t)]
        ], dtype=torch.float32)
        B_t = torch.tensor([
            [(mu_guess - 0.5 * sigma_guess**2) * delta_t],
            [0.0]
        ], dtype=torch.float32)
        Q_t = torch.tensor([
            [sigma_guess**2 * delta_t, (sigma_guess * alpha_guess * rho_guess / theta_guess) * (1 - torch.exp(-theta_guess * delta_t))],
            [(sigma_guess * alpha_guess * rho_guess / theta_guess) * (1 - torch.exp(-theta_guess * delta_t)), (alpha_guess**2 / (2 * theta_guess)) * (1 - torch.exp(-2 * theta_guess * delta_t))]
        ], dtype=torch.float32)
        state = F_t @ state + B_t
        state_covariance_matrix = F_t @ state_covariance_matrix @ F_t.T + Q_t
        # update with observation
        observation = torch.tensor([[train_x['log_SP500_Adj_Close'].iloc[time_idx]]], dtype=torch.float32)
        surprise = observation - observation_matrix @ state # shape (1,1)
        # compute NLL of surprise, which is scalar
        prior_covariance_observation = observation_matrix @ state_covariance_matrix @ observation_matrix.T + observation_noise_variance
        sign, logdet = torch.linalg.slogdet(prior_covariance_observation)
        nll_observation = 0.5 * (logdet + surprise.T @ torch.linalg.inv(prior_covariance_observation) @ surprise + torch.log(torch.tensor(2 * np.pi)))
        nll_observations.append(nll_observation.squeeze())

        kalman_gain_numerator = state_covariance_matrix @ observation_matrix.T
        kalman_gain_denominator = prior_covariance_observation
        kalman_gain = kalman_gain_numerator @ torch.linalg.inv(kalman_gain_denominator)
        # derive posterior state
        posterior_state = state + kalman_gain @ surprise

        # before updating state covariance, calculate NLL of posterior estimate under prior
        latent_surprise = kalman_gain @ surprise
        sign, logdet = torch.linalg.slogdet(state_covariance_matrix)
        nll_latent = 0.5 * (logdet + latent_surprise.T @ torch.linalg.inv(state_covariance_matrix) @ latent_surprise + torch.log(torch.tensor(2 * np.pi)))
        nll_latents.append(nll_latent.squeeze())

        # update state covariance
        state_covariance_matrix = (torch.eye(2) - kalman_gain @ observation_matrix) @ state_covariance_matrix @ (torch.eye(2) - kalman_gain @ observation_matrix).T + kalman_gain @ observation_noise_variance @ kalman_gain.T

        # set current state to posterior
        state = posterior_state

    # loss is total negative log likelihood of observed data and estimated latent states
    nll_observations_total = torch.stack(nll_observations).sum()
    nll_latents_total = torch.stack(nll_latents).sum()
    nll_total = nll_observations_total + nll_latents_total
    optimizer.zero_grad()
    nll_total.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch}: NLL={nll_total.item()}, mu={mu_guess.item()}, sigma={sigma_guess.item()}, theta={theta_guess.item()}, alpha={alpha_guess.item()}, rho={rho_guess.item()}, initial_y={initial_y_guess.item()}, state_covariance_matrix={state_covariance_matrix.detach().numpy()}, observation_noise_variance={observation_noise_variance.item()}")
print("Final parameters:")
print(f"mu={mu_guess.item()}, sigma={sigma_guess.item()}, theta={theta_guess.item()}, alpha={alpha_guess.item()}, rho={rho_guess.item()}, initial_y={initial_y_guess.item()}, state_covariance_matrix={state_covariance_matrix.detach().numpy()}, observation_noise_variance={observation_noise_variance.item()}")

