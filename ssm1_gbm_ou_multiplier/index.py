import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

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

date_20_pctile = kaggle_headline_data['Date'].drop_duplicates().quantile(0.20)
date_40_pctile = kaggle_headline_data['Date'].drop_duplicates().quantile(0.40)
date_60_pctile = kaggle_headline_data['Date'].drop_duplicates().quantile(0.60)
date_80_pctile = kaggle_headline_data['Date'].drop_duplicates().quantile(0.80)

train_chunk_1 = kaggle_headline_data[kaggle_headline_data['Date'] <= date_20_pctile].sort_values(by='Date', ascending=True).reset_index(drop=True)
train_chunk_2 = kaggle_headline_data[(kaggle_headline_data['Date'] <= date_40_pctile) & (kaggle_headline_data['Date'] > date_20_pctile)].sort_values(by='Date', ascending=True).reset_index(drop=True)
train_chunk_3 = kaggle_headline_data[(kaggle_headline_data['Date'] <= date_60_pctile) & (kaggle_headline_data['Date'] > date_40_pctile)].sort_values(by='Date', ascending=True).reset_index(drop=True)
train_chunk_4 = kaggle_headline_data[(kaggle_headline_data['Date'] <= date_80_pctile) & (kaggle_headline_data['Date'] > date_60_pctile)].sort_values(by='Date', ascending=True).reset_index(drop=True)
train_chunk_5 = kaggle_headline_data[kaggle_headline_data['Date'] > date_80_pctile].sort_values(by='Date', ascending=True).reset_index(drop=True)

def ssm_with_em_algorithm(train_data, test_data, out_file_path_params, out_file_path_estimates):
    train_data = train_data[['Date','SP500_Adj_Close','SP500_1_day_return_forward']].copy().drop_duplicates().sort_values(by='Date', ascending=True).reset_index(drop=True)
    test_data = test_data[['Date','SP500_Adj_Close','SP500_1_day_return_forward']].copy().drop_duplicates().sort_values(by='Date', ascending=True).reset_index(drop=True)

    train_data['time_in_days'] = (train_data['Date'] - train_data['Date'].min()).dt.days
    train_data['days_elapsed'] = train_data['time_in_days'].diff()
    train_data['log_SP500_Adj_Close'] = np.log(train_data['SP500_Adj_Close'])

    test_data['time_in_days'] = (test_data['Date'] - test_data['Date'].min()).dt.days
    test_data['days_elapsed'] = test_data['time_in_days'].diff()
    test_data['log_SP500_Adj_Close'] = np.log(test_data['SP500_Adj_Close'])

    # initial guesses for mu and sigma from hierarchical approach - fit gbm with MLE
    mu_guess = torch.tensor(0.0029898293480198263, dtype=torch.float32, requires_grad=True)
    sigma_guess = torch.tensor(0.0735540618142129, dtype=torch.float32, requires_grad=True)
    # initial guesses for theta, alpha, rho and initial y layered on top of gbm parameters
    # these are just guesses based on some degree of intuition with the idea that the EM
    # algorithm will "learn" parameters that fit the model in a way more consistent with
    # observed data
    theta_guess = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    alpha_guess = torch.tensor(0.01, dtype=torch.float32, requires_grad=True)
    rho_guess = torch.tensor(-0.15, dtype=torch.float32, requires_grad=True)

    initial_y_guess = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    # initial covariance matrix - if real x is lower than initial_ln_x_guess, y is greater than 0.
    # if real x is higher than initial_ln_x_guess, y is less than 0.
    initial_ln_x_guess_variance = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
    initial_y_guess_variance = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
    initial_ln_x_y_guess_covariance = torch.tensor(-0.15, dtype=torch.float32, requires_grad=True)
    observation_matrix = torch.tensor([[1.0, 1.0]], dtype=torch.float32)  # shape (1,2)
    observation_noise_variance = torch.tensor([[1e-5]], dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([mu_guess, sigma_guess, theta_guess, alpha_guess, rho_guess, initial_y_guess, initial_ln_x_guess_variance, initial_y_guess_variance, initial_ln_x_y_guess_covariance, observation_noise_variance], lr=2e-6)

    best_loss = float('inf')
    best_params = None

    # We implement the EM algorithm with gradient descent here - basically, we run the kalman filter
    # through each time step in the training data, accumulating the negative log likelihood of the
    # observed data under the prior on the observations and the negative log likelihood of the
    # posterior estimate of the latent state under the prior on the latent state before receiving
    # the observation. Then at the end of running through all of the timesteps, we take the gradient
    # of this total log likelihood with respect to the state space model parameters. We start out with
    # a "true" implementation of the EM algorithm incentivizing the optimizer to maximize the total log
    # likelihood. In later epochs, we mute the importance of the likelihood of the posterior estimate
    # of the latent state under the prior, and place more mathematical emphasis on the likelihood of the
    # observed data under the prior before the observation, because this is the actual predictive task
    # we care about
    for epoch in tqdm(range(400), desc="Training SSM Parameters with EM and Kalman Filter"):
        initial_ln_x_guess = torch.tensor(train_data['log_SP500_Adj_Close'].iloc[0], dtype=torch.float32) - initial_y_guess
        state = torch.stack([initial_ln_x_guess, initial_y_guess]).reshape(2, 1)
        state_covariance_matrix = torch.stack([
            torch.stack([initial_ln_x_guess_variance, initial_ln_x_y_guess_covariance]),
            torch.stack([initial_ln_x_y_guess_covariance, initial_y_guess_variance])
        ])

        # create variables for nll loss accumulation - we'll take gradients wrt these later
        nll_observations = []
        nll_latents = []

        # kalman filter forward pass
        for time_idx in range(1, len(train_data)):
            # extrapolate from previous state
            delta_t = torch.tensor(train_data['days_elapsed'].iloc[time_idx], dtype=torch.float32)
            F_t = torch.stack([
                torch.stack([torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)]),
                torch.stack([torch.tensor(0.0, dtype=torch.float32), torch.exp(-theta_guess * delta_t)])
            ])
            B_t = torch.stack([
                (mu_guess - 0.5 * sigma_guess**2) * delta_t, torch.tensor(0.0, dtype=torch.float32)
            ]).reshape(2,1)
            Q_t = torch.stack([
                torch.stack([
                    sigma_guess**2 * delta_t,
                    (sigma_guess * alpha_guess * rho_guess / theta_guess) * (1 - torch.exp(-theta_guess * delta_t))
                ]),
                torch.stack([
                    (sigma_guess * alpha_guess * rho_guess / theta_guess) * (1 - torch.exp(-theta_guess * delta_t)),
                    (alpha_guess**2 / (2 * theta_guess)) * (1 - torch.exp(-2 * theta_guess * delta_t))
                ])
            ])
            state = F_t @ state + B_t
            state_covariance_matrix = F_t @ state_covariance_matrix @ F_t.T + Q_t
            # update with observation
            observation = torch.tensor([[train_data['log_SP500_Adj_Close'].iloc[time_idx]]], dtype=torch.float32)
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
        if epoch > 175:
            nll_total = 1.05 * nll_observations_total + 0.95 * nll_latents_total
        elif epoch > 275:
            nll_total = 1.1 * nll_observations_total + 0.9 * nll_latents_total
        elif epoch > 350:
            nll_total = 1.15 * nll_observations_total + 0.85 * nll_latents_total
        else:
            nll_total = nll_observations_total + nll_latents_total
        optimizer.zero_grad()
        nll_total.backward()
        optimizer.step()
        # clipping to keep variances positive
        with torch.no_grad():
            sigma_guess.clamp_(min=1e-2)
            alpha_guess.clamp_(min=1e-6)
            initial_ln_x_guess_variance.clamp_(min=1e-10)
            initial_y_guess_variance.clamp_(min=1e-10)
            observation_noise_variance.clamp_(min=1e-10)

        if nll_observations_total.item() < best_loss:
            best_loss = nll_observations_total.item()
            best_params = {
                'mu': mu_guess.item(),
                'sigma': sigma_guess.item(),
                'theta': theta_guess.item(),
                'alpha': alpha_guess.item(),
                'rho': rho_guess.item(),
                'initial_y': initial_y_guess.item(),
                'initial_ln_x_variance': initial_ln_x_guess_variance.item(),
                'initial_y_variance': initial_y_guess_variance.item(),
                'initial_ln_x_y_covariance': initial_ln_x_y_guess_covariance.item(),
                'observation_noise_variance': observation_noise_variance.item(),
            }
    
    # at this point, we've trained the SSM parameters using the EM algorithm. Let's
    # review performance and print out the best parameters found.
    print("\n==============================\n")
    print(f"Best NLL: {best_loss}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("\n==============================\n")
    # and let's save the best parameters to a text file
    with open(out_file_path_params, 'w') as f:
        f.write("Best parameters found by EM algorithm:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    # set parameters to best found values, and infer latent states on the train and test data
    # using the optimized parameters
    mu_guess.data = torch.tensor(best_params['mu'], dtype=torch.float32)
    sigma_guess.data = torch.tensor(best_params['sigma'], dtype=torch.float32)
    theta_guess.data = torch.tensor(best_params['theta'], dtype=torch.float32)
    alpha_guess.data = torch.tensor(best_params['alpha'], dtype=torch.float32)
    rho_guess.data = torch.tensor(best_params['rho'], dtype=torch.float32)
    initial_y_guess.data = torch.tensor(best_params['initial_y'], dtype=torch.float32)
    initial_ln_x_guess_variance.data = torch.tensor(best_params['initial_ln_x_variance'], dtype=torch.float32)
    initial_y_guess_variance.data = torch.tensor(best_params['initial_y_variance'], dtype=torch.float32)
    initial_ln_x_y_guess_covariance.data = torch.tensor(best_params['initial_ln_x_y_covariance'], dtype=torch.float32)
    observation_noise_variance.data = torch.tensor([[best_params['observation_noise_variance']]], dtype=torch.float32)

    train_data['which_split'] = 'train'
    test_data['which_split'] = 'test'
    train_and_test_data = pd.concat([train_data, test_data], ignore_index=True).sort_values(by='Date', ascending=True).reset_index(drop=True)
    # re-compute the time in days and days elapsed
    train_and_test_data['time_in_days'] = (train_and_test_data['Date'] - train_and_test_data['Date'].min()).dt.days
    train_and_test_data['days_elapsed'] = train_and_test_data['time_in_days'].diff()

    train_and_test_data['prior_estimated_log_fair_value'] = -1.0
    train_and_test_data['prior_estimated_ou_exponent'] = -100.0
    train_and_test_data['posterior_estimated_log_fair_value'] = -1.0
    train_and_test_data['posterior_estimated_ou_exponent'] = -100.0
    train_and_test_data['prior_variance_log_fair_value'] = -1.0
    train_and_test_data['prior_variance_ou_exponent'] = -1.0
    train_and_test_data['prior_covariance_log_fair_value_ou_exponent'] = -1.0

    # set the first row values equal to 'log_SP500_Adj_Close' and 0.0
    train_and_test_data.iloc[0, train_and_test_data.columns.get_loc('prior_estimated_log_fair_value')] = train_and_test_data.iloc[0, train_and_test_data.columns.get_loc('log_SP500_Adj_Close')]
    train_and_test_data.iloc[0, train_and_test_data.columns.get_loc('prior_estimated_ou_exponent')] = 0.0
    train_and_test_data.iloc[0, train_and_test_data.columns.get_loc('posterior_estimated_log_fair_value')] = train_and_test_data.iloc[0, train_and_test_data.columns.get_loc('log_SP500_Adj_Close')] - best_params['initial_y']
    train_and_test_data.iloc[0, train_and_test_data.columns.get_loc('posterior_estimated_ou_exponent')] = best_params['initial_y']

    with torch.no_grad():
        initial_ln_x_guess = torch.tensor(train_and_test_data['log_SP500_Adj_Close'].iloc[0], dtype=torch.float32) - initial_y_guess
        state = torch.stack([initial_ln_x_guess, initial_y_guess]).reshape(2, 1)
        state_covariance_matrix = torch.stack([
            torch.stack([initial_ln_x_guess_variance, initial_ln_x_y_guess_covariance]),
            torch.stack([initial_ln_x_y_guess_covariance, initial_y_guess_variance])
        ])

        # create variables for nll loss accumulation - we'll take gradients wrt these later
        nll_observations = []
        nll_latents = []

        # kalman filter forward pass
        for time_idx in range(1, len(train_and_test_data)):
            # extrapolate from previous state
            delta_t = torch.tensor(train_and_test_data['days_elapsed'].iloc[time_idx], dtype=torch.float32)
            F_t = torch.stack([
                torch.stack([torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)]),
                torch.stack([torch.tensor(0.0, dtype=torch.float32), torch.exp(-theta_guess * delta_t)])
            ])
            B_t = torch.stack([
                (mu_guess - 0.5 * sigma_guess**2) * delta_t, torch.tensor(0.0, dtype=torch.float32)
            ]).reshape(2,1)
            Q_t = torch.stack([
                torch.stack([
                    sigma_guess**2 * delta_t,
                    (sigma_guess * alpha_guess * rho_guess / theta_guess) * (1 - torch.exp(-theta_guess * delta_t))
                ]),
                torch.stack([
                    (sigma_guess * alpha_guess * rho_guess / theta_guess) * (1 - torch.exp(-theta_guess * delta_t)),
                    (alpha_guess**2 / (2 * theta_guess)) * (1 - torch.exp(-2 * theta_guess * delta_t))
                ])
            ])
            # extrapolate state (point estimate for next time step)
            state = F_t @ state + B_t
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('prior_estimated_log_fair_value')] = state[0, 0].item()
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('prior_estimated_ou_exponent')] = state[1, 0].item()
            # extrapolate uncertainty (covariance matrix for next time step)
            state_covariance_matrix = F_t @ state_covariance_matrix @ F_t.T + Q_t
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('prior_variance_log_fair_value')] = state_covariance_matrix[0, 0].item()
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('prior_variance_ou_exponent')] = state_covariance_matrix[1, 1].item()
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('prior_covariance_log_fair_value_ou_exponent')] = state_covariance_matrix[0, 1].item()
            # update with observation
            observation = torch.tensor([[train_and_test_data['log_SP500_Adj_Close'].iloc[time_idx]]], dtype=torch.float32)
            surprise = observation - observation_matrix @ state # shape (1,1)
            # compute Kalman gain
            prior_covariance_observation = observation_matrix @ state_covariance_matrix @ observation_matrix.T + observation_noise_variance
            kalman_gain_numerator = state_covariance_matrix @ observation_matrix.T
            kalman_gain_denominator = prior_covariance_observation
            kalman_gain = kalman_gain_numerator @ torch.linalg.inv(kalman_gain_denominator)
            # derive posterior state
            posterior_state = state + kalman_gain @ surprise
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('posterior_estimated_log_fair_value')] = posterior_state[0, 0].item()
            train_and_test_data.iloc[time_idx, train_and_test_data.columns.get_loc('posterior_estimated_ou_exponent')] = posterior_state[1, 0].item()

            # update state covariance
            state_covariance_matrix = (torch.eye(2) - kalman_gain @ observation_matrix) @ state_covariance_matrix @ (torch.eye(2) - kalman_gain @ observation_matrix).T + kalman_gain @ observation_noise_variance @ kalman_gain.T

            # set current state to posterior
            state = posterior_state
    # save the train and test data with the inferred latent states
    train_and_test_data.to_csv(out_file_path_estimates, index=False)

# split the kaggle_headline_data into train and test sets
ssm_with_em_algorithm(train_chunk_1, train_chunk_2, 'ssm1_gbm_ou_multiplier/ssm_params_chunk_1.txt', 'ssm1_gbm_ou_multiplier/ssm_estimates_chunk_1.csv')
ssm_with_em_algorithm(pd.concat([train_chunk_1, train_chunk_2], ignore_index=True).sort_values(by='Date', ascending=True), train_chunk_3, 'ssm1_gbm_ou_multiplier/ssm_params_chunk_1_2.txt', 'ssm1_gbm_ou_multiplier/ssm_estimates_chunk_1_2.csv')
ssm_with_em_algorithm(pd.concat([train_chunk_2, train_chunk_3], ignore_index=True).sort_values(by='Date', ascending=True), train_chunk_4, 'ssm1_gbm_ou_multiplier/ssm_params_chunk_2_3.txt', 'ssm1_gbm_ou_multiplier/ssm_estimates_chunk_2_3.csv')
ssm_with_em_algorithm(pd.concat([train_chunk_3, train_chunk_4], ignore_index=True).sort_values(by='Date', ascending=True), train_chunk_5, 'ssm1_gbm_ou_multiplier/ssm_params_chunk_3_4.txt', 'ssm1_gbm_ou_multiplier/ssm_estimates_chunk_3_4.csv')
