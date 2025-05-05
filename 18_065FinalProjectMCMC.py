#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sydsaenz/18.065FinalProject/blob/main/18_065FinalProjectModel1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from collections import Counter 
from sklearn.base import clone
from scipy.special import expit

# # Data Processing
# mappers of team names to ids and vice versa
url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MTeams.csv"
names_and_ids = pd.read_csv(url)
names_and_ids = names_and_ids[["TeamID", "TeamName"]]
ids_to_names = {}
for i,row in names_and_ids.iterrows():
  ids_to_names[row["TeamID"]] = row["TeamName"]
names_to_ids = {v:k for k,v in ids_to_names.items()}

url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MNCAATourneySeeds.csv"
initial_seeds = pd.read_csv(url)
# url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MNCAATourneySeedRoundSlots.csv"
# team data
url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/team_features.csv"
team_data = pd.read_csv(url).drop(columns=["index"])

# team data exploration
for col in team_data.columns:
  print(col)
  print(team_data[col].describe())
  print(f"n_unique={team_data[col].nunique()}")
  print("------------------------------------------")

for team_id in team_data["TeamID"]:
  if team_id not in ids_to_names:
    print(f"unknown team id: {team_id}")
print(f"number of unique rows: {len(team_data[ ['TeamID', 'Season'] ].drop_duplicates())}")

# matches data
url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MNCAATourneyCompactResults.csv"
matches_raw = pd.read_csv(url)
matches = matches_raw[["Season","WTeamID","LTeamID"]]
train_matches = matches[matches.Season < 2025]
test_matches  = matches[matches.Season == 2025] #oops, empty

def make_dataset(matches, team_data):
    # combined winner and loser data for all matches
    matches_W = pd.merge(left=matches,right=team_data, left_on=["Season","WTeamID"], right_on=["Season","TeamID"], how="left").drop(columns=["TeamID"])
    cols_to_rename = list(set(team_data.columns)-{"Season","TeamID"})
    W_names = {n:n+"_W" for n in cols_to_rename}
    matches_W.rename(columns=W_names, inplace=True)
    matches_WL = pd.merge(left=matches_W,right=team_data, left_on=["Season","LTeamID"], right_on=["Season","TeamID"], how="left").drop(columns=["TeamID"])
    L_names = {n:n+"_L" for n in cols_to_rename}
    matches_WL.rename(columns=L_names, inplace=True)

    # add column for winner - loser difference
    for col in cols_to_rename:
      matches_WL[col+"_diff"] = matches_WL[col+"_W"] - matches_WL[col+"_L"]

    # label these as 1 (team1 wins)
    matches_diff = matches_WL[[col for col in matches_WL.columns if "diff" in col]]
    matches_diff["team_1_win"] = 1

    # negate copy of this dataframe so it's loser-winner. label with 0 (team1 loses)
    matches_diff_negated = matches_diff.copy()
    # adding 10000 to the index so they are unique but also so i can find the corresponding pair easily
    matches_diff_negated.index = matches_diff.index+ np.ones_like(matches_diff.index)*10000
    matches_diff_negated = -matches_diff_negated
    matches_diff_negated["team_1_win"] = 0
    # concatenate W-L and L-W to get training dataset
    model_dataset = pd.concat([matches_diff, matches_diff_negated])
    return model_dataset 

train_model_dataset = make_dataset(train_matches, team_data)
test_model_dataset = make_dataset(test_matches, team_data) #this is empty for 2025, bc 2025 not in dataset
model_dataset = make_dataset(matches, team_data)

# evaluate logisitic regression as a predictor
X = model_dataset.drop('team_1_win', axis=1)
y = model_dataset['team_1_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Step 3: Use the updated model for predictions
predictions = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

#STATS FOR LOGISTIC REGRESSION MODEL
def display_logistic_stats(y_test, y_train, y_pred_prob, y_pred_train):
    # precision recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"{test_accuracy=}        {train_accuracy=}")

    test_precision = precision_score(y_test, y_pred)
    train_precision = precision_score(y_train, y_pred_train)
    print(f"{test_precision=}        {train_precision=}")

    test_recall = recall_score(y_test, y_pred)
    train_recall = recall_score(y_train, y_pred_train)
    print(f"{test_recall=}        {train_recall=}")

    test_f1 = f1_score(y_test, y_pred)
    train_f1 = f1_score(y_train, y_pred_train)
    print(f"{test_f1=}        {train_f1=}")

    # confusion matrices
    test_conf_matrix = confusion_matrix(y_test, y_pred)
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Test Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(train_conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title("Train Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.show()
    return 

# display_logistic_stats(y_test=y_test, y_train=y_train, y_pred_prob=y_pred_prob, y_pred_train=y_pred_train)

# #RUN LOGISTIC REGRESSION
# logreg_full = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
# logreg_full.fit(X, y)

# # MCMC Bracket Simulator # # -----------------------------------------------------------------------------------------------------
class Team(): # data structure for node
  def __init__(self, team_id=None, team_name=None):
    if (team_id is None) and (team_name is None):
      raise ValueError("must provide at least one of team name or team id")
    if team_id is None:
      team_id = names_to_ids[team_name]
    if team_name is None:
      team_name = ids_to_names[team_id]

    self.id = team_id
    self.name = team_name
    self.prev_games = []

class Game(): # node for tree
  def __init__(self, team1, team2):
        self.team1 = team1
        self.team2 = team2
        self.team1_prev_game = team1.prev_games[-1] if team1.prev_games else None
        self.team2_prev_game = team2.prev_games[-1] if team2.prev_games else None
        self.winner = None
        self.winner_next_game = None # don't use this yet idk how

class Game_simulator():
  def __init__(self, model, team_stats):
        self.model = model  # Should be a fitted LogisticRegression
        self.team_stats = team_stats

  def get_prob_team1_win(self, team1, team2, coeffs=None):
      # Compute feature difference
      data = self.team_stats.loc[team1.id, :] - self.team_stats.loc[team2.id, :]
      if isinstance(data, pd.Series):
          data = data.to_frame().T
      data.rename(columns={c: c+"_diff" for c in data.columns}, inplace=True)
      data = data[X.columns]  # Ensure column order matches training

      # Get predicted probabilities
      prob = self.model.predict_proba(data)
      prob = np.atleast_2d(prob)
      # Handle single-class output
      if prob.shape[1] == 1:
          # If only one class, assume it's P(team1 wins)
          return prob[0, 0]
      else:
          # Standard two-class case
          return prob[0, 1]

  def play(self, game):
    game.team1.prev_games.append(game) # only after play is called is game added to prev games. not when game is created
    game.team2.prev_games.append(game)
    prob_team_1_wins = self.get_prob_team1_win( game.team1, game.team2)
    if np.random.rand() < prob_team_1_wins:
      game.winner = game.team1
    else:
      game.winner = game.team2

def get_initial_seeds_pairs(year):
  df = initial_seeds[initial_seeds["Season"] == year]
  pairs = []
  for i,row in df.iterrows():
    pairs.append((Team(team_id=row["TeamID"]),row["Seed"]))
  return pairs

class Bracket_simulation():
  def __init__(self, team_data_all_years, year, model, ids_to_names):
    self.region_names = ["W", "X", "Y", "Z"]
    self.team_data = team_data_all_years[team_data_all_years["Season"] == year].set_index("TeamID").drop(columns=["Season"])
    self.initial_seeds= get_initial_seeds_pairs(year) # list of (team_id,seed) pairs
    # self.game_simulator = Game_simulator(model, self.team_data)
    self.game_simulator = Game_simulator(mcmc, self.team_data)
    self.first_four_matches = []
    self.regional_matches = {}
    self.coeffs = None

  def set_coeffs(self, coeffs):
    n_features = len(coeffs) - 1
    self.game_simulator.model.coef_ = coeffs[:n_features].reshape(1, -1)
    self.game_simulator.model.intercept_ = coeffs[-1:].reshape(1,)
    # Only set classes_ if needed
    if not hasattr(self.game_simulator.model, "classes_"):
        self.game_simulator.model.classes_ = np.array([0, 1])

  def run_first_four(self, verbose=True):
    first_round = []
    for i in range(len(self.initial_seeds)-1,-1,-1):
      t = self.initial_seeds[i]
      if ((t[1][-1]=="a") or (t[1][-1]=="b")): # first round
        first_round.append(self.initial_seeds.pop(i))
    first_round.sort(key=lambda x:x[1] )

    if verbose:
      print("Running first four:")

    while first_round:
      t1 = first_round.pop(0)
      t2 = first_round.pop(0)
      team1 = t1[0]
      team2 = t2[0]
      game = Game(team1,team2)
      self.game_simulator.play(game)
      self.first_four_matches.append(game)
      self.initial_seeds.append( ( game.winner, t1[1][:3] ) ) #add the winner to the initial seeds without the extra letter a or b

      if verbose:
        print(f"- {team1.name} played {team2.name} and {game.winner.name} won.")

    if verbose:
      print("----------------------------------------------------------------")

  def run_region(self, region_letter, verbose=True):
    seeds = self.region_initial_seeds[region_letter]
    # level 1
    if verbose:
      print(f"Running level 1 of region {region_letter}")

    level_1 = []
    game = Game(seeds[1],seeds[16])
    self.game_simulator.play(game)
    seeds[17] = game.winner # these seed numbers (17+) don't mean anything, it's so i can find who needs to play who
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[8],seeds[9])
    self.game_simulator.play(game)
    seeds[18] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[5],seeds[12])
    self.game_simulator.play(game)
    seeds[19] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[4],seeds[13])
    self.game_simulator.play(game)
    seeds[20] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[6],seeds[11])
    self.game_simulator.play(game)
    seeds[21] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[3],seeds[14])
    self.game_simulator.play(game)
    seeds[22] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[7],seeds[10])
    self.game_simulator.play(game)
    seeds[23] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[2],seeds[15])
    self.game_simulator.play(game)
    seeds[24] = game.winner
    level_1.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    # level 2
    if verbose:
      print(" ")
      print(f"Running level 2 of region {region_letter}")

    level_2 = []
    game = Game(seeds[17],seeds[18])
    self.game_simulator.play(game)
    seeds[25] = game.winner
    level_2.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[19],seeds[20])
    self.game_simulator.play(game)
    seeds[26] = game.winner
    level_2.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[21],seeds[22])
    self.game_simulator.play(game)
    seeds[27] = game.winner
    level_2.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[23],seeds[24])
    self.game_simulator.play(game)
    seeds[28] = game.winner
    level_2.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    # level 3
    if verbose:
      print(" ")
      print(f"Running level 3 (regional quarter final) of region {region_letter}")

    level_3 = []
    game = Game(seeds[25],seeds[26])
    self.game_simulator.play(game)
    seeds[29] = game.winner
    level_3.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(seeds[27],seeds[28])
    self.game_simulator.play(game)
    seeds[30] = game.winner
    level_3.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    # level 4 (regional final)
    if verbose:
      print(" ")
      print(f"Running level 4 (regional final) of region {region_letter}")

    level_4 = []
    game = Game(seeds[29],seeds[30])
    self.game_simulator.play(game)
    seeds[31] = game.winner
    level_4.append(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")
      print(f"{game.winner.name} is the regional winner!")
      print("------------------------------------------------------------------")

    self.regional_matches[region_letter] = {
        "level1": level_1,
        "level2": level_2,
        "level3": level_3,
        "level4": level_4,
        "winner": game.winner
    }

  def run_semi_and_final(self, verbose=True):
    if verbose:
      final4 = [self.regional_matches[r]["winner"].name for r in self.region_names ]
      print(f"The final 4 are {final4}.")
    # region W plays X
    # region Y plays Z
    # semi
    semis = {}
    if verbose:
      print("Running semi finals:")

    game = Game(self.regional_matches["W"]["winner"], self.regional_matches["X"]["winner"] )
    self.game_simulator.play(game)
    semis["WX"] = game
    wx_winner = game.winner
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    game = Game(self.regional_matches["Y"]["winner"], self.regional_matches["Z"]["winner"] )
    self.game_simulator.play(game)
    semis["YZ"] = game
    yz_winner = game.winner
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")

    # final
    if verbose:
      print(" ")
      print("Running final:")

    game = Game(wx_winner,yz_winner)
    self.game_simulator.play(game)
    if verbose:
      print(f"{game.team1.name} played {game.team2.name} and {game.winner.name} won.")
      print(f"The winner is {game.winner.name}!")
      print("==================================================================")

    self.semis = semis
    self.final = game

  def run_tournament(self, verbose=True):
    self.run_first_four(verbose=verbose)

    #self.region_initial_seeds = {"W": {}, "X": {}, "Y": {}, "Z": {} }
    self.region_initial_seeds = {r: {} for r in self.region_names }
    for t in self.initial_seeds:
      self.region_initial_seeds[t[1][0]] [int(t[1][1:])] = t[0]

    for region_letter in  self.region_names:
      self.run_region(region_letter,verbose=verbose)

    self.run_semi_and_final(verbose=verbose)

class MCMCLogisticWrapper:
    def __init__(self, base_model, num_samples=2000, burn_in=1000, prior_var=100):
        print('MCMC initialized')
        self.base_model = base_model
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.prior_var = prior_var
        self.chain_ = None
        self.acceptance_rate_ = None

    def fit(self, X, y, adapt_proposal=True, thinning=1, verbose=True):
        current = np.concatenate([self.base_model.coef_.flatten(), self.base_model.intercept_])
        N = (self.num_samples * thinning) + self.burn_in
        self.chain_ = np.zeros((self.num_samples, len(current)))

        proposal_stds = np.ones_like(current) * 0.01
        accept_counts = np.zeros_like(current)
        window = 50
        window_accepts = np.zeros_like(current)

        for i in range(N):
            if verbose and not (i % 100):
                print(f"iteration {i}/{N}")

            for j in range(len(current)):
                prop = current.copy()
                prop[j] += np.random.normal(0, proposal_stds[j])
                lp_cur = self._log_joint(current, X, y)
                lp_prp = self._log_joint(prop, X, y)
                accepted = np.log(np.random.rand()) < (lp_prp - lp_cur)
                if accepted:
                    current[j] = prop[j]
                    if i >= self.burn_in:
                        accept_counts[j] += 1
                    window_accepts[j] += 1

            # Adapt proposal stds during burn-in
            if adapt_proposal and i < self.burn_in and (i+1) % window == 0:
                rates = window_accepts / window
                for j in range(len(current)):
                    if rates[j] < 0.1:
                        proposal_stds[j] *= 0.7
                    elif rates[j] > 0.5:
                        proposal_stds[j] *= 1.3
                if verbose:
                    print(f"  window ending at iter {i}: per-parameter accept rates={rates}, stds={proposal_stds}")
                window_accepts[:] = 0

            # Store after burn-in, with thinning
            if i >= self.burn_in and ((i - self.burn_in) % thinning == 0):
                idx = (i - self.burn_in) // thinning
                if idx < self.num_samples:
                    self.chain_[idx] = current

        self.acceptance_rate_ = accept_counts / self.num_samples
        if verbose:
            print(f"Final per-parameter acceptance rates: {self.acceptance_rate_}")
            print(f"Final proposal stds: {proposal_stds}")
        return self

    def _log_joint(self, params, X, y):
        # Logistic regression log-likelihood (numerically stable)
        logit = X @ params[:-1] + params[-1]
        log_likelihood = np.sum(y * logit - np.logaddexp(0, logit))
        # Gaussian prior on all parameters (mean 0, variance prior_var)
        log_prior = -0.5 * np.sum(params ** 2) / self.prior_var
        return log_likelihood + log_prior

    def predict_proba(self, X):
        """Posterior predictive mean probability for each sample in X."""
        if self.chain_ is None:
            raise ValueError("Must call fit() first")
        logits = X @ self.chain_[:, :-1].T + self.chain_[:, -1]
        # Average over all posterior samples
        return expit(logits).mean(axis=1)

    def get_posterior_samples(self):
        """Return the full posterior chain for custom diagnostics or plotting."""
        if self.chain_ is None:
            raise ValueError("Must call fit() first")
        return self.chain_

def print_and_plot_champion_and_final4(champion_counts, final_four_counts, ids_to_names):
    import matplotlib.pyplot as plt

    # Totals
    total_champs = sum(champion_counts.values())
    total_f4     = sum(final_four_counts.values())

    # Text output
    print("Champion frequencies:")
    for tid, cnt in champion_counts.most_common():
        print(f"  {ids_to_names[tid]}: {cnt} ({cnt/total_champs:.2%})")

    print("\nFinal Four frequencies:")
    for tid, cnt in final_four_counts.most_common():
        print(f"  {ids_to_names[tid]}: {cnt} ({cnt/total_f4:.2%})")

    # Prepare for horizontal bar plots
    champs, champ_vals = zip(*champion_counts.most_common())
    champs_labels = [ids_to_names[tid] for tid in champs]

    f4s, f4_vals = zip(*final_four_counts.most_common())
    f4_labels = [ids_to_names[tid] for tid in f4s]
    f4_perc   = [cnt/total_f4*100 for cnt in f4_vals]

    fig, axes = plt.subplots(1,2, figsize=(10,6), sharey=False)

    # horizontal bar, orange
    axes[0].barh(champs_labels, np.array(champ_vals) / 25000, color='orange')
    axes[0].set_xlabel("Times Champion")
    axes[0].invert_yaxis()   # highest at top
    axes[0].set_title("Champion Frequency")

    axes[1].barh(f4_labels, f4_perc, color='orange')
    axes[1].set_xlabel("% of Final Four")
    axes[1].invert_yaxis()
    axes[1].set_title("Final Four Frequency")

    plt.tight_layout()
    plt.show()

def plot_coef_traces(chains, param_names):
    import matplotlib.pyplot as plt
    n_params = chains[0].shape[1]
    fig, axes = plt.subplots(n_params, 2, figsize=(10, 2*n_params))
    for p in range(n_params):
        # trace
        for c,chain in enumerate(chains):
            axes[p,0].plot(chain[:,p], label=f'chain{c+1}', alpha=0.6)
        axes[p,0].set_ylabel(param_names[p])
        # autocorr
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(np.concatenate([c[:,p] for c in chains]), ax=axes[p,1], lags=50)
        axes[p,1].set_title(f"ACF of {param_names[p]}")
    axes[0,0].legend()
    plt.tight_layout()
    plt.show()

X_train = train_model_dataset.drop("team_1_win", axis=1)
y_train = train_model_dataset["team_1_win"]

# Fit MCMC‑wrapped logistic on pre‑2025
logreg = LogisticRegression(max_iter=2000).fit(X_train, y_train)
mcmc = MCMCLogisticWrapper(logreg, num_samples=2000, burn_in=1000)
mcmc.fit(X_train.values, y_train.values)

# Posterior predictive probabilities
probs = mcmc.predict_proba(X_test)
# Access posterior samples for diagnostics
samples = mcmc.get_posterior_samples()

# For each draw, simulate the 2025 bracket
champ_counts, final4_counts = Counter(), Counter()
for coeffs in samples:
    m = clone(logreg)
    m.coef_, m.intercept_, m.classes_ = coeffs[:-1].reshape(1,-1), coeffs[-1:], np.array([0,1])
    sim = Bracket_simulation(team_data, 2025, m, ids_to_names)
    sim.set_coeffs(coeffs)
    sim.run_tournament(verbose=False)
    champ_counts[sim.final.winner.id] += 1
    for r in sim.region_names:
        final4_counts[ sim.regional_matches[r]["winner"].id ] += 1

# Plot
print_and_plot_champion_and_final4(champ_counts, final4_counts, ids_to_names)
feature_names = list(X_train.columns)
param_names = feature_names + ["intercept"]
plot_coef_traces([mcmc.chain_], param_names=param_names)

# Evaluate against real 2025 results:
# Real champion: Florida 
fid = names_to_ids["Florida"]
p_florida = champ_counts[fid] / sum(champ_counts.values())
print(f"Posterior P(Florida champion) = {p_florida:.2%}")

# Real Final Four: Florida, Duke, Houston, Auburn
for name in ["Florida","Duke","Houston","Auburn"]:
    tid = names_to_ids[name]
    print(f"{name}: predicted {final4_counts[tid]/sum(final4_counts.values()):.2%} Final‑Four freq")

#the end 