#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sydsaenz/18.065FinalProject/blob/main/18_065FinalProjectModel1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Imports

# In[21]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from collections import defaultdict


# # Data Processing

# In[22]:


# mappers of team names to ids and vice versa
url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MTeams.csv"
names_and_ids = pd.read_csv(url)
names_and_ids = names_and_ids[["TeamID", "TeamName"]]
ids_to_names = {}
for i,row in names_and_ids.iterrows():
  ids_to_names[row["TeamID"]] = row["TeamName"]
names_to_ids = {v:k for k,v in ids_to_names.items()}


# In[23]:


url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MNCAATourneySeeds.csv"
initial_seeds = pd.read_csv(url)
initial_seeds


# In[24]:


# url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MNCAATourneySeedRoundSlots.csv"
# matchings = pd.read_csv(url)
# matchings


# In[25]:


#matchings[(matchings["GameRound"]==1) & (matchings["GameSlot"]=="R1W1")]


# In[26]:


# team data
url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/team_features.csv"
team_data = pd.read_csv(url).drop(columns=["index"])
team_data


# In[27]:


# team data exploration
for col in team_data.columns:
  print(col)
  print(team_data[col].describe())
  print(f"n_unique={team_data[col].nunique()}")
  print("------------------------------------------")


# In[28]:


for team_id in team_data["TeamID"]:
  if team_id not in ids_to_names:
    print(f"unknown team id: {team_id}")
print(f"number of unique rows: {len(team_data[ ['TeamID', 'Season'] ].drop_duplicates())}")


# In[29]:


# matches data
url = "https://raw.githubusercontent.com/sydsaenz/18.065FinalProject/main/MNCAATourneyCompactResults.csv"
matches_raw = pd.read_csv(url)
matches_raw


# In[30]:


matches = matches_raw[["Season","WTeamID","LTeamID"]]


# In[31]:


# combined winner and loser data for all matches
matches_W = pd.merge(left=matches,right=team_data, left_on=["Season","WTeamID"], right_on=["Season","TeamID"], how="left").drop(columns=["TeamID"])
cols_to_rename = list(set(team_data.columns)-{"Season","TeamID"})
W_names = {n:n+"_W" for n in cols_to_rename}
matches_W.rename(columns=W_names, inplace=True)
matches_WL = pd.merge(left=matches_W,right=team_data, left_on=["Season","LTeamID"], right_on=["Season","TeamID"], how="left").drop(columns=["TeamID"])
L_names = {n:n+"_L" for n in cols_to_rename}
matches_WL.rename(columns=L_names, inplace=True)
matches_WL


# In[32]:


# add column for winner - loser difference
for col in cols_to_rename:
  matches_WL[col+"_diff"] = matches_WL[col+"_W"] - matches_WL[col+"_L"]
matches_WL


# In[33]:


# label these as 1 (team1 wins)
matches_diff = matches_WL[[col for col in matches_WL.columns if "diff" in col]]
matches_diff["team_1_win"] = 1
matches_diff


# In[34]:


# negate copy of this dataframe so it's loser-winner. label with 0 (team1 loses)
matches_diff_negated = matches_diff.copy()
# adding 10000 to the index so they are unique but also so i can find the corresponding pair easily
matches_diff_negated.index = matches_diff.index+ np.ones_like(matches_diff.index)*10000
matches_diff_negated = -matches_diff_negated
matches_diff_negated["team_1_win"] = 0
matches_diff_negated


# In[35]:


# concatenate W-L and L-W to get training dataset
model_dataset = pd.concat([matches_diff, matches_diff_negated])
model_dataset


# In[36]:


# evaluate logisitic regression as a predictor
X = model_dataset.drop('team_1_win', axis=1)
y = model_dataset['team_1_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred_train = logreg.predict(X_train)

y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


# In[37]:


# precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# In[38]:


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


# In[39]:


# confusion matrices
test_conf_matrix = confusion_matrix(y_test, y_pred)
train_conf_matrix = confusion_matrix(y_train, y_pred_train)


# In[40]:


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


# In[41]:


logreg_full = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
logreg_full.fit(X, y)


# 
# 
# ---
# 
# # MCMC Bracket Simulator
# 

# In[42]:


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


# In[43]:


class Game_simulator():
  def __init__(self, model,team_stats):
    self.model = model # logreg
    self.team_stats = team_stats # df, index is team id

  def get_prob_team1_win(self, team1,team2):
    data = self.team_stats.loc[team1.id,:] - self.team_stats.loc[team2.id,:]
    data = pd.DataFrame(data).T
    data.rename(columns={c:c+"_diff" for c in data.columns},inplace=True)
    data = data[X.columns]
    #print(data)
    prob = self.model.predict_proba(data)
    #print(prob.shape)
    return prob[0,1] # probability of class 1 (team 1 wins)

  def play(self, game):
    game.team1.prev_games.append(game) # only after play is called is game added to prev games. not when game is created
    game.team2.prev_games.append(game)
    prob_team_1_wins = self.get_prob_team1_win( game.team1, game.team2)
    if np.random.rand() < prob_team_1_wins:
      game.winner = game.team1
    else:
      game.winner = game.team2


# In[44]:


def get_initial_seeds_pairs(year):
  df = initial_seeds[initial_seeds["Season"] == year]
  pairs = []
  for i,row in df.iterrows():
    pairs.append((Team(team_id=row["TeamID"]),row["Seed"]))
  return pairs


# In[45]:


class Bracket_simulation():
  def __init__(self, team_data_all_years, year, model, ids_to_names):
    self.region_names = ["W", "X", "Y", "Z"]
    self.team_data = team_data_all_years[team_data_all_years["Season"] == year].set_index("TeamID").drop(columns=["Season"])
    self.initial_seeds= get_initial_seeds_pairs(year) # list of (team_id,seed) pairs
    self.game_simulator = Game_simulator(model, self.team_data)
    self.first_four_matches = []
    self.regional_matches = {}

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





# In[46]:


# example
my_bracket = Bracket_simulation(team_data, 2025, logreg_full, ids_to_names)
my_bracket.run_tournament(verbose=True)


# 

# In[ ]:


iters = 10000
winner_counts = defaultdict(int)

for _ in range(iters):
  bracket = Bracket_simulation(team_data, 2025, logreg_full, ids_to_names)
  bracket.run_tournament(verbose=False)
  winner = bracket.final.winner.name
  winner_counts[winner] += 1


# In[ ]:


# prompt: make a bar plot with winner_counts. sort the teams in decreasing number of wins. make it a percentage of wins

import matplotlib.pyplot as plt

# Assuming winner_counts is already populated from the previous code

# Sort teams by decreasing number of wins
sorted_winners = dict(sorted(winner_counts.items(), key=lambda item: item[1], reverse=True))

# Calculate percentages
total_iterations = sum(winner_counts.values())
percentages = {team: (count / total_iterations) * 100 for team, count in sorted_winners.items()}

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(percentages.keys(), percentages.values())
plt.xlabel("Teams")
plt.ylabel("Percentage of Wins (%)")
plt.title("Tournament Simulation Results (Percentage of Wins)")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

