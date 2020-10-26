import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def best_response_dynamics_plurality(n, m, preferences):
    # initialize game
    preferences = np.array(preferences) - 1  # make candidates start from 0 for simplicity
    game = np.zeros((n, m), dtype=int)
    game[range(n), preferences[:, 0]] = 1  # vote first preference of each voter
    start_winner = np.argmax(np.sum(game, axis=0))
    start_winner_votes = np.sum(game[:, start_winner])

    # start learning algorithm
    t, tmax, changed = 1, 1000, True
    while changed and t <= tmax:
        winner = np.argmax(np.sum(game, axis=0))
        changed = False
        for i in range(n):
            current_vote = np.where(game[i, :] == 1)[0][0]
            if current_vote != winner:
                alt_choices = preferences[i, :np.where(preferences[i, :] == winner)[0][0] + 1]
                alt_votes = np.sum(game[:, alt_choices], axis=0)
                # add 1 vote to check if a preferred winner can be elected
                for j in range(len(alt_choices)):
                    if alt_choices[j] != current_vote and alt_choices[j] != winner:
                        alt_votes[j] += 1
                new_vote = alt_choices[np.argmax(alt_votes)]  # most preferred candidate to vote for at this point
                if new_vote != current_vote:
                    game[i, current_vote] = 0
                    game[i, new_vote] = 1
                    changed = True
                    break
        t += 1
    end_winner = np.argmax(np.sum(game, axis=0))
    end_winner_votes = np.sum(game[:, end_winner])

    return start_winner + 1, start_winner_votes, end_winner + 1, end_winner_votes, t-1


def best_response_dynamics_borda(n, m, preferences):
    # initialize game
    preferences = np.array(preferences) - 1  # make candidates start from 0 for simplicity
    game = np.zeros((n, m), dtype=int)
    for j in range(m):
        game[range(n), preferences[:, j]] = m-1-j
    start_winner = np.argmax(np.sum(game, axis=0))
    start_winner_votes = np.sum(game[:, start_winner])

    # start learning algorithm
    t, tmax, changed = 1, 1000, True
    while changed and t <= tmax:
        winner = np.argmax(np.sum(game, axis=0))
        changed = False
        for i in range(n):
            current_vote = np.where(game[i, :] == m-1)[0][0]
            if current_vote != winner:
                alt_choices = preferences[i, :np.where(preferences[i, :] == winner)[0][0] + 1]
                # also subtract current votes
                alt_votes = np.sum(game[:, alt_choices], axis=0) - game[i, alt_choices]
                # add m-1 votes to check if a preferred winner can be elected
                for j in range(len(alt_choices)):
                    if alt_choices[j] != winner:
                        alt_votes[j] += m-1
                new_vote = alt_choices[np.argmax(alt_votes)]  # most preferred candidate to vote for at this point
                if new_vote != current_vote:
                    # temporary preference order where new_vote is first preference
                    temp_preferences = preferences[i, :]
                    temp_preferences = np.delete(temp_preferences, np.where(temp_preferences == new_vote)[0])
                    temp_preferences = np.insert(temp_preferences, 0, new_vote)
                    for j in range(m):
                        game[i, temp_preferences[j]] = m - 1 - j
                    changed = True
                    break
        t += 1
    end_winner = np.argmax(np.sum(game, axis=0))
    end_winner_votes = np.sum(game[:, end_winner])

    return start_winner + 1, start_winner_votes, end_winner + 1, end_winner_votes, t-1


if __name__ == "__main__":
    c1_plurality = []
    c2_plurality = []
    c1_borda = []
    c2_borda = []
    rounds_plurality = []
    rounds_borda = []

    for file in os.listdir('games'):
        with open('games/'+file) as json_file:
            data = json.load(json_file)
            voters = data['voters']
            candidates = data['candidates']
            electoral_rule = data['electoral_rule']
            voting_preferences = data['voting_preferences']

        if electoral_rule == 'Plurality':
            c1, c1_score, c2, c2_score, rounds = \
                best_response_dynamics_plurality(voters, candidates, voting_preferences)
            c1_plurality.append([c1, c1_score])
            c2_plurality.append([c2, c2_score])
            rounds_plurality.append(rounds)
        elif electoral_rule == 'Borda':
            c1, c1_score, c2, c2_score, rounds = \
                best_response_dynamics_borda(voters, candidates, voting_preferences)
            c1_borda.append([c1, c1_score])
            c2_borda.append([c2, c2_score])
            rounds_borda.append(rounds)

    sns.set(style="whitegrid")
    # convergence
    plt.figure()
    plt.plot(rounds_plurality, label='Plurality (MO: '+str(int(np.mean(rounds_plurality)))+')')
    plt.plot(rounds_borda, label='Borda (ΜΟ: '+str(int(np.mean(rounds_borda)))+')')
    plt.title('Σύγκλιση Best Response Dynamics')
    plt.legend()
    plt.xlabel('Παίγνειο')
    plt.ylabel('Γύροι')

    # election results for plurality
    c1_plurality = np.asarray(c1_plurality)
    c2_plurality = np.asarray(c2_plurality)
    mean_difference = np.mean(c2_plurality[:, 1]-c1_plurality[:, 1])
    different_winners = len(np.where(c1_plurality[:, 0] != c2_plurality[:, 0])[0])
    plt.figure()
    plt.plot(c2_plurality[:, 1], label='Σκορ νικητών στην ισορροπία')
    plt.plot(c1_plurality[:, 1], label='Σκορ νικητών με πραγματικές δηλώσεις')
    plt.title('Διαφορά Σκορ Νικητών στον Κανόνα Plurality\nΑριθμός διαφορετικών νικητών: '+str(different_winners) +
              '\nΛόγος μέσης διαφοράς σκορ προς μέσο σκορ νικητών στην ισορροπία: ' +
              str(np.around(mean_difference/np.mean(c2_plurality[:, 1]), 2)))
    plt.legend()
    plt.xlabel('Παίγνειο')
    plt.ylabel('Σκορ')

    # election results for borda
    c1_borda = np.asarray(c1_borda)
    c2_borda = np.asarray(c2_borda)
    mean_difference = np.mean(c2_borda[:, 1]-c1_borda[:, 1])
    different_winners = len(np.where(c1_borda[:, 0] != c2_borda[:, 0])[0])
    plt.figure()
    plt.plot(c2_borda[:, 1], label='Σκορ νικητών στην ισορροπία')
    plt.plot(c1_borda[:, 1], label='Σκορ νικητών με πραγματικές δηλώσεις')
    plt.title('Διαφορά Σκορ Νικητών στον Κανόνα Borda\nΑριθμός διαφορετικών νικητών: '+str(different_winners) +
              '\nΛόγος μέσης διαφοράς σκορ προς μέσο σκορ νικητών στην ισορροπία: ' +
              str(np.around(mean_difference/np.mean(c2_borda[:, 1]), 2)))
    plt.legend()
    plt.xlabel('Παίγνειο')
    plt.ylabel('Σκορ')

    plt.show()
