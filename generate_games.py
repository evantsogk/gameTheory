import json
import random

path = 'games/'
n_m_values = [5, 10, 20, 50, 100]
rules = ['Plurality', 'Borda']
num_random_games = 30
file_count = 1


for n in n_m_values:
    data = {'voters': n}
    for m in n_m_values:
        data['candidates'] = m
        candidates = list(range(1, m+1))
        for i in range(num_random_games):
            data['voting_preferences'] = []
            for j in range(n):
                shuffled_candidates = candidates[:]
                random.shuffle(candidates)
                data['voting_preferences'].append(shuffled_candidates)
            for rule in rules:
                data["electoral_rule"] = rule
                with open(path+"game"+str(file_count)+'.json', 'w') as outfile:
                    json.dump(data, outfile)
                file_count += 1
