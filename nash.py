import itertools
import time

def find_nash_equilibria(strategies, payoffs):
    nash_equilibria = []
    num_players = len(strategies)
    for profile in itertools.product(*strategies):
        is_nash = True
        current_payoffs = payoffs[profile]
        # Check each player's possible deviation
        for player in range(num_players):
            original_strategy = profile[player]
            current_payoff = current_payoffs[player]
            # Try each alternative strategy for this player
            for new_strategy in strategies[player]:
                if new_strategy == original_strategy:
                    continue
                # Create new profile with player's strategy changed
                new_profile = list(profile)
                new_profile[player] = new_strategy
                new_profile = tuple(new_profile)
                new_payoff = payoffs[new_profile][player]
                if new_payoff > current_payoff:
                    is_nash = False
                    break
            if not is_nash:
                break
        if is_nash:
            nash_equilibria.append(profile)
    return nash_equilibria

# Define strategies
# 1,814,400 possible profiles
strategies = [
    ['A', 'B', 'C'],                # Player 1: 3 strategies
    ['A', 'B', 'C', 'D'],           # Player 2: 4 strategies
    ['A', 'B', 'C', 'D', 'E'],      # Player 3: 5 strategies
    ['A', 'B', 'C', 'D', 'E', 'F'], # Player 4: 6 strategies
    ['A', 'B', 'C', 'D', 'E', 'F', 'G'],          # Player 5
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],     # Player 6
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],# Player 7
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],      # Player 8
    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'], # Player 9
    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'] # Player 10
]

# Generate payoffs
payoffs = {}
for s in itertools.product(*strategies):
    payoffs[s] = tuple(strategies[i].index(s[i]) + 1 for i in range(len(strategies)))

# Find Nash equilibria
start = time.time()
equilibria = find_nash_equilibria(strategies, payoffs)
end = time.time()
print(f'Time:', end-start, 's')
print("Nash Equilibria:", equilibria)