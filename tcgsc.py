import itertools

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
strategies = [[0.1, 0.2], [0.3, 0.4, 0.5]]

# Generate payoffs
payoffs = {
    (0.1, 0.3): (1, 5),
    (0.1, 0.4): (4, 2),
    (0.1, 0.5): (3, 1),
    (0.2, 0.3): (2, 3),
    (0.2, 0.4): (1, 4),
    (0.2, 0.5): (5, 2)
}

# Find Nash equilibria
equilibria = find_nash_equilibria(strategies, payoffs)
print(equilibria)