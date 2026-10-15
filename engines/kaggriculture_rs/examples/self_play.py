"""Run independent self-play games with explicit seeds and simple policies."""
import argparse
import json
from kaggriculture_rs import BatchEnv

def policy(obs):
    me = obs['farms'][obs['player']]
    private = obs['private']
    x, y = me['farmer']
    tile = me['tiles'][y][x]
    market = []
    if private['shed'].get('CARROT', 0):
        market.append(['SELL', 'CARROT', private['shed']['CARROT']])
    if private['seeds'].get('CARROT', 0) == 0 and me['money'] >= 20:
        market.append(['BUY_SEED', 'CARROT', 1])
    unit = ['PASS']
    if tile is None and private['seeds'].get('CARROT', 0):
        unit = ['PLANT', 'CARROT']
    elif isinstance(tile, dict):
        if tile.get('kind') == 'WEED':
            unit = ['DIG']
        elif tile.get('kind') == 'PLANT':
            if obs['day'] - tile['planted_day'] >= 3:
                unit = ['HARVEST']
            elif not tile['watered_today']:
                unit = ['WATER']
    return {'farmer': unit, 'hands': [], 'market': market}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.games < 1 or args.batch_size < 1:
        parser.error('games and batch-size must be positive')
    width = min(args.games, args.batch_size)
    env = BatchEnv(width, num_threads=args.threads)
    seeds = [args.seed + i for i in range(width)]
    observations = env.reset(seeds)
    active = [True] * width
    next_seed = args.seed + width
    completed = []
    while any(active):
        actions = [[policy(o) for o in pair] for pair in observations]
        results = env.step(actions)
        observations = [r['observations'] for r in results]
        reset_indices, reset_seeds = [], []
        for i, result in enumerate(results):
            if active[i] and result['done']:
                scores = result['rewards']
                completed.append({'seed': seeds[i], 'scores': scores, 'winner': None if scores[0] == scores[1] else int(scores[1] > scores[0])})
                if next_seed < args.seed + args.games:
                    reset_indices.append(i)
                    reset_seeds.append(next_seed)
                    seeds[i] = next_seed
                    next_seed += 1
                else:
                    active[i] = False
        for i, pair in zip(reset_indices, env.reset_at(reset_indices, reset_seeds)):
            observations[i] = pair
    print(json.dumps(sorted(completed, key=lambda r: r['seed']), indent=2))

if __name__ == '__main__':
    main()
