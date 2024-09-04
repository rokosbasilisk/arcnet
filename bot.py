import torch
import torch.multiprocessing as mp
import random
import json
import time
import copy
from tetris import Tetris, GRID_SIZE
import os

MAX_MOVES = 1000

class TetrisBot:
    def __init__(self, game):
        self.game = game

    def make_move(self):
        moves = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'ROTATE']
        move = random.choice(moves)
        
        if move == 'LEFT':
            self.game.move(0, -1)
        elif move == 'RIGHT':
            self.game.move(0, 1)
        elif move == 'UP':
            self.game.move(-1, 0)
        elif move == 'DOWN':
            self.game.move(1, 0)
        elif move == 'ROTATE':
            self.game.rotate()
        
        return move

def get_game_state(game):
    state = copy.deepcopy(game.grid)
    if game.current_piece and game.current_position:
        for i, row in enumerate(game.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    x = game.current_position[0] + i
                    y = game.current_position[1] + j
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        color_index = (i * len(game.current_piece[0]) + j) % len(game.current_colors)
                        state[x][y] = game.current_colors[color_index]
    return state

def play_game():
    game = Tetris()
    bot = TetrisBot(game)
    trajectory = []
    game.new_piece()
    for _ in range(MAX_MOVES):
        move = bot.make_move()
        game_state = get_game_state(game)
        trajectory.append({
            "move": move,
            "state": game_state
        })
        game.update()
        
        if game.game_over:
            break

    return trajectory

def worker(num_games, return_dict, worker_id):
    trajectories = []
    start_time = time.time()
    for _ in range(num_games):
        trajectory = play_game()
        trajectories.append(trajectory)
    end_time = time.time()
    return_dict[worker_id] = {
        'trajectories': trajectories,
        'time': end_time - start_time
    }

def generate_trajectories_parallel(num_games, num_processes):
    os.makedirs("trajectories", exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    games_per_process = num_games // num_processes
    processes = []

    for i in range(num_processes):
        p = mp.Process(target=worker, args=(games_per_process, return_dict, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_trajectories = []
    total_time = 0

    for i in range(num_processes):
        result = return_dict[i]
        total_trajectories.extend(result['trajectories'])
        total_time += result['time']

    for i, trajectory in enumerate(total_trajectories):
        game_data = {
            "trajectory": trajectory,
            "duration": total_time / len(total_trajectories)  # Average duration per game
        }
        with open(f"trajectories/trajectory-{i+1}.json", 'w') as f:
            json.dump(game_data, f)

    return len(total_trajectories), total_time

if __name__ == "__main__":
    num_games = 1000
    num_processes = os.cpu_count()  # Use the number of CPU cores
    start_time = time.time()

    print(f"Starting trajectory generation using {num_processes} processes")
    games_played, total_time = generate_trajectories_parallel(num_games, num_processes)

    print(f"Generated and saved {games_played} trajectories.")
    print(f"Total time taken: {total_time:.2f} seconds")

    avg_time_per_trajectory = total_time / games_played if games_played else 0
    print(f"Average time per trajectory: {avg_time_per_trajectory:.2f} seconds")

    estimated_total_time = avg_time_per_trajectory * num_games
    print(f"Estimated total time for {num_games} trajectories: {estimated_total_time:.2f} seconds ({estimated_total_time/60:.2f} minutes)")
