import pygame
import random
import sys
import json
import time
import copy
from tetris import Tetris, GRID_SIZE, CELL_SIZE, SCREEN_SIZE, FPS, COLOR_MAP, BLACK, WHITE
import os 

MAX_MOVES = 500

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

def play_game(screen, clock):
    game = Tetris()
    bot = TetrisBot(game)
    trajectory = []

    game.new_piece()

    for _ in range(MAX_MOVES):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        move = bot.make_move()
        game_state = get_game_state(game)
        trajectory.append({
            "move": move,
            "state": game_state
        })

        game.update()
        
        if game.game_over:
            break
        
        screen.fill(BLACK)
        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    return trajectory

def generate_trajectories(num_games):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Tetris Bot")
    clock = pygame.time.Clock()

    # Create trajectories folder if it doesn't exist
    os.makedirs("trajectories", exist_ok=True)

    for i in range(num_games):
        start_time = time.time()
        trajectory = play_game(screen, clock)
        end_time = time.time()
        
        if trajectory is None:  # Game was closed
            break
        
        duration = end_time - start_time
        game_data = {
            "trajectory": trajectory,
            "duration": duration
        }
        
        # Save individual trajectory
        with open(f"trajectories/trajectory-{i+1}.json", 'w') as f:
            json.dump(game_data, f)
        
        print(f"Game {i+1} completed. Trajectory length: {len(trajectory)}. Time taken: {duration:.2f} seconds")

    pygame.quit()
    return i + 1  # Return the number of games actually played

if __name__ == "__main__":
    num_games = 200
    start_time = time.time()
    
    games_played = generate_trajectories(num_games)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Generated and saved {games_played} trajectories.")
    print(f"Total time taken: {total_time:.2f} seconds")
    
    avg_time_per_trajectory = total_time / games_played if games_played else 0
    print(f"Average time per trajectory: {avg_time_per_trajectory:.2f} seconds")
    
    estimated_total_time = avg_time_per_trajectory * num_games
    print(f"Estimated total time for {num_games} trajectories: {estimated_total_time:.2f} seconds ({estimated_total_time/60:.2f} minutes)")
