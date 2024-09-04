from tetris import *

def main():
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Minimal Tetris")
    clock = pygame.time.Clock()

    game = Tetris()
    game.new_piece()  # Ensure we start with a piece

    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move(0, -1)
                    game.trajectory.append("LEFT")
                elif event.key == pygame.K_RIGHT:
                    game.move(0, 1)
                    game.trajectory.append("RIGHT")
                elif event.key == pygame.K_UP:
                    game.move(-1, 0)
                    game.trajectory.append("UP")
                elif event.key == pygame.K_DOWN:
                    game.move(1, 0)
                    game.trajectory.append("DOWN")
                elif event.key == pygame.K_SPACE:
                    game.rotate()
                    game.trajectory.append("ROTATE")

        game.update()
        screen.fill(BLACK)
        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    print("Game Over!")
    print(f"Final Score: {game.score}")
    
    # Save trajectory
    with open("tetris_trajectory.json", "w") as f:
        json.dump(game.trajectory, f)
    print("Trajectory saved to tetris_trajectory.json")

    pygame.quit()

if __name__ == "__main__":
    main()
