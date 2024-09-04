from tetris import *


def main():
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Enhanced Tetris")
    clock = pygame.time.Clock()
    game = Tetris()

    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move(0, -1)
                if event.key == pygame.K_RIGHT:
                    game.move(0, 1)
                if event.key == pygame.K_DOWN:
                    game.move(1, 0)
                if event.key == pygame.K_UP:
                    game.rotate()

        game.update()
        screen.fill(BLACK)
        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    print(f"Game Over! Final Score: {game.score}")
    pygame.quit()

if __name__ == "__main__":
    main()
