from __future__ import annotations

import pygame

from .direction import Direction
from .game import Game

BOARD_SIZE = 20
CELL_SIZE = 28
SCORE_BAR_HEIGHT = 52
TICK_INTERVAL_MS = 120
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
SCORE_BAR_COLOR = (30, 30, 30)
DIGIT_SIZE = 4
GLYPHS = {
    "0": ("111", "101", "101", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "010", "010", "111"),
    "2": ("111", "001", "001", "111", "100", "100", "111"),
    "3": ("111", "001", "001", "111", "001", "001", "111"),
    "4": ("101", "101", "101", "111", "001", "001", "001"),
    "5": ("111", "100", "100", "111", "001", "001", "111"),
    "6": ("111", "100", "100", "111", "101", "101", "111"),
    "7": ("111", "001", "001", "010", "010", "010", "010"),
    "8": ("111", "101", "101", "111", "101", "101", "111"),
    "9": ("111", "101", "101", "111", "001", "001", "111"),
    "A": ("010", "101", "101", "111", "101", "101", "101"),
    "E": ("111", "100", "100", "110", "100", "100", "111"),
    "G": ("111", "100", "100", "101", "101", "101", "111"),
    "M": ("101", "111", "111", "101", "101", "101", "101"),
    "O": ("111", "101", "101", "101", "101", "101", "111"),
    "P": ("110", "101", "101", "110", "100", "100", "100"),
    "R": ("110", "101", "101", "110", "101", "101", "101"),
    "S": ("111", "100", "100", "111", "001", "001", "111"),
    "V": ("101", "101", "101", "101", "101", "010", "010"),
}


def main() -> None:
    pygame.init()
    window_size = BOARD_SIZE * CELL_SIZE
    screen = pygame.display.set_mode((window_size, window_size + SCORE_BAR_HEIGHT))
    pygame.display.set_caption("Snake DQN")
    clock = pygame.time.Clock()
    game = Game(BOARD_SIZE)
    last_tick = pygame.time.get_ticks()
    game_over = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    game = Game(BOARD_SIZE)
                    game_over = False
                    last_tick = pygame.time.get_ticks()
                    pygame.display.set_caption("Snake DQN")
                else:
                    change_direction(game, event.key)

        now = pygame.time.get_ticks()
        if not game_over and now - last_tick >= TICK_INTERVAL_MS:
            game_over = not game.tick()
            last_tick = now
            if game_over:
                pygame.display.set_caption("Snake DQN — Game over, press R")

        draw(screen, game, game_over)
        clock.tick(60)

    pygame.quit()


def change_direction(game: Game, key: int) -> None:
    directions = {
        pygame.K_UP: Direction.UP,
        pygame.K_w: Direction.UP,
        pygame.K_DOWN: Direction.DOWN,
        pygame.K_s: Direction.DOWN,
        pygame.K_LEFT: Direction.LEFT,
        pygame.K_a: Direction.LEFT,
        pygame.K_RIGHT: Direction.RIGHT,
        pygame.K_d: Direction.RIGHT,
    }
    if direction := directions.get(key):
        game.change_direction(direction)


def draw(screen: pygame.Surface, game: Game, game_over: bool) -> None:
    screen.fill(BLACK)
    pygame.draw.rect(
        screen, SCORE_BAR_COLOR, (0, 0, screen.get_width(), SCORE_BAR_HEIGHT)
    )
    draw_score(screen, game.score)

    for segment in game.snake.segments:
        draw_cell(screen, segment.x, segment.y, WHITE)

    draw_cell(screen, game.food.x, game.food.y, RED)

    if game_over:
        draw_game_over(screen)

    pygame.display.flip()


def draw_score(screen: pygame.Surface, score: int) -> None:
    draw_text(
        screen,
        str(score),
        screen.get_width() // 2,
        SCORE_BAR_HEIGHT // 2,
        DIGIT_SIZE,
        WHITE,
    )


def draw_game_over(screen: pygame.Surface) -> None:
    overlay = pygame.Surface(
        (screen.get_width(), screen.get_height() - SCORE_BAR_HEIGHT), pygame.SRCALPHA
    )
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, SCORE_BAR_HEIGHT))
    board_center_y = SCORE_BAR_HEIGHT + (screen.get_height() - SCORE_BAR_HEIGHT) // 2
    draw_text(screen, "GAME OVER", screen.get_width() // 2, board_center_y - 32, 6, RED)
    draw_text(screen, "PRESS R", screen.get_width() // 2, board_center_y + 32, 4, WHITE)


def draw_text(
    screen: pygame.Surface,
    text: str,
    center_x: int,
    center_y: int,
    pixel_size: int,
    color: tuple[int, int, int],
) -> None:
    gap = pixel_size
    text_width = sum(character_width(character, pixel_size) for character in text)
    text_width += gap * (len(text) - 1)
    start_x = center_x - text_width // 2
    start_y = center_y - 7 * pixel_size // 2

    for character in text:
        if character != " ":
            for row, pattern in enumerate(GLYPHS[character]):
                for column, pixel in enumerate(pattern):
                    if pixel == "1":
                        pygame.draw.rect(
                            screen,
                            color,
                            (
                                start_x + column * pixel_size,
                                start_y + row * pixel_size,
                                pixel_size,
                                pixel_size,
                            ),
                        )
        start_x += character_width(character, pixel_size) + gap


def character_width(character: str, pixel_size: int) -> int:
    if character == " ":
        return 2 * pixel_size
    return 3 * pixel_size


def draw_cell(
    screen: pygame.Surface, x: int, y: int, color: tuple[int, int, int]
) -> None:
    pygame.draw.rect(
        screen,
        color,
        (x * CELL_SIZE, SCORE_BAR_HEIGHT + y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
    )
