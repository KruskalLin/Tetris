import sys
import time
import pygame
from pygame.locals import *
import blocks

SIZE = 30  # UNIT RECTANGLE SIZE
BLOCK_HEIGHT = 25
BLOCK_WIDTH = 10
BORDER_WIDTH = 4
BORDER_COLOR = (40, 40, 200)
SCREEN_WIDTH = SIZE * (BLOCK_WIDTH + 5)
SCREEN_HEIGHT = SIZE * BLOCK_HEIGHT
BG_COLOR = (40, 40, 60)  # BACKGROUND COLOR
BLOCK_COLOR = (20, 128, 200)
BLACK = (0, 0, 0)
RED = (200, 30, 30)  # GAME OVER COLOR


def print_text(screen, font, x, y, text, fcolor=(255, 255, 255)):
    imgText = font.render(text, True, fcolor)
    screen.blit(imgText, (x, y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Tetris')

    font1 = pygame.font.SysFont('SimHei', 24)  # FONT TYPE
    font2 = pygame.font.Font(None, 72)  # GAME OVER FONT
    font_pos_x = BLOCK_WIDTH * SIZE + BORDER_WIDTH + 10  # FONT POSITION ON THE RIGHT
    gameover_size = font2.size('GAME OVER')
    font1_height = int(font1.size('POINTS')[1])

    cur_block = None
    next_block = None
    cur_pos_x, cur_pos_y = 0, 0

    game_area = None
    game_over = True
    start = False  # IF start = True, game_over = True, THEN WE HAVE GAME OVER
    score = 0  # SCORE
    orispeed = 1.0  # ORIGINAL SPEED
    speed = orispeed  # CURRENT SPEED
    pause = False
    last_drop_time = None
    last_press_time = None

    def _dock():
        nonlocal cur_block, next_block, game_area, cur_pos_x, cur_pos_y, game_over, score, speed
        for _i in range(cur_block.start_pos.Y, cur_block.end_pos.Y + 1):
            for _j in range(cur_block.start_pos.X, cur_block.end_pos.X + 1):
                if cur_block.template[_i][_j] != '.':
                    game_area[cur_pos_y + _i][cur_pos_x + _j] = '0'
        if cur_pos_y + cur_block.start_pos.Y <= 0:
            game_over = True
        else:
            # ELIMINATION
            remove_idxs = []
            for _i in range(cur_block.start_pos.Y, cur_block.end_pos.Y + 1):
                if all(_x == '0' for _x in game_area[cur_pos_y + _i]):
                    remove_idxs.append(cur_pos_y + _i)
            if remove_idxs:
                # CALCULATE THE SCORE
                remove_count = len(remove_idxs)
                if remove_count == 1:
                    score += 100
                elif remove_count == 2:
                    score += 300
                elif remove_count == 3:
                    score += 700
                elif remove_count == 4:
                    score += 1500
                speed = orispeed - 0.01 * min(score // 100, 99)
                # ELIMINATION
                _i = _j = remove_idxs[-1]
                while _i >= 0:
                    while _j in remove_idxs:
                        _j -= 1
                    if _j < 0:
                        game_area[_i] = ['.'] * BLOCK_WIDTH
                    else:
                        game_area[_i] = game_area[_j]
                    _i -= 1
                    _j -= 1
            cur_block = next_block
            next_block = blocks.get_block()
            cur_pos_x, cur_pos_y = (BLOCK_WIDTH - cur_block.end_pos.X - 1) // 2, -1 - cur_block.end_pos.Y

    def _judge(pos_x, pos_y, block):
        nonlocal game_area
        for _i in range(block.start_pos.Y, block.end_pos.Y + 1):
            if pos_y + block.end_pos.Y >= BLOCK_HEIGHT:
                return False
            for _j in range(block.start_pos.X, block.end_pos.X + 1):
                if pos_y + _i >= 0 and block.template[_i][_j] != '.' and game_area[pos_y + _i][pos_x + _j] != '.':
                    return False
        return True

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_RETURN:
                    if game_over:
                        start = True
                        game_over = False
                        score = 0
                        speed = orispeed - 0.01 * min(score // 100, 99)
                        last_drop_time = time.time()
                        last_press_time = time.time()
                        game_area = [['.'] * BLOCK_WIDTH for _ in range(BLOCK_HEIGHT)]
                        cur_block = blocks.get_block()
                        next_block = blocks.get_block()
                        cur_pos_x, cur_pos_y = (BLOCK_WIDTH - cur_block.end_pos.X - 1) // 2, -1 - cur_block.end_pos.Y
                elif event.key == K_SPACE:
                    if not game_over:
                        pause = not pause
                elif event.key == K_UP:
                    if 0 <= cur_pos_x <= BLOCK_WIDTH - len(cur_block.template[0]):
                        _next_block = blocks.get_next_block(cur_block)
                        if _judge(cur_pos_x, cur_pos_y, _next_block):
                            cur_block = _next_block

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if not game_over and not pause:
                    if time.time() - last_press_time > 0.1:
                        last_press_time = time.time()
                        if cur_pos_x > - cur_block.start_pos.X:
                            if _judge(cur_pos_x - 1, cur_pos_y, cur_block):
                                cur_pos_x -= 1
            if event.key == pygame.K_RIGHT:
                if not game_over and not pause:
                    if time.time() - last_press_time > 0.1:
                        last_press_time = time.time()
                        # DON'T REMOVE THE RIGHT INFO BORDER
                        if cur_pos_x + cur_block.end_pos.X + 1 < BLOCK_WIDTH:
                            if _judge(cur_pos_x + 1, cur_pos_y, cur_block):
                                cur_pos_x += 1
            if event.key == pygame.K_DOWN:
                if not game_over and not pause:
                    if time.time() - last_press_time > 0.1:
                        last_press_time = time.time()
                        if not _judge(cur_pos_x, cur_pos_y + 1, cur_block):
                            _dock()
                        else:
                            last_drop_time = time.time()
                            cur_pos_y += 1

        _draw_background(screen)

        _draw_game_area(screen, game_area)

        _draw_gridlines(screen)

        _draw_info(screen, font1, font_pos_x, font1_height, score)
        # DRAW NEXT BLOCK
        _draw_block(screen, next_block, font_pos_x, 30 + (font1_height + 6) * 5, 0, 0)

        if not game_over:
            cur_drop_time = time.time()
            if cur_drop_time - last_drop_time > speed:
                if not pause:
                    # WE CAN MOVE THE BLOCK WITH A SHORT PERIOD WHEN DOCKING
                    if not _judge(cur_pos_x, cur_pos_y + 1, cur_block):
                        _dock()
                    else:
                        last_drop_time = cur_drop_time
                        cur_pos_y += 1
        else:
            if start:
                print_text(screen, font2,
                           (SCREEN_WIDTH - gameover_size[0]) // 2, (SCREEN_HEIGHT - gameover_size[1]) // 2,
                           'GAME OVER', RED)

        # DRAW CURRENT BLOCK
        _draw_block(screen, cur_block, 0, 0, cur_pos_x, cur_pos_y)

        pygame.display.flip()


def _draw_background(screen):
    screen.fill(BG_COLOR)
    pygame.draw.line(screen, BORDER_COLOR,
                     (SIZE * BLOCK_WIDTH + BORDER_WIDTH // 2, 0),
                     (SIZE * BLOCK_WIDTH + BORDER_WIDTH // 2, SCREEN_HEIGHT), BORDER_WIDTH)


def _draw_gridlines(screen):
    # HORIZON LINE
    for x in range(BLOCK_WIDTH):
        pygame.draw.line(screen, BLACK, (x * SIZE, 0), (x * SIZE, SCREEN_HEIGHT), 1)
    # VERTICAL LINE
    for y in range(BLOCK_HEIGHT):
        pygame.draw.line(screen, BLACK, (0, y * SIZE), (BLOCK_WIDTH * SIZE, y * SIZE), 1)


# DRAW DOCKING BLOCKS
def _draw_game_area(screen, game_area):
    if game_area:
        for i, row in enumerate(game_area):
            for j, cell in enumerate(row):
                if cell != '.':
                    pygame.draw.rect(screen, BLOCK_COLOR, (j * SIZE, i * SIZE, SIZE, SIZE), 0)


# DRAW UNIT BLOCK
def _draw_block(screen, block, offset_x, offset_y, pos_x, pos_y):
    if block:
        for i in range(block.start_pos.Y, block.end_pos.Y + 1):
            for j in range(block.start_pos.X, block.end_pos.X + 1):
                if block.template[i][j] != '.':
                    pygame.draw.rect(screen, BLOCK_COLOR,
                                     (offset_x + (pos_x + j) * SIZE, offset_y + (pos_y + i) * SIZE, SIZE, SIZE), 0)


# DRAW INFO
def _draw_info(screen, font, pos_x, font_height, score):
    print_text(screen, font, pos_x, 10, f'SCORE: ')
    print_text(screen, font, pos_x, 10 + font_height + 6, f'{score}')
    print_text(screen, font, pos_x, 20 + (font_height + 6) * 2, f'SPEED: ')
    print_text(screen, font, pos_x, 20 + (font_height + 6) * 3, f'{min(score // 100, 99)}')
    print_text(screen, font, pos_x, 30 + (font_height + 6) * 4, f'NEXT: ')


if __name__ == '__main__':
    main()
