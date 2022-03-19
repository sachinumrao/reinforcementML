import random
import sys
import time

import pygame as pg

import config


def get_new_ball():
    height = config.BALL_HEIGHT
    width = config.BALL_WIDTH
    x = (config.SCREEN_WIDTH - width) / 2
    y = (config.SCREEN_HEIGHT - height) / 2
    ball = pg.Rect(x, y, width, height)
    return ball


def get_new_paddle(side="left"):
    width = config.PADDLE_WIDTH
    height = config.PADDLE_HEIGHT
    paddle_offset = config.PADDLE_OFFSET

    y = (config.SCREEN_HEIGHT - height) / 2

    if side == "left":
        x = paddle_offset
    else:
        x = config.SCREEN_WIDTH - paddle_offset - width

    paddle = pg.Rect(x, y, width, height)
    return paddle


def ball_animation():
    pass


def main():
    # start pygame
    pg.mixer.pre_init(44100, -16, 2, 512)
    pg.init()
    clock = pg.time.Clock()

    # setup screen window
    screen = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pg.display.set_caption(config.GAME_TITLE)

    # game assets
    ball = get_new_ball()
    player = get_new_paddle("left")
    opponent = get_new_paddle("right")

    ball_speed_x = config.BALL_SPEED_X
    ball_speed_y = config.BALL_SPEED_Y

    player_speed = 0
    opponent_speed = config.OPPONENT_SPEED

    # text variables
    player_score = 0
    opponent_score = 0
    volley = 0

    game_font = pg.font.Font("freesansbold.ttf", 16)

    # add sound assets
    paddle_hit_sound = pg.mixer.Sound(config.PADDLE_SOUND)
    score_sound = pg.mixer.Sound(config.SCORE_SOUND)

    # event loop
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_DOWN:
                    player_speed += 7
                if event.key == pg.K_UP:
                    player_speed -= 7

            if event.type == pg.KEYUP:
                if event.key == pg.K_DOWN:
                    player_speed -= 7
                if event.key == pg.K_UP:
                    player_speed += 7

        # update location of ball
        ball.x = ball.x + ball_speed_x
        ball.y = ball.y + ball_speed_y

        player.y = player.y + player_speed

        # keep player within screen
        if player.top < 0:
            player.top = 0
        if player.bottom >= config.SCREEN_HEIGHT:
            player.bottom = config.SCREEN_HEIGHT

        # collide with vertical boundary of screen
        if ball.top <= 0 or ball.bottom >= config.SCREEN_HEIGHT:
            ball_speed_y = ball_speed_y * (-1)

        # check game termination condition
        if ball.left <= 0:
            # if ball goes to the left side, give opponents points and reset the ball
            opponent_score += 1
            ball.center = (config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2)

            # reset the volley counter
            volley = 0
            pg.mixer.Sound.play(score_sound)
            time.sleep(2)

        if ball.right >= config.SCREEN_WIDTH:
            # if ball goes to the right side, give points to the player and reset the ball
            player_score += 1
            ball.center = (config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2)

            # reset the volley counter
            volley = 0
            pg.mixer.Sound.play(score_sound)
            time.sleep(2)

        # check collision of ball and paddle
        if ball.colliderect(player) and ball_speed_x < 0:
            if abs(ball.left - player.right) < 10:
                ball_speed_x = ball_speed_x * (-1)
                volley += 1
                pg.mixer.Sound.play(paddle_hit_sound)
            elif abs(ball.bottom - player.top) < 10 and ball_speed_y > 0:
                ball_speed_y = ball_speed_y * (-1)
                volley += 1
                pg.mixer.Sound.play(paddle_hit_sound)
            elif abs(ball.top - player.bottom) < 10 and ball_speed_y < 0:
                ball_speed_y = ball_speed_y * (-1)
                volley += 1
                pg.mixer.Sound.play(paddle_hit_sound)

        if ball.colliderect(opponent) and ball_speed_x > 0:
            if abs(ball.right - opponent.left) < 10:
                ball_speed_x = ball_speed_x * (-1)
                volley += 1
                pg.mixer.Sound.play(paddle_hit_sound)
            elif abs(ball.bottom - opponent.top) < 10 and ball_speed_y > 0:
                ball_speed_y = ball_speed_y * (-1)
                volley += 1
                pg.mixer.Sound.play(paddle_hit_sound)
            elif abs(ball.top - player.bottom) < 10 and ball_speed_y < 0:
                ball_speed_y = ball_speed_y * (-1)
                volley += 1
                pg.mixer.Sound.play(paddle_hit_sound)

        # move opponent
        if opponent.top < ball.y:
            opponent.top = opponent.top + opponent_speed
        if opponent.bottom > ball.y:
            opponent.bottom = opponent.bottom - opponent_speed

        # keep opponent within screen
        if opponent.top < 0:
            opponent.top = 0
        if opponent.bottom >= config.SCREEN_HEIGHT:
            opponent.bottom = config.SCREEN_HEIGHT

        # add assets to screen (Display surface)
        screen.fill(config.BG_COLOR)

        # add middle line
        pg.draw.aaline(
            screen,
            config.LINE_COLOR,
            (config.SCREEN_WIDTH / 2, 0),
            (config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT),
        )

        pg.draw.rect(screen, config.PADDLE_COLOR, player)
        pg.draw.rect(screen, config.PADDLE_COLOR, opponent)
        pg.draw.ellipse(screen, config.BALL_COLOR, ball)

        # Add surface for score
        player_text = game_font.render(
            f"Player1:  {player_score:03d}", False, config.TEXT_COLOR
        )
        screen.blit(player_text, config.PLAYER_SCORE_LOC)

        opponent_text = game_font.render(
            f"Player2:  {opponent_score:03d}", False, config.TEXT_COLOR
        )
        screen.blit(opponent_text, config.OPPONENT_SCORE_LOC)

        volley_text = game_font.render(
            f"Current Volley:  {volley:03d}", False, config.TEXT_COLOR
        )
        screen.blit(volley_text, config.VOLLEY_SCORE_LOC)

        # Update window
        pg.display.flip()
        clock.tick(config.FPS)


if __name__ == "__main__":
    main()
