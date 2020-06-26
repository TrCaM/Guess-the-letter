import pygame
from enum import Enum


class Click_Element(Enum):
    BOARD = 1
    RESET = 2
    MLR = 3
    LDA = 4
    SVD = 5
    KNN = 6
    NN = 7


def draw_screen(screen, board, board_size, event, clicked_button):
    px_size = int(board_size / 28)
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (board_size, 0),
                     (board_size, board_size), 1)
    pygame.draw.line(screen, (0, 0, 0), (0, board_size),
                     (board_size, board_size), 1)
    draw_buttons(screen, board_size, event, clicked_button)

    reset = pygame.font.Font('freesansbold.ttf', 20).render(
        'Reset', True, (0, 0, 0))

    for i in range(28):
        for j in range(28):
            if board[j][i] == 1:
                pygame.draw.rect(screen, (0, 0, 0),
                                 (px_size * i, px_size * j, px_size, px_size))

def draw_prediction(screen, board_size, prediction):
  button_height = screen.get_size()[1] - board_size - 20 
  button_width = screen.get_size()[0] - board_size - 10

  prediction = pygame.font.SysFont('georgia', 350).render(prediction, True, (0, 0, 0))
  prediction_rect = prediction.get_rect()
  prediction_rect.center = (board_size + int(button_width / 2), int(screen.get_size()[1] * 3 / 5)) 
  screen.blit(prediction, prediction_rect)

def draw_buttons(screen, board_size, event, clicked_button):
  reset = pygame.font.Font('freesansbold.ttf', 20).render('Reset', True, (255, 255, 255))
  reset_rect = reset.get_rect()
  multi_lgistic_reg = pygame.font.Font('freesansbold.ttf', 20).render('Multinomial Logistic Regression', True, (0, 0, 0))
  multi_lgistic_reg_rect = multi_lgistic_reg.get_rect()
  lda = pygame.font.Font('freesansbold.ttf', 20).render('Linear Discriminant Analysis', True, (0, 0, 0))
  lda_rect = lda.get_rect()
  svd = pygame.font.Font('freesansbold.ttf', 20).render('SVD Method', True, (0, 0, 0))
  svd_rect = reset.get_rect()
  k_nearest_nbs = pygame.font.Font('freesansbold.ttf', 20).render('K Nearest Neighbors', True, (0, 0, 0))
  k_nearest_nbs_rect = k_nearest_nbs.get_rect()
  neural_net = pygame.font.Font('freesansbold.ttf', 20).render('Neural Network', True, (0, 0, 0))
  neural_net_rect = neural_net.get_rect()

  button_height = screen.get_size()[1] - board_size - 20 
  button_width = screen.get_size()[0] - board_size - 10

  pygame.draw.rect(screen, (80, 208, 70),
    (board_size + 5, 5, button_width, button_height))
  multi_lgistic_reg_rect.center = (board_size + 5 + int(button_width / 2), button_height / 2 + 5) 
  pygame.draw.rect(screen, (217, 77, 245),
    (board_size + 5, 10 + button_height, button_width, button_height))
  lda_rect.center = (board_size + 5 + int(button_width / 2), button_height * 3 / 2 + 10) 
  pygame.draw.rect(screen, (51, 215, 255),
    (board_size + 5, 15 + 2 * button_height, button_width, button_height))
  svd_rect.center = (board_size - 15 + int(button_width / 2), button_height * 5 / 2 + 15) 
  pygame.draw.rect(screen, (255, 201, 51),
    (board_size + 5, 20 + 3 * button_height, button_width, button_height))
  k_nearest_nbs_rect.center = (board_size + 5 + int(button_width / 2), button_height * 7 / 2 + 20) 
  pygame.draw.rect(screen, (255, 99, 51),
    (board_size + 5, 25 + 4 * button_height, button_width, button_height))
  neural_net_rect.center = (board_size + 5 + int(button_width / 2), button_height * 9 / 2 + 25) 
  if event == pygame.MOUSEBUTTONDOWN:
    if clicked_button == Click_Element.RESET:
      pygame.draw.rect(screen, (225, 65, 65), (
        5, board_size + 15, board_size - 5, button_height))
      reset_rect.center = (int(board_size / 2), board_size + 35)
    else:
      pygame.draw.rect(screen, (225, 65, 65), (
        5, board_size + 5, board_size - 5, button_height))
      pygame.draw.rect(screen, (144, 35, 35),
        (5, board_size + 45, board_size - 5, 10))
      reset_rect.center = (int(board_size / 2), board_size + 25)
  else:
      pygame.draw.rect(screen, (225, 65, 65), (
        5, board_size + 5, board_size - 5, button_height))
      pygame.draw.rect(screen, (144, 35, 35),
        (5, board_size + 45, board_size - 5, 10))
      reset_rect.center = (int(board_size / 2), board_size + 25)

  screen.blit(reset, reset_rect)
  screen.blit(multi_lgistic_reg, multi_lgistic_reg_rect)
  screen.blit(lda, lda_rect)
  screen.blit(svd, svd_rect)
  screen.blit(k_nearest_nbs, k_nearest_nbs_rect)
  screen.blit(neural_net, neural_net_rect)