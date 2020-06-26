import pygame
import numpy as np
import matplotlib.pyplot as plt
from drawUI import Click_Element, draw_buttons, draw_screen, draw_prediction
from mouse_handler import board_position_handler, interaction_handler
from ml_utils import get_letter_from_label, load_model
from models.knn import KNNClassifier
from models.lda import LDAClassifier
from models.lgreg import LogisticRegression
from models.nnet import NeuralNet
from models.svd import SVDClassifier


def init_board():
    board = []

    for i in range(28):
        board.append([])
        for j in range(28):
            board[i].append(0)

    return board


def program_start(board_size):
    pygame.init()
    board = init_board()
    px_size = int(board_size / 28)
    screen = pygame.display.set_mode(
        [int(board_size * 5 / 3), board_size + 60])

    mouse_detector(board, board_size, px_size, screen)

    pygame.quit()


def apply_board(board):
    board = np.array(board)
    return board.reshape(1, 28, 28)


def logistic_regression_prediction(lr_model, board):
    return get_letter_from_label(lr_model.predict(apply_board(board)))


def knn_prediction(knn_model, board):
    return get_letter_from_label(knn_model.predict(apply_board(board)))


def lda_prediction(lda_model, board):
    return get_letter_from_label(lda_model.predict(apply_board(board)))


def nnet_prediction(nnet_model, board):
    return get_letter_from_label(nnet_model.predict(apply_board(board)))


def svd_prediction(svd_model, board):
    return get_letter_from_label(svd_model.predict(apply_board(board)))


def mouse_detector(board, board_size, px_size, screen):
    logistic_reg_model = load_model("data/LR.sav")
    knn_model = load_model("data/KNN.sav")
    lda_model = load_model("data/LDA.sav")
    nnet_model = load_model("data/NN.sav")
    svd_model = load_model("data/SVD.sav")

    stroke = 3
    running = True
    is_drawing = False
    prediction = ''
    while running:
        for event in pygame.event.get():
            mouse_pos = pygame.mouse.get_pos()
            draw_screen(screen, board, board_size, event.type,
                        interaction_handler(screen, mouse_pos, board_size))
            draw_prediction(screen, board_size, prediction)
            strokeDraw = pygame.font.Font('freesansbold.ttf', 20).render('Stroke = ' + str(stroke), True, (0, 0, 0))
            stroke_rect = strokeDraw.get_rect()
            stroke_rect.center = (screen.get_size()[0] * 4 / 5, screen.get_size()[1] * 19 / 20)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if (interaction_handler(screen, mouse_pos, board_size) == Click_Element.RESET):
                    prediction = ''
                    board = init_board()
                elif (interaction_handler(screen, mouse_pos, board_size) == Click_Element.MLR):
                    prediction = logistic_regression_prediction(logistic_reg_model, board)
                elif (interaction_handler(screen, mouse_pos, board_size) == Click_Element.KNN):
                    prediction = knn_prediction(knn_model, board)
                elif (interaction_handler(screen, mouse_pos, board_size) == Click_Element.LDA):
                    prediction = lda_prediction(lda_model, board)
                elif (interaction_handler(screen, mouse_pos, board_size) == Click_Element.SVD):
                    prediction = svd_prediction(svd_model, board)
                elif (interaction_handler(screen, mouse_pos, board_size) == Click_Element.NN):
                    prediction = nnet_prediction(nnet_model, board)
                is_drawing = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP_MINUS and stroke > 1:
                    stroke -= 2
                elif event.key == pygame.K_KP_PLUS and stroke < 5:
                    stroke += 2
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing and mouse_pos[0] < board_size and mouse_pos[1] < board_size:
                    index = board_position_handler(mouse_pos, px_size)
                    if index != None:
                        draw_stroke(board, index[0], index[1], stroke)
            elif event.type == pygame.MOUSEBUTTONUP:
                is_drawing = False
            if event.type == pygame.QUIT:
                running = False
            screen.blit(strokeDraw, stroke_rect)
        pygame.display.update()

def draw_stroke(board, x, y, stroke_px):
    startX = x - int((stroke_px - 1) / 2)
    startY = y - int((stroke_px - 1) / 2)
    stroke_x = stroke_px
    stroke_y = stroke_px
    if startX < 0:
        stroke_x += startX
        startX = 0
    if startY < 0:
        stroke_y += startY
        startY = 0
    for i in range(stroke_x):
        for j in range(stroke_y):
            if startX < len(board) and startY + j < len(board[0]): 
                board[startX][startY + j] = 1
        startX += 1

board_size = 560 + int(560 / 3)
program_start(board_size)
