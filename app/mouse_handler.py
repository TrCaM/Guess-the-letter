import pygame
from drawUI import Click_Element

def board_position_handler(mouse_pos, px_size): 
  for i in range(28):
    for j in range(28):
      xBox = px_size * i
      yBox = px_size * j
      if (mouse_pos[0] - xBox <= px_size and mouse_pos[1] - yBox <= px_size):
        return((j, i))

def interaction_handler(screen, mouse_pos, board_size):
  button_height = screen.get_size()[1] - board_size - 20 
  button_width = screen.get_size()[0] - board_size - 10
  if (mouse_pos[0] < board_size and mouse_pos[1] < board_size):
    return Click_Element.BOARD
  elif (mouse_pos[1] > board_size and mouse_pos[0] > 5 and mouse_pos[0] < board_size):
    return Click_Element.RESET
  elif (mouse_pos[0] > board_size and mouse_pos[1] > 5 and mouse_pos[1] < 5 + button_height):
    return Click_Element.MLR
  elif (mouse_pos[0] > board_size and mouse_pos[1] > 10 + button_height and mouse_pos[1] < 10 +  2 * button_height):
    return Click_Element.LDA
  elif (mouse_pos[0] > board_size and mouse_pos[1] > 15 + 2 * button_height and mouse_pos[1] < 15 + 3 * button_height):
    return Click_Element.SVD
  elif (mouse_pos[0] > board_size and mouse_pos[1] > 20 + 3 * button_height and mouse_pos[1] < 20 + 4 * button_height):
    return Click_Element.KNN
  elif (mouse_pos[0] > board_size and mouse_pos[1] > 25 + 4 * button_height and mouse_pos[1] < 25 + 5 * button_height):
    return Click_Element.NN