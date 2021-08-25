"""
Credit: Addison Sears-Collins (https://automaticaddison.com)
"""

import cv2 
import numpy as np 
 
def bin_ones(array):
  return np.ones_like(array) 

def bin_zeros(array):
  return np.zeros_like(array)

def binary_array(array, thresh, value=0):
  """
  Retorna um array 2D onde os pixels sao 0 ou 1
  """
  if value == 0:
    binary = bin_ones(array)

  else:
    binary = bin_zeros(array) 
    value = 1

  binary[(array >= thresh[0]) & (array <= thresh[1])] = value
  return binary

def blur_gaussian(channel, ksize=3):
  """
  Filtro gaussiano para reduzir ruido e detalhes da imagem
  ksize = tamanho do kernel k x k
  """
  return cv2.GaussianBlur(channel, (ksize, ksize), 0)
         
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
  """
  Filtro sobel para detecção de bordas
  """
  # Get the magnitude of the edges that are vertically aligned on the image
  sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
         
  # Get the magnitude of the edges that are horizontally aligned on the image
  sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))
 
  # Find areas that have the strongest pixel intensity changes in both the x and y directions. 
  # These have the strongest gradients and represent potential lane lines
  # mag = n rows x n columns = n pixels from top to bottom x n pixels from left to right
  mag = np.sqrt(sobelx ** 2 + sobely ** 2)
  # Return a 2D array that contains 0s and 1s   
  return binary_array(mag, thresh)
 
def sobel(img_channel, orient='x', sobel_kernel=3):
  """
  Find edges that are aligned vertically and horizontally on the image
  orient: Across which axis of the image are we detecting edges
  """
  # cv2.Sobel(input image, data type, order of the derivative x, order of the derivative y, small matrix used to calculate the derivative)
  if orient == 'x':
    # Will detect differences in pixel intensities going from left to right on the image (edges that are vertically aligned)
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
  if orient == 'y':
    # Will detect differences in pixel intensities going from top to bottom on the image (edges that are horizontally aligned)
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)
  return sobel
 
def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
  #If pixel intensity is greater than thresh[0], make that value white (255), else set it to black (0)
  return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)
  