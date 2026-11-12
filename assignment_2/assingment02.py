# Assignment 2 - Computer Vision
# Name: Fernanda Coutinho Correa Calazans
#       Augusto Lopes Zanotelli

# Import the required libraries
# Add any other libraries you want

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Function to normalize points
# Input: points (image points to be normalized)
# Output: norm_points (normalized points)
#         T (normalization matrix)

def normalize_points(points):
  # Calculate centroid
  cx = np.mean(points[0,:])
  cy = np.mean(points[1,:])

  # Calculate the average distance of the points having the centroid as origin
  dist = np.sqrt((points[0,:]-cx)**2 + (points[1,:]-cy)**2)
  dist_med = np.mean(dist)

  # Define the scale to have the average distance as sqrt(2)
  S = np.sqrt(2)/dist_med

  # Define the normalization matrix (similar transformation)
  Mov = np.array(([1, 0, -cx], 
                  [0, 1, -cy], 
                  [0, 0,   1]))
  Sc = np.eye(3)*S
  Sc[2,2] = 1

  T = Sc@Mov

  # Normalize points
  norm_points = T@points
  
  return T, norm_points

# Function to build the matrix A of the DLT equation system
# Input: pts1, pts2 (points "pts1" from the first image and points "pts2" from the second image satisfying pts2 = H.pts1)
# Output: A (matrix containing the two or three rows resulting from the relation pts2 × H.pts1 = 0)

def compute_A(pts1, pts2):
    numpts = pts1.shape[1]

    A = np.zeros((2*numpts, 9))

    for i in range (numpts):
        A[2*i,3:6]     = -pts2[2,i]*pts1[:,i]
        A[2*i,6:9]     =  pts2[1,i]*pts1[:,i]

        A[2*i+1,0:3] =   pts2[2,i]*pts1[:,i]
        A[2*i+1,6:9]   =  -pts2[0,i]*pts1[:,i]

    return A

# Normalized DLT function
# Input: pts1, pts2 (points "pts1" from the first image and points "pts2" from the second image satisfying pts2 = H.pts1)
# Output: H (estimated homography matrix)

def compute_normalized_dlt(pts1, pts2):
    # Add homogeneous coordinates (n tinha neste arquivo atual)
    pts1 = pts1.T
    pts1 = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2 = pts2.T
    pts2 = np.vstack((pts2, np.ones(pts2.shape[1])))

    # Normalize points
    T1, p1 = normalize_points(pts1)
    T2, p2 = normalize_points(pts2)
    
    # Build the equation system by stacking the matrix A for each pair of normalized corresponding points
    A = compute_A(p1, p2)

    # Compute the SVD of the stacked matrix A and estimate the normalized homography H_normalized
    U,S,Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H_matrix = h.reshape((3,3))
    
    # Denormalize H_normalized to obtain H
    H = np.linalg.inv(T2) @ H_matrix @ T1
    
    return H


########################################################################################################################
# RANSAC function
# Inputs:
# pts1: points from the first image
# pts2: points from the second image
# dis_threshold: distance threshold to be used in RANSAC
# N: maximum number of iterations (it may be defined inside the function and should be updated
#    dynamically according to the number of inliers/outliers)
# Ninl: desired inlier threshold (it may or may not be used — this is your choice)
# Outputs:
# H: estimated homography
# pts1_in, pts2_in: set of inlier points from the first and second images


def RANSAC(pts1, pts2, dis_threshold, N, Ninl=0):  #Deixei o Ninl=0 pq nao usei essa coisa

    # Define other parameters such as the number of model samples, probabilities used in the equation for N, etc.
    s = 4                                       # Número de amostras mínimo (cada ponto fornece 2. Como precisamos de 8, é necessário ter 4 amostras)
    confianca = 0.99                            # Confiança mínima sobre as amostras de pares de pontos
    iteracoes_max = max(1, int(N))              # Define o numero máximo de iterações
    inliers_best = np.array([], dtype=int)      # Matriz "ponto de partida". Ela esta aqui pra dar alguma coisa pro algoritmo comparar inicialmente

    # Loop
    iteration = 0

    # While the stopping criterion is not satisfied 
    while iteration < iteracoes_max:

        # Randomly select "s" samples from the set of point pairs pts1 and pts2 
        sample_indices = np.random.choice(len(pts1), s, replace=False)  # Cata s (4) pontos do array de pontos de forma aleatoria
        sample_pts1 = pts1[sample_indices]  # Ao inves de fazer um for ou outro laço de repetição, coloco essa 'sample_indices' (que é um array) como indice
        sample_pts2 = pts2[sample_indices]  # que já retorna, dentro desssas duas variaveis, os sub-arrays com os pontos sorteados

        # Use the samples to estimate a homography using the Normalized DLT 

        # Se o algoritmo aleatorio achar 3 pontos colineares (mesma linha), ele da essa resposta de ignorar essa iteração.
        # Se eles forem colineares, a matriz dentro do 'compute_normalized_dlt' não conseguira ser invertida, o que causaria erro
        
        try:
            H_sample = compute_normalized_dlt(sample_pts1, sample_pts2)
        except np.linalg.LinAlgError:
            iteration += 1
            continue

        # Test this homography with the remaining point pairs using dis_threshold and count 
        # the number of supposed inliers obtained with the estimated model
        
        pts1_h = np.column_stack((pts1, np.ones(len(pts1))))                            # Faz a transformação para coordenadas homogeneas
        projected = (H_sample @ pts1_h.T).T                                             # Estima onde cada ponto deveria estar na segunda imagem 
        valid = np.abs(projected[:, 2]) > 1e-10                                         # Tá aqui pra evitar a divisao por 0. Ele faz a comparação da 3a coluna com um valor bem pequenininho e ve se ele é maior. Se for, é maior que zero
        errors = np.full(len(pts1), np.inf)                                             # Assumimos que p erro padrão de cada ponto é infinito ate provarmos o contrário, entao colocamos esse array como preenchido por valores infinitos "np.inf"
        projected[valid] = projected[valid] / projected[valid, 2, np.newaxis]           # Se os valores dos indices forem mostrados como válidos (foi feito 2 linhas atras), os calculos sao feitos usando os indices cuja a resposta da avaliação booleana do ">0" é TRUE
        errors[valid] = np.linalg.norm(projected[valid, :2] - pts2[valid], axis=1)      # Mesma coisa com os erros: só sao calculados erros para os pontos que sao validos fazer essa conta
        inlier_indices = np.where(errors < dis_threshold)[0]                            # Compara e pega os indices onde o erro é menor que o threshold dado pelo usuario

        # If the number of inliers is the largest obtained so far, store this set along with 
        # the "s" samples used.       
        # Also update the required number N of iterations
        if len(inlier_indices) > len(inliers_best):                         # Entra nesse loop caso os inliers sejam melhores que os das iterações anteriores
            inliers_best = inlier_indices                                   # Torna esse o grupo de melhores inliers

            # Update the number of iterations according to the current inlier ratio
            inlier_ratio = len(inliers_best) / len(pts1)                    # Vê qual é a razão de pontos bons em relação a quantidade total de pontos       
            probability = inlier_ratio ** s                                 # Vê qual é a probabilidade de se pegar 4 bons pontos ao acaso dentro desse 'pool' de pontos. (Se eu tenho um valor que me diz a razão de pontos bons em relação a todos os pontos (V), a chance de pegar 1 ponto bom ao acaso, pela probabilidade, se torna esse valor. A chance de pegar s valores bons se torna esse valor multiplicado por ele mesmo s vezes: V^s, ou V**s)
            if probability >= 1:                                               
                iteracoes_max = iteration + 1                               # Se achar 100% dos pontos perfeitos, para o loop agora
            elif probability > 0:
                required_iterations = math.log(1 - confianca) / math.log(1 - probability)               # Calcula, usando logaritmo, uma estipulação da quantidade de iterações necessárias pra termos 99% de certeza (confianca) de encontrarmos a matriz perfeita
                iteracoes_max = min(iteracoes_max, max(iteration + 1, math.ceil(required_iterations)))  # Muda o limite total de tentativas pra se adequar ao que ele acabou de calcular. Se escolhermos um valor muito alto de iterações sem encessidade e o ransac perceber que precisa de menos, ele abaixa essa quantdade automaticamente

        iteration += 1

    # After exiting the loop
    # Estimate the final homography H using all selected inliers.
    if len(inliers_best) >= s:                          # Agora a gente pega todos os pontos "inliers_best" e usa eles pra fazer 'o resto'
        pts1_in = pts1[inliers_best]
        pts2_in = pts2[inliers_best]
        H = compute_normalized_dlt(pts1_in, pts2_in)    # Faz 'o resto'
    else:
        # If no valid model was found, return the input points as empty inlier sets
        H = np.eye(3)
        pts1_in = np.empty((0, 2), dtype=np.float32)
        pts2_in = np.empty((0, 2), dtype=np.float32)

    return H, pts1_in, pts2_in


########################################################################################################################
# Application example for testing your function for estimating homography


MIN_MATCH_COUNT = 10
img1 = cv.imread("box.jpg", 0)         # queryImage 
img2 = cv.imread("photo01a.jpg", 0)    # trainImage 

# SIFT initialization
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
    
    #################################################
    # M = compute_normalized_dlt(src_pts, dst_pts) # trocar depois pra função com o ransac implementado
    M, src_pts_in, dst_pts_in = RANSAC(src_pts, dst_pts, dis_threshold=5.0, N=2000)
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor       = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags            = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('First image')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Second image')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('First image after warping')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
