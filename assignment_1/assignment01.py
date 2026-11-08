# ==============================================================================
# ASSIGNMENT 1:  Rigid Body Motion and Perspective Projection Model
# NOMES: Fernanda Coutinho Corrêa Calazans - 2025101420
#        Augusto Lopes Zanotelli           - 2025101427        
# ANEXOS: requirements.txt
# ==============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
from stl import mesh
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit,
                             QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QFileDialog)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.set_printoptions(precision=3, suppress=True)


###### Funções auxiliares

# Movimentos no referencial do mundo: multiplica pela esquerda
def move(dx, dy, dz, obj=None):
    T = np.eye(4)
    T[0:3, 3] = [dx, dy, dz]
    if obj is None: return T
    else:           return T, T @ obj

def x_rotation(angle, obj=None):
    a = angle * pi / 180
    R = np.eye(4)
    R[1, 1:3] = [cos(a), -sin(a)]
    R[2, 1:3] = [sin(a),  cos(a)]
    if obj is None: return R
    else:           return R, R @ obj

def y_rotation(angle, obj=None):
    a = angle * pi / 180
    R = np.eye(4)
    R[0, [0, 2]] = [ cos(a), sin(a)]
    R[2, [0, 2]] = [-sin(a), cos(a)]
    if obj is None: return R
    else:           return R, R @ obj

def z_rotation(angle, obj=None):
    a = angle * pi / 180
    R = np.eye(4)
    R[0, 0:2] = [cos(a), -sin(a)]
    R[1, 0:2] = [sin(a),  cos(a)]
    if obj is None: return R
    else:           return R, R @ obj

# Movimentos no referencial da própria câmera: multiplica pela direita
def own_move(dx, dy, dz, cam):
    T = move(dx, dy, dz)
    return T, cam @ T

def own_x_rotation(angle, cam):
    R = x_rotation(angle)
    return R, cam @ R

def own_y_rotation(angle, cam):
    R = y_rotation(angle)
    return R, cam @ R

def own_z_rotation(angle, cam):
    R = z_rotation(angle)
    return R, cam @ R

def draw_arrows(point, base, axis, length=1.5):
    # cada coluna de base é um eixo (x, y, z)
    cores = ['red', 'green', 'blue']
    for i in range(3):
        axis.quiver(point[0], point[1], point[2],
                    base[0, i], base[1, i], base[2, i],
                    color=cores[i], pivot='tail', length=length)
    return axis

def carregar_stl(caminho):
    modelo = mesh.Mesh.from_file(caminho)
    # 3 x N, um vértice por coluna (3 colunas seguidas = 1 triângulo)
    vertices = modelo.vectors.reshape(-1, 3).T
    return np.vstack((vertices, np.ones((1, vertices.shape[1]))))


class MainWindow(QMainWindow):

    # função que inicializa a janela principal
    def __init__(self):
        super().__init__()

        self.set_variables()
        self.setWindowTitle("Assignment 01")
        self.setGeometry(100, 100, 1280, 720)
        self.setup_ui()

        # abre o seletor de arquivo assim que a janela aparecer
        QTimer.singleShot(0, self.abrir_stl)

    # função que inicializa as primeiras variáveis do programa
    def set_variables(self):
        self.objeto_original = None
        self.objeto = None

        # câmera em inicia (0,-100,50) pra ficar já de frente pro objeto
        self.cam_original = move(0, -100, 50) @ x_rotation(-90)
        self.cam = self.cam_original.copy()

        # parâmetros intrínsecos
        self.px_base = 1280
        self.px_altura = 720
        self.dist_foc = 50
        self.stheta = 0
        self.ox = self.px_base / 2
        self.oy = self.px_altura / 2
        self.ccd = [36, 24]
        self.projection_matrix = []
        self.parametros_iniciais = {"px_base": self.px_base,
                                    "px_altura": self.px_altura,
                                    "dist_foc": self.dist_foc,
                                    "stheta": self.stheta,
                                    "ccd": self.ccd.copy()}

    # função que cria o esqueleto principal da interface 
    def setup_ui(self):
        grid_layout = QGridLayout()

        # criando os widgets pra colocar os valores pro usuário digitar a partir das funções que detalham cada widhget
        line_edit_widget1 = self.create_world_widget("Ref mundo")
        line_edit_widget2 = self.create_cam_widget("Ref camera")
        line_edit_widget3 = self.create_intrinsic_widget("params instr")

        self.canvas = self.create_matplotlib_canvas()

        grid_layout.addWidget(line_edit_widget1, 0, 0)
        grid_layout.addWidget(line_edit_widget2, 0, 1)
        grid_layout.addWidget(line_edit_widget3, 0, 2)
        grid_layout.addWidget(self.canvas, 1, 0, 1, 3)

        # definindo layout pra colocar os botões de reset e pra abrir STL
        reset_widget = QWidget()
        reset_layout = QHBoxLayout()
        reset_widget.setLayout(reset_layout)

        # botão de reset
        reset_button = QPushButton("Reset")
        reset_button.setFixedSize(50, 30)
        reset_button.clicked.connect(self.reset_canvas)

        # botão de abrir STL
        botao_abrir_stl = QPushButton("Abrir STL")
        botao_abrir_stl.setFixedSize(70, 30)
        botao_abrir_stl.clicked.connect(self.abrir_stl)

        # adicionando os widgets do botões dentro do layout de reset e depois colocando esse layout dentro do layout do grid
        reset_layout.addWidget(reset_button)
        reset_layout.addWidget(botao_abrir_stl)
        grid_layout.addWidget(reset_widget, 2, 0, 1, 3)

        # unindo todos o resto dentro de um widget central pra colocar dentro da janela principal
        central_widget = QWidget()
        central_widget.setLayout(grid_layout)
        self.setCentralWidget(central_widget)

    # detalha graficamente o widget que vai receber os parâmetros intrínsecos da câmera
    def create_intrinsic_widget(self, title):
        # define o título e o layout do widget
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        grid_layout = QGridLayout()

        line_edits = []
        labels = ['n_pixels_base:', 'n_pixels_altura:', 'ccd_x:', 'ccd_y:', 'dist_focal:', 'sθ:']

        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()
            line_edit.setValidator(validator)
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        self.intr_edits = line_edits

        update_button = QPushButton("Atualizar")
        update_button.clicked.connect(lambda: self.update_params_intrinsc(line_edits))

        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        return line_edit_widget

    # detalha graficamente o widget que vai receber os parâmetros de posição e rotação da câmera no referencial do mundo
    def create_world_widget(self, title):
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']

        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()
            line_edit.setValidator(validator)
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        self.world_edits = line_edits

        update_button = QPushButton("Atualizar")
        update_button.clicked.connect(lambda: self.update_world(line_edits))

        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        return line_edit_widget

    # detalha graficamente o widget que vai receber os parâmetros de posição e rotação da câmera no referencial da própria câmera
    def create_cam_widget(self, title):
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']

        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()
            line_edit.setValidator(validator)
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        self.cam_edits = line_edits

        update_button = QPushButton("Atualizar")
        update_button.clicked.connect(lambda: self.update_cam(line_edits))

        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        return line_edit_widget

    # cria o canvas do matplotlib que vai receber a projeção 2D e a cena 3D
    def create_matplotlib_canvas(self):
        canvas_widget = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(canvas_layout)

        # imagem 2D (origem no canto superior esquerdo -> eixo y invertido)
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title("Imagem")
        self.ax1.set_xlim(0, self.px_base)
        self.ax1.set_ylim(self.px_altura, 0)
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')
        self.canvas1 = FigureCanvas(self.fig1)
        canvas_layout.addWidget(self.canvas1)

        # cena 3D
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.canvas2 = FigureCanvas(self.fig2)
        canvas_layout.addWidget(self.canvas2)

        # desenha só os referenciais (ainda não tem objeto, então inicializa sozinho só com os referenciais)
        self.update_canvas()

        return canvas_widget

    ###### Funções do programa

    # lê os seis campos de cada widget, devolve None para os vazios e troca , por . antes de converter para float pra não correr o risco de dar nenhum erro
    def ler_campos(self, line_edits):
        # campo vazio vira None; aceita vírgula como separador decimal
        valores = []
        for campo in line_edits:
            texto = campo.text().strip().replace(',', '.')
            valores.append(float(texto) if texto else None)
        return valores

    # atualiza os parâmetros intrínsecos da câmera a partir dos valores digitados pelo usuário
    def update_params_intrinsc(self, line_edits):
        # tenta ler o valores e se der errado mostra uma erro
        try:
            novos = self.ler_campos(line_edits)
        except ValueError:
            print("Erro: valor inválido")
            return

        # campo vazio mantém o valor atual e não altera nada
        atuais = [self.px_base, self.px_altura, self.ccd[0], self.ccd[1], self.dist_foc, self.stheta]
        px_base, px_altura, ccd_x, ccd_y, dist_foc, stheta = [
            atual if novo is None else novo for novo, atual in zip(novos, atuais)
        ]
        px_base, px_altura = int(px_base), int(px_altura)

        # verifica se esses valores são positivos, caso contrário mostra uma mensagem de erro
        if min(px_base, px_altura, ccd_x, ccd_y, dist_foc) <= 0:
            print("Erro: pixels, ccd e distância focal devem ser positivos")
            return

        # atualiza as variaveis iniciais com as entradas do usuário
        self.px_base = px_base
        self.px_altura = px_altura
        self.ccd = [ccd_x, ccd_y]
        self.dist_foc = dist_foc
        self.stheta = stheta

        # ponto principal no centro da imagem
        self.ox = self.px_base / 2
        self.oy = self.px_altura / 2

        # atualiza o canvas e mostra o resultado na nova matriz intrinseca no terminal
        self.update_canvas()
        print(self.generate_intrinsic_params_matrix())
    
    # atualiza os valores e o canvas pelo que foi digitado pelo usuário em relação ao referencial do mundo
    def update_world(self, line_edits):
        try:
            valores = self.ler_campos(line_edits)
        except ValueError:
            print("Erro: valor inválido")
            return

        dx, ax, dy, ay, dz, az = [v or 0 for v in valores]

        # rotação em torno dos eixos do mundo, depois translação no mundo
        T = move(dx, dy, dz)
        R = z_rotation(az) @ y_rotation(ay) @ x_rotation(ax)
        self.cam = T @ R @ self.cam

        self.update_canvas()

    # atualiza os valores e o canvas pelo que foi digitado pelo usuário em relação ao referencial da camera
    def update_cam(self, line_edits):
        try:
            valores = self.ler_campos(line_edits)
        except ValueError:
            print("Erro: valor inválido")
            return

        dx, ax, dy, ay, dz, az = [v or 0 for v in valores]

        # tudo nos eixos da própria câmera
        _, self.cam = own_move(dx, dy, dz, self.cam)
        _, self.cam = own_x_rotation(ax, self.cam)
        _, self.cam = own_y_rotation(ay, self.cam)
        _, self.cam = own_z_rotation(az, self.cam)

        self.update_canvas()

    # função que projeta os pontos do objeto 3D para o plano da imagem 2D, retornando as coordenadas 2D
    def projection_2d(self):
        if self.objeto is None:
            return np.empty((2, 0))

        K = self.generate_intrinsic_params_matrix()

        # mundo -> câmera
        pontos_cam = (np.linalg.inv(self.cam) @ self.objeto)[:3]
        z = pontos_cam[2]

        uv = K @ pontos_cam
        pontos_2d = np.full((2, pontos_cam.shape[1]), np.nan)

        # só projeta o que está na frente da câmera
        frente = z > 1e-9
        pontos_2d[:, frente] = uv[:2, frente] / z[frente]

        return pontos_2d

    # função que gera a matriz de parâmetros intrínsecos da câmera a partir dos valores atuais
    def generate_intrinsic_params_matrix(self):
        fx = self.dist_foc * self.px_base / self.ccd[0]
        fy = self.dist_foc * self.px_altura / self.ccd[1]

        self.projection_matrix = np.array([[fx, self.stheta, self.ox],
                                           [0,  fy,          self.oy],
                                           [0,  0,           1]], dtype=float)
        return self.projection_matrix

    # função que desenha a câmera no canvas 3D, chamando a função auxiliar draw_arrows
    def draw_camera(self, comprimento):
        draw_arrows(self.cam[:3, 3], self.cam[:3, :3], self.ax2, length=comprimento)

    # função de atualização do canvas, que desenha o objeto 3D, a câmera e os referenciais
    def update_canvas(self):
        # guarda o ângulo de visão pra não voltar pro padrão a cada atualização
        elev, azim = self.ax2.elev, self.ax2.azim
        self.ax2.clear()
        self.ax2.view_init(elev=elev, azim=azim)

        pos_cam = self.cam[:3, 3]
        pontos = [pos_cam, np.zeros(3)]

        if self.objeto is not None:
            vertices = self.objeto[:3].T
            triangulos = vertices.reshape(-1, 3, 3)
            self.ax2.add_collection3d(Poly3DCollection(triangulos, alpha=0.8,
                                                       edgecolor='black', linewidth=0.2))
            pontos += [vertices.min(axis=0), vertices.max(axis=0)]

        # limites cúbicos que cobrem objeto, câmera e origem do mundo
        pontos = np.array(pontos)
        centro = (pontos.max(axis=0) + pontos.min(axis=0)) / 2
        raio = (pontos.max(axis=0) - pontos.min(axis=0)).max() / 2 * 1.1
        if raio == 0:
            raio = 1

        self.ax2.set_xlim(centro[0] - raio, centro[0] + raio)
        self.ax2.set_ylim(centro[1] - raio, centro[1] + raio)
        self.ax2.set_zlim(centro[2] - raio, centro[2] + raio)
        self.ax2.set_box_aspect((1, 1, 1))
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("Y")
        self.ax2.set_zlabel("Z")
        self.ax2.set_title("Objeto 3D e câmera")

        comprimento = raio * 0.3
        draw_arrows(np.zeros(3), np.eye(3), self.ax2, length=comprimento)  # mundo
        self.draw_camera(comprimento)                                      # câmera

        self.canvas2.draw()
        self.update_projection_canvas()

    # função que atualiza o canvas da projeção 2D, chamando a função de projeção e desenhando os triângulos do stl que estão na frente da câmera
    def update_projection_canvas(self):
        self.ax1.clear()
        self.ax1.set_title("Imagem")

        pontos_2d = self.projection_2d()

        # cada 3 pontos consecutivos formam um triângulo; descarta os que ficaram atrás da câmera
        triangulos = pontos_2d.T.reshape(-1, 3, 2)
        triangulos = triangulos[np.isfinite(triangulos).all(axis=(1, 2))]
        if len(triangulos) > 0:
            self.ax1.add_collection(PolyCollection(triangulos, facecolors='none',
                                                   edgecolors='blue', linewidths=0.5))

        self.ax1.set_xlim(0, self.px_base)
        self.ax1.set_ylim(self.px_altura, 0)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True)
        self.canvas1.draw()

    # função pra reiniciar o canvas
    def reset_canvas(self):
        self.cam = self.cam_original.copy()

        self.px_base = self.parametros_iniciais["px_base"]
        self.px_altura = self.parametros_iniciais["px_altura"]
        self.dist_foc = self.parametros_iniciais["dist_foc"]
        self.stheta = self.parametros_iniciais["stheta"]
        self.ccd = self.parametros_iniciais["ccd"].copy()
        self.ox = self.px_base / 2
        self.oy = self.px_altura / 2

        for campo in self.world_edits + self.cam_edits + self.intr_edits:
            campo.clear()

        self.update_canvas()

    # função extra pa escolher o arquivo STL, que chama a função auxiliar carregar_stl e atualiza o canvas
    def abrir_stl(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Selecionar arquivo STL", "",
                                                 "Arquivos STL (*.stl)")
        if not caminho:
            return

        try:
            self.objeto_original = carregar_stl(caminho)
        except Exception as erro:
            print(f"Erro ao carregar STL: {erro}")
            return

        self.objeto = self.objeto_original.copy()
        self.update_canvas()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
