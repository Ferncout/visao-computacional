# 👁️ Visão Computacional — UFES

Repositório dedicado ao desenvolvimento das entregas práticas e avaliações da disciplina de **Visão Computacional** da **Universidade Federal do Espírito Santo (UFES)**, com ênfase na fundamentação matemática e na implementação de algoritmos a partir de princípios fundamentais.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=flat&logo=qt&logoColor=white)

---

## 📂 Estrutura das Entregas

| Entrega | Tema | Descrição | Autoria |
|---------|------|-----------|---------|
| [`assignment_1`](./assignment_1) | 🎥 Câmeras e projeção | Simulação interativa em PyQt5 de modelagem de corpo rígido, projeção perspectiva 3D→2D e manipulação de câmeras virtuais utilizando matrizes de transformação homogênea. | Fernanda Coutinho e Augusto Lopes |
| [`assignment_2`](./assignment_2) | 🧩 Homografia planar | Estimativa de homografia ponta a ponta com **DLT Normalizado** e **SVD** implementados do zero, integrada a **SIFT**, **FLANN** e **RANSAC**. | Fernanda Coutinho e Augusto Lopes |

---

## 🎯 Conceitos Abordados

- 📐 Coordenadas homogêneas e transformações rígidas
- 🔭 Modelo de câmera pinhole e projeção perspectiva
- 🧮 Álgebra linear aplicada: decomposição SVD e sistemas homogêneos
- 🔗 Estimativa de homografia com DLT Normalizado
- ✨ Detecção e descrição de características (SIFT)
- 🔍 Correspondência de características (FLANN)
- 🛡️ Estimação robusta com RANSAC

---

## 🛠️ Stack Tecnológica

| Categoria | Ferramenta |
|-----------|------------|
| Linguagem | Python |
| Computação numérica | NumPy |
| Processamento de imagem | OpenCV |
| Interface gráfica e visualização | PyQt5, Matplotlib |

---

## 🚀 Como Executar

```bash
# Clonar o repositório
git clone https://github.com/Ferncout/visao-computacional-ufes.git
cd visao-computacional-ufes

# (Opcional) Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Instalar dependências
pip install numpy opencv-python matplotlib PyQt5
```

Cada entrega possui seu próprio diretório. Consulte os arquivos de cada pasta para executar o respectivo programa.

---

## ⚠️ Observação

O conteúdo deste repositório tem finalidade acadêmica e de registro pessoal.

## 👥 Autoria

**Fernanda Coutinho Correa Calazans**
🔗 [GitHub](https://github.com/Ferncout)

### 🤝 Colaborador

Algumas entregas foram desenvolvidas em dupla com **Augusto Lopes Zanotelli**
🔗 [GitHub](https://github.com/Augusto528)

> A autoria de cada atividade está indicada na tabela de entregas.
