import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Polygon, MultiPoint
from shapely.affinity import translate
from gerador_datasets import figpoly

# CONVEX HULL (FECHO CONVEXO)
def generate_clean_polygon(num_sides=random.randint(10, 25), size=50):
    """
    Gera um polígono convexo simples e limpo utilizando o algoritmo de Convex Hull.
    """
    # 1. Gera uma nuvem de pontos aleatórios dentro do tamanho especificado
    points = [(random.uniform(-size, size), random.uniform(-size, size)) for _ in range(num_sides)]

    # 2. Usa a biblioteca Shapely para criar o Convex Hull (o menor polígono convexo que envolve os pontos)
    # Isso garante que não haja cruzamentos (auto-interseções) e que o polígono seja simples.
    # O método convex_hull sempre retorna um Polygon único (ou Point/Line se num_sides for menor que 3)
    return MultiPoint(points).convex_hull


def generate_random_star_polygon(min_radius=10, max_radius=30, num_sides=8):
    """
    Gera um polígono em formato de estrela irregular (não convexa) definindo ângulos aleatórios.
    """
    # Define ângulos ordenados ao longo de 360 graus (2*pi radianos)
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_sides)])
    points = []

    for angle in angles:
        # Variar o raio de forma menos agressiva evita a criação de "pontas" muito finas,
        # o que poderia prejudicar o algoritmo de nesting mais tarde.
        r = random.uniform(min_radius, max_radius)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))

    poly = Polygon(points)

    # Verifica a validade geométrica do polígono gerado.
    # Se por erro numérico ele for considerado inválido, tenta corrigi-lo com a técnica de buffer(0)
    if not poly.is_valid:
        poly = poly.buffer(0)
        # Se a correção particionar o polígono em vários polígonos (MultiPolygon),
        # aciona o gerador de polígonos limpos (Convex Hull) como plano de segurança
        if poly.geom_type == 'MultiPolygon':
            poly = generate_clean_polygon(num_sides=num_sides, size=max_radius)

    return poly

def generate_regular_polygon(num_sides=6, min_radius=10, max_radius=30, center=(0, 0)):
    """
    Gera um polígono regular perfeito (lados e ângulos iguais).
    """
    if num_sides < 3:
        num_sides = 3  # Força no mínimo um triângulo para compor um polígono

    points = []
    # Determina o tamanho do polígono fixando o raio
    radius = random.uniform(min_radius, max_radius)
    for i in range(num_sides):
        # Calcula o ângulo exato para cada vértice mantendo a equidistância (distribuição perfeita no círculo)
        angle = 2 * math.pi * i / num_sides

        # Converte a coordenada polar para coordenadas cartesianas no plano XY
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))

    return Polygon(points)

def generate_polygon_with_hole(n_sides=6, m_sides=None, min_radius_ext=15, max_radius_ext=30, min_radius_int=5, max_radius_int=10):
    """
    Gera um polígono (regular externo) que pode ou não conter um furo no centro (interior irregular).

    :param n_sides: Número de lados da casca do polígono (exterior).
    :param m_sides: Número de lados do furo. Se for igual a None, nenhum furo é gerado.
    :param min_radius_ext: Raio mínimo aceitável para o limite externo.
    :param max_radius_ext: Raio máximo aceitável para o limite externo.
    :param min_radius_int: Raio mínimo aceitável para o buraco central.
    :param max_radius_int: Raio máximo aceitável para o buraco central.
    """
    # 1. Gera as coordenadas para a casca do polígono externo
    shell_points = []
    radius_ext = random.uniform(min_radius_ext, max_radius_ext)
    for i in range(n_sides):
        angle = 2 * math.pi * i / n_sides
        x = radius_ext * math.cos(angle)
        y = radius_ext * math.sin(angle)
        shell_points.append((x, y))

    holes = []

    # 2. Gera os vértices que compõem o buraco interno, se a quantidade de lados m_sides foi definida e for maior ou igual a 3
    if m_sides and m_sides >= 3:
        hole_points = []
        # Boa prática em CG (Computação Gráfica): inverter a ordem de desenho do furo ou desenhá-lo na direção oposta ao anel externo.
        radius_int = random.uniform(min_radius_int, max_radius_int)
        for i in range(m_sides):
            angle = 2 * math.pi * i / m_sides
            # As coordenadas são calculadas baseando-se no centro (0,0) para manter o furo centralizado dentro da casca
            x = radius_int * math.cos(angle)
            y = radius_int * math.sin(angle)
            hole_points.append((x, y))
        holes.append(hole_points)

    # Retorna um Polygon passando a casca construída e os buracos, caso existam.
    return Polygon(shell=shell_points, holes=holes)

def distribute_figures(figures, max_width=250, spacing_min=5, spacing_max=15, plot=True):
    """
    Responsável por distribuir/posicionar horizontal e verticalmente as figuras em uma chapa fictícia
    (Simula o empacotamento ou distanciamento pré-nesting).
    """
    positioned_figures = []
    x, y = 0, 0
    max_row_height = 0

    for fig in figures:
        # Extrai os limites da bounding box do polígono individual (Left, Bottom, Right, Top)
        min_x, min_y, max_x, max_y = fig.bounds
        width = max_x - min_x
        height = max_y - min_y

        # Se ultrapassar a largura máxima estipulada e não for a primeira peça da linha,
        # Reinicia o cursor x (volta para a borda esquerda) e incrementa y (pula para a próxima linha).
        if x + width > max_width and x != 0:
            x = 0
            y += max_row_height + random.randint(spacing_min, spacing_max)
            max_row_height = 0

        # Calcula a diferença entre onde o polígono precisa ir e onde o canto inferior esquerdo dele está no momento
        dx = x - min_x
        dy = y - min_y

        # Transfere/desloca o polígono para o layout montado na coordenada desejada
        translated_fig = translate(fig, xoff=dx, yoff=dy)
        positioned_figures.append(translated_fig)

        # Prepara o X para a próxima peça somando a largura e o espaçamento aleatório
        x += width + random.randint(spacing_min, spacing_max)

        # Mantém a maior altura da linha atual registrada, para que a próxima quebra de linha seja limpa
        if height > max_row_height:
            max_row_height = height

    # Visualização gráfica via Matplotlib (Apenas acionado caso a flag plot seja True)
    if plot:
        fig_plot, ax = plt.subplots(figsize=(12, 6))
        for poly in positioned_figures:
            color = random.choice(['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'])
            # A. Obtem as coordenadas do anel externo do polígono e preenche com a cor escolhida
            ext_x, ext_y = poly.exterior.xy
            ax.fill(ext_x, ext_y, facecolor=color, edgecolor="black", lw=1.5, zorder=1)

            # B. Preenche cada furo interno da figura, desenhando com a cor branca
            for interior in poly.interiors:
                int_x, int_y = interior.xy
                # A camada zorder=2 garante que a máscara branca se sobreponha à cor exterior
                ax.fill(int_x, int_y, facecolor="white", edgecolor="black", lw=1, zorder=2)

            # C. Esboça as linhas tracejadas da Bounding Box correspondente (Caixa Limite) (Opcional, ativado para debug)
            min_x, min_y, max_x, max_y = poly.bounds
            ax.plot([min_x, max_x, max_x, min_x, min_x],
                    [min_y, min_y, max_y, max_y, min_y],
                    color="red", linestyle=":", linewidth=0.8, alpha=0.4)

        # Ajusta os eixos baseados nas margens
        ax.set_xlim(-10, max_width + 10)
        ax.set_ylim(-10, y + max_row_height + 10)
        ax.set_aspect('equal')
        ax.set_title(f"Layout Distribuído - {len(figures)} Polígonos (3 a 8 lados)")
        # plt.gca().invert_yaxis()  # Opcional se for desejado inverter a visualização no estilo gráfico de computador
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.show()

    return positioned_figures


def analyze_spacing(figures):
    """
    Avalia a dispersão de tamanho dos polígonos presentes para deduzir qual seria
    um espaçamento 'min' e 'max' saudável e proporcional para a separação.
    """
    sizes = []

    for fig in figures:
        # Puxa os dados da Bounding box
        min_x, min_y, max_x, max_y = fig.bounds
        width = max_x - min_x
        height = max_y - min_y

        # Define a métrica de tamanho como o cálculo da hipotenusa/diagonal da caixa (Pode ser ajustado)
        diag = math.sqrt(width**2 + height**2)
        sizes.append(diag)

    sizes = np.array(sizes)

    # Extrai medidas estatísticas robustas
    median_size = np.median(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)

    # Definição fina dos espaçamentos ancorados no tamanho da mediana detectada
    spacing_min = median_size * 0.05   # 5% do tamanho típico
    spacing_max = median_size * 0.15   # 15% do tamanho típico

    return spacing_min, spacing_max


# ==========================================
# EXECUTANDO O TESTE PARA CRIAÇÃO DAS INSTÂNCIAS (ARQUIVOS DE DATASETS)
# ==========================================
random.seed(42)  # Reforça a consistência para resultados reproduzíveis

instancias_pack = []
instancias_sep = []

# 1. Cria instâncias compostas por Figuras Complexas (Estrelas irregulares)
for i in range(25):
    # Gera polígonos variando entre 10 a 45 por instância
    figuras = [generate_random_star_polygon() for _ in range(random.randint(10, 45))]
    larguras = [p.bounds[2] - p.bounds[0] for p in figuras]
    media = sum(larguras) / len(larguras)

    # Pack: Representa os polígonos perfeitamente "empacotados", colados uns aos outros (espaçamento 0)
    instancias_pack.append(
        (
            f'art_{i+1}_irregular_pack.txt',
            distribute_figures(
                figures=figuras,
                # Define dinamicamente o corte da largura combinando a média geométrica dos shapes
                max_width=math.ceil(media * math.ceil(math.sqrt(len(figuras)))),
                spacing_min=0,
                spacing_max=0,
                plot=False
            )
        )
    )

    # Regenera e faz o cenário Sep: Representa as instâncias "separadas" com respiros aleatórios
    figuras = [generate_random_star_polygon() for _ in range(random.randint(10, 45))]
    larguras = [p.bounds[2] - p.bounds[0] for p in figuras]
    media = sum(larguras) / len(larguras)
    instancias_sep.append(
        (
            f'art_{i+1}_irregular_sep.txt',
            distribute_figures(
                figures=figuras,
                max_width=math.ceil(media * math.ceil(math.sqrt(len(figuras)))),
                spacing_min=5,
                spacing_max=15,
                plot=False
            )
        )
    )

# 2. Cria instâncias compostas por Figuras Regulares (Triângulos, quadrados, hexágonos...)
for i in range(25):
    # Entre 3 a 8 lados por polígono
    figuras = [generate_regular_polygon(num_sides=random.randint(3, 8)) for _ in range(random.randint(10, 45))]
    larguras = [p.bounds[2] - p.bounds[0] for p in figuras]
    media = sum(larguras) / len(larguras)

    # Processa os layouts unidos sem espaçamento (pack)
    instancias_pack.append(
        (
            f'art_{i+1}_regular_pack.txt',
            distribute_figures(
                figures=figuras,
                max_width=math.ceil(media * math.ceil(math.sqrt(len(figuras)))),
                spacing_min=0,
                spacing_max=0,
                plot=False
            )
        )
    )

    # Gera layouts dispersos com margens (sep)
    figuras = [generate_regular_polygon(num_sides=random.randint(3, 8)) for _ in range(random.randint(10, 45))]
    larguras = [p.bounds[2] - p.bounds[0] for p in figuras]
    media = sum(larguras) / len(larguras)
    instancias_sep.append(
        (
            f'art_{i+1}_regular_sep.txt',
            distribute_figures(
                figures=figuras,
                max_width=math.ceil(media * math.ceil(math.sqrt(len(figuras)))),
                spacing_min=5,
                spacing_max=15,
                plot=False
            )
        )
    )

# 3. Cria instâncias compostas por Figuras contendo Furos
for i in range(25):
    figuras = []
    # Determina quantidades mistas de geração (Parte da lista vai para 'pack', parte vai para 'sep')
    pack = random.randint(10, 45)
    sep = random.randint(10, 45)
    for _ in range(pack + sep):
        # Cada polígono dessa batelada tem 50% de probabilidade de carregar um anel vazio
        tem_furo = random.choice([True, False])
        n = random.randint(3, 8)
        m = random.randint(3, 6) if tem_furo else None

        fig = generate_polygon_with_hole(n_sides=n, m_sides=m, max_radius_int=8)
        figuras.append(fig)

    # Separação das medidas e empacotamento apenas das partes designadas ao Pack
    larguras_pack = [p.bounds[2] - p.bounds[0] for p in figuras[:pack]]
    media_pack = sum(larguras_pack) / len(larguras_pack)

    larguras_sep = [p.bounds[2] - p.bounds[0] for p in figuras[pack:]]
    media_sep = sum(larguras_sep) / len(larguras_sep)

    instancias_pack.append(
        (
            f'art_{i+1}_hole_pack.txt',
            distribute_figures(
                figures=figuras[:pack],
                max_width=math.ceil(media_pack * math.ceil(math.sqrt(len(figuras[:pack])))),
                spacing_min=0,
                spacing_max=0,
                plot=False
            )
        )
    )
    # Empacotamento apenas da metada desenhada ao Sep
    instancias_sep.append(
        (
            f'art_{i+1}_hole_sep.txt',
            distribute_figures(
                figures=figuras[pack:],
                max_width=math.ceil(media_sep * math.ceil(math.sqrt(len(figuras[pack:])))),
                spacing_min=5,
                spacing_max=15,
                plot=False
            )
        )
    )

# Distribui e reordena os dados colhidos diretamente do gerador de datasets externo (figpoly)
for fig in figpoly:
    # Bagunça a lista de figuras importadas para o teste ser isento
    random.shuffle(fig[1])
    larguras = [p.bounds[2] - p.bounds[0] for p in fig[1]]
    media = sum(larguras) / len(larguras)

    instancias_pack.append(
        (
            # Formata a string contendo o nome final do arquivo original vindo de .csv ou .xml
            fig[0].split('/')[-1].replace('.csv', '_pack.txt').replace('.xml', '_pack.txt'),
            distribute_figures(
                figures=fig[1],
                max_width=math.ceil(media * math.ceil(math.sqrt(len(fig[1])))),
                spacing_min=0,
                spacing_max=0,
                plot=False
            )
        )
    )

    # Inicia a distribuição do Sep puxando os dados de folga via `analyze_spacing`
    spacing_min, spacing_max = analyze_spacing(fig[1])
    instancias_sep.append(
        (
            fig[0].split('/')[-1].replace('.csv', '_sep.txt').replace('.xml', '_sep.txt'),
            distribute_figures(
                figures=fig[1],
                max_width=math.ceil(media * math.ceil(math.sqrt(len(fig[1])))),
                spacing_min=int(spacing_min),
                spacing_max=int(spacing_max),
                plot=False
            )
        )
    )
