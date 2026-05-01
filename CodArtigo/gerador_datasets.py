import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import random
import math

# Define uma seed fixa para garantir que a geração de números aleatórios seja reproduzível
random.seed(42)

# Datasets do repositorio: https://github.com/ESICUP/datasets/tree/main
files_2d = []

# Percorre todos os diretórios a partir de 'datasets'
for root, dirs, files in os.walk('datasets'):
    # Verifica se o diretório atual ou algum de seus pais no caminho começa com '2d_'
    path_parts = root.split(os.sep)
    if any(part.startswith('2d_') for part in path_parts):
        for file in files:
            # Filtra os arquivos que interessam: XML (Nesting) ou CSV
            if file.endswith('.xml'):
                files_2d.append(os.path.join(root, file))
            if file.endswith('.csv'):
                files_2d.append(os.path.join(root, file))


# -----------------------------
# Geometria
# -----------------------------

def is_valid_polygon(points):
    """Verifica se um conjunto de pontos forma um polígono válido e com área maior que zero."""
    poly = Polygon(points)
    return poly.is_valid and poly.area > 0

def fix_polygon(points):
    """
    Tenta corrigir um polígono problemático (como laços ou auto-interseções).
    Retorna uma lista de coordenadas (o anel externo) se corrigido.
    """
    poly = Polygon(points)

    # Aplicar buffer(0) é um truque clássico na biblioteca Shapely
    # para corrigir/unificar geometrias inválidas (auto-intersecção).
    fixed = poly.buffer(0)

    # Se a correção resultar em nada, falha gracefully
    if fixed.is_empty:
        return None

    # Se a correção gerou vários polígonos independentes,
    # escolhemos aquele que tem a maior área.
    if fixed.geom_type == "MultiPolygon":
      fixed = max(fixed.geoms, key=lambda g: g.area)

    # Pega apenas o contorno/anel externo e descarta o último ponto
    # (que é duplicado do primeiro para fechar o ciclo no formato interno do Shapely).
    return list(fixed.exterior.coords)[:-1]

def jitter(points, intensidade=5):
    """Adiciona um ruído ('jitter') aleatório às coordenadas dos pontos para deformar levemente o polígono."""
    return [
        (
            x + random.uniform(-intensidade, intensidade),
            y + random.uniform(-intensidade, intensidade),
        )
        for x, y in points
    ]


def transform(points, scale=1.0, rotation=0.0, tx=0, ty=0):
    """Aplica transformações afins (escala, rotação em radianos e translação/deslocamento) aos pontos."""
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    result = []
    for x, y in points:
        # 1. Aplica escala
        x *= scale
        y *= scale

        # 2. Aplica rotação em torno da origem
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r

        # 3. Aplica a translação e salva o ponto
        result.append((xr + tx, yr + ty))

    return result


def subdivide(points):
    """
    Subdivide cada aresta do polígono, criando um novo vértice
    exatamente no ponto médio de cada segmento.
    """
    new_pts = []
    n = len(points)

    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]

        new_pts.append(a)
        # Adiciona o ponto médio entre A e B
        new_pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

    return new_pts

def generate_valid_polygon(base_pts, max_attempts=10):
    """
    Aplica uma série de deformações (subdivisão, jitter e transformações afins)
    em uma cópia dos pontos base, tentando gerar um novo polígono válido.
    """
    for _ in range(max_attempts):
        pts = base_pts[:]

        # 50% de chance de subdividir as arestas antes de aplicar ruído
        if random.random() > 0.5:
            pts = subdivide(pts)

        # Adiciona o ruído/jitter de intensidade 5
        pts = jitter(pts, 5)

        # Aplica uma transformação aleatória de escala, rotação e translação
        pts = transform(
            pts,
            scale=random.uniform(0.9, 1.1),
            rotation=random.uniform(0, math.pi * 2),
            tx=random.uniform(-10, 10),
            ty=random.uniform(-10, 10),
        )

        # Tenta corrigir possíveis laços/auto-interseções que a deformação tenha causado
        pts_fixed = fix_polygon(pts)

        # Se conseguiu gerar e corrigir um polígono, e ele é válido, o retorna
        if pts_fixed and is_valid_polygon(pts_fixed):
            return pts_fixed

    # Se esgotar as tentativas (fallback seguro), retorna os pontos inalterados
    return base_pts

def clean_polygon(points):
    """
    Limpa e simplifica o objeto Polygon. Usa buffer(0) se estiver inválido e
    simplify para remover vértices praticamente colineares.
    """
    poly = Polygon(points)

    if not poly.is_valid:
        poly = poly.buffer(0)

    # Remove pontos redundantes, mantendo a tolerância/desvio máximo de 0.5 unidades
    poly = poly.simplify(0.5)

    return poly

######################################

def convert_polygons_generic(xml_file, modificar_poly=False):
    """
    Lê um arquivo XML de configuração de Nesting/Packing, extrai os polígonos e suas quantidades.
    Permite opcionalmente aplicar as deformações da função `generate_valid_polygon`.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Tratamento de Namespace (lida com o formato específico do repositório ESICUP)
    ns = {'n': 'http://www.fe.up.pt/~esicup/nesting.xsd'}
    if root.tag.startswith('{'):
        uri = root.tag.split('}')[0][1:]
        ns = {'n': uri}

    # 1. Mapear as quantidades de peças associadas aos polígonos
    polygon_quantities = {}
    for piece in root.findall('.//n:boards/n:piece', ns) + root.findall('.//n:lot/n:piece', ns):
        qty = piece.get('quantity', '1')
        comp = piece.find('n:component', ns)
        if comp is not None:
            poly_id = comp.get('idPolygon')
            if poly_id: polygon_quantities[poly_id] = qty

    # 2. Extrair polígonos buscando os nós no XML
    polygons = []
    for poly in root.findall('.//n:polygon', ns):
        poly_id = poly.get('id')

        # Ignora o "polygon0", que na convenção ESICUP costuma ser o bin/board (placa mãe)
        if poly_id and poly_id.startswith('polygon') and poly_id != 'polygon0':
            points = []

            # Lê cada segmento/vértice (x0, y0) do polígono
            for segment in poly.findall('.//n:segment', ns):
                x0 = float(segment.get('x0'))
                y0 = float(segment.get('y0'))
                points.append((x0, y0))

            if points:
                p = Polygon(points)
                # Se a flag for verdadeira, aplica a deformação randômica
                if modificar_poly:
                    pts = list(p.exterior.coords)[:-1]
                    pts = generate_valid_polygon(pts)
                    p = clean_polygon(pts)
                polygons.append((poly_id, p))

    # 3. Multiplica o polígono na lista final conforme sua quantidade (quantity)
    polys = []
    for poly_id, poly in polygons:
        qty = polygon_quantities.get(poly_id, '1')
        for _ in range(int(qty)):
            polys.append(poly)

    # Retorna o nome ajustado caso tenha ocorrido modificação e a lista final de polígonos
    return (xml_file.replace('.xml', '_mod.xml') if modificar_poly else xml_file, polys)

######################################

figpoly = []

# Processamento de arquivos CSV
for i in filter(lambda x: x.endswith('.csv'), files_2d):
    # Pula o cabeçalho (primeira linha) do arquivo
    file = open(i).readlines()[1:]
    polys = {}

    # Lê as linhas, dividindo pelo separador ponto e vírgula
    for l in file:
        l = l.replace('\n', '').split(';')

        # Estrutura do polys: polys[id_poligono][id_ponto] = (X, Y)
        if l[0] not in polys.keys():
            polys[l[0]] = {}
        polys[l[0]][l[1]] = (float(l[2]), float(l[3]))

    poligonos = {}
    for poly_id, pontos in polys.items():
        # Ordenar pelos índices dos pontos (convertendo as chaves para int) para garantir o anel fechado
        coords = [coord for _, coord in sorted(pontos.items(), key=lambda x: int(x[0]))]

        # Criar o polígono usando Shapely se possuir no mínimo 3 pontos
        if len(coords) >= 3:
            poligonos[poly_id] = Polygon(coords)

    # Guarda o arquivo original
    figpoly.append((i, list(poligonos.values())))

    # Gera a versão deformada (mod) das formas do CSV
    poligonos_mod = []
    for p in poligonos.values():
        pts = list(p.exterior.coords)[:-1]
        pts = generate_valid_polygon(pts)
        p = clean_polygon(pts)
        poligonos_mod.append(p)

    # Salva os polígonos deformados com a extensão _mod.csv
    figpoly.append((i.replace('.csv', '_mod.csv'), poligonos_mod))

# Executar a conversão normal para todos os arquivos XML
for xml_file in filter(lambda x: x.endswith('.xml'), files_2d):
    figpoly.append(convert_polygons_generic(f'{xml_file}'))

# Executar a conversão com deformação (modificar_poly=True) para todos os arquivos XML
for xml_file in filter(lambda x: x.endswith('.xml'), files_2d):
    figpoly.append(convert_polygons_generic(f'{xml_file}', modificar_poly=True))
