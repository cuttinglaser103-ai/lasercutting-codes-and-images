from os import mkdir
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import chebyshev
from shapely import Polygon, Point

from segment import segment_intersections
from tempo import timeit
from intersect import construir_arvore_direcionada, plotar_poligonos_e_arvore, cortar_arvore, distancia_e_aresta_ate_raiz, linha_esta_na_lista
from QT import preprocess_layout_with_quadtree, \
    optimize_cluster_connections_kdtree_vertices_with_full_connectivity
from distribuicao import instancias_pack, instancias_sep

plt.switch_backend('TkAgg')

# ==========================================
# LOOP PRINCIPAL DE PROCESSAMENTO
# ==========================================
# Itera sobre todas as instâncias geradas (tanto as empacotadas quanto as separadas)
for instance in (instancias_pack + instancias_sep):

    # 1. Criação da estrutura de diretórios para salvar os resultados
    try:
        p_out = './instances'
        mkdir(p_out)
    except:
        ... # Omitido: Se a pasta já existir, simplesmente ignora o erro
    try:
        p_plot = './plot'
        mkdir(p_plot)
    except:
        ...
    try:
        p_plot_result = './plot_result'
        mkdir(p_plot_result)
    except:
        ...

    # Abre (ou cria) um arquivo de texto para registrar o relatório da instância atual
    with open(f"{p_out}/{instance[0]}", 'w+', encoding='utf-8') as f:
        print(f"Processando arquivo: {instance[0]}", file=f)
        print(file=f)

        # Inicia a contagem de tempo de execução usando o gerenciador de contexto
        with timeit(file_write=f):
            layout = instance[1]

            # 2. Tratamento de Furos (Buracos nos Polígonos)
            # Se um polígono tem um anel interior (furo), ele é extraído e
            # adicionado ao layout como um polígono independente para que a máquina também o corte.
            for p in list(layout):
                if len(p.interiors) > 0:
                    for i in p.interiors:
                        layout.append(Polygon(i))

            # 3. Tratamento de Interseções e Sobreposições
            intersecoes = set()
            # Varre todas as combinações de peças buscando colisões
            for i, f1 in enumerate(layout):
                for j, f2 in enumerate(layout):
                    if i != j:
                        if f1.intersects(f2):
                            # Salva a dupla colidente ordenada para não processar duas vezes
                            intersecoes.add(tuple(sorted((i, j))))

            # Processa as interseções fatiando os segmentos compartilhados
            for i,j in list(intersecoes):
                part = segment_intersections(layout[i], layout[j])
                layout[i], layout[j] = [Polygon(k) for k in part]

            # 4. Clusterização e Otimização da Rede (Grafos)
            # Organiza o espaço usando a QuadTree
            clustered_positions = preprocess_layout_with_quadtree(layout)

            # Gera a Árvore Geradora Mínima (MST) conectando todos os vértices de forma ótima
            G = optimize_cluster_connections_kdtree_vertices_with_full_connectivity(clustered_positions, plot=False)

            # 5. Geração das Rotas de Corte
            for componente in nx.connected_components(G):
                subgrafo = G.subgraph(componente)

                # Descobre qual polígono está mais perto da origem (0,0) da máquina
                dists = [(i, distancia_e_aresta_ate_raiz(Point(0,0), clustered_positions[i])) for i in subgrafo.nodes]

                # Elege esse polígono como a 'Raiz' da árvore de corte
                raiz = min([(i, d[0][0].length if len(d[0]) > 0 else 0) for i, d in dists], key=lambda x: x[1])[0]

                # Constrói a árvore direcionada a partir da raiz eleita
                arvore = construir_arvore_direcionada(subgrafo, raiz)

                # Gera a sequência bruta de movimentos (o caminho do laser/faca)
                caminho = cortar_arvore(arvore, raiz)

                # 6. Pós-processamento do Caminho (Classificação Corte vs Deslocamento em Vazio)
                # O caminho retorna passos com a tag 'x'. Aqui é verificado se a linha já foi percorrida antes.
                # Se NÃO foi percorrida, tag 'c' (corte). Se JÁ foi, tag 'd' (deslocamento vazio / laser desligado).
                caminho = [(j[0], j[1] if j[1] != 'x' else ('c' if not linha_esta_na_lista(j[0], caminho[:i]) else 'd')) for i, j in enumerate(caminho)]

                # 7. Cálculo da Função Objetivo (FO)
                # Soma os tempos de cada trecho. Tempo = Distância de Chebyshev / Velocidade.
                # Velocidade de corte ('c') = 100/6. Velocidade em vazio = 400.
                fo = sum([chebyshev(i[0].coords[0], i[0].coords[1]) / (100/6 if i[1] == 'c' else 400) for i in caminho])

        # ==========================================
        # SAÍDA DE DADOS E GERAÇÃO DE GRÁFICOS
        # ==========================================
        print(file=f)
        print("Função Objetivo:",fo, file=f)
        print(file=f)

        # Salva o arquivo de log as coordenadas de cada linha do caminho e seu estado (c ou d)
        [print(*i[0].coords[0], *i[0].coords[1], i[1], file=f) for i in caminho]

        # Plota e salva a imagem do caminho detalhado com os índices das etapas
        plotar_poligonos_e_arvore(arvore, caminho=caminho, raiz=raiz, hierarquia=False)
        plt.savefig(f"{p_plot}/{instance[0].split('.')[0]}.png")
        plt.close()

        # Plota e salva a imagem do caminho "limpo", sem a poluição visual dos índices (números)
        plotar_poligonos_e_arvore(arvore, caminho=caminho, raiz=raiz, hierarquia=False, indices=False)
        plt.savefig(f"{p_plot_result}/{instance[0].split('.')[0]}.png")
        plt.close()
