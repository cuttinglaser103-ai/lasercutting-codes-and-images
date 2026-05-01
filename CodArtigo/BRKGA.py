"""Algoritmo Genético de Chaves Aleatórias Viciadas (BRKGA - Biased Random Keys Genetic Algorithm)."""
import os
import random

import numpy

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

from contextlib import contextmanager
from datetime import datetime, timedelta

from math import ceil, fabs


@contextmanager
def timeit(file_write=None):
    """Gerenciador de contexto para verificar e registrar o tempo de execução."""
    start_time = datetime.now()
    print(f'Tempo de Inicio (hh:mm:ss.ms) {start_time}', file=file_write)
    yield
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    print(f'Tempo de Termino (hh:mm:ss.ms) {end_time}', file=file_write)
    print(f'Tempo Total (hh:mm:ss.ms) {time_elapsed}', file=file_write)


def dist2pt(x1, y1, x2, y2):
    """Calcula a distância de Chebyshev entre as coordenadas de dois pontos."""
    return max(fabs(x2 - x1), fabs(y2 - y1))  # Distancia de Chebyschev


def midPoint(x1, y1, x2, y2):
    """Encontra as coordenadas do ponto médio entre dois pontos."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def plotar(indiv, f, fig=False):
    """Gera gráficos exibindo o trajeto completo formado pelas arestas do indivíduo decodificado."""
    # Decodifica as chaves aleatórias para obter a permutação e direção reais
    individuo = decode(indiv)

    fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    f1_axes.set_aspect('equal')

    x1, y1, x, y = [], [], [], []
    colors = ['red', 'gray']
    cutA = 1
    i1 = individuo[0][0]

    # Define a direção da primeira aresta com base no gene decodificado
    a1 = edges[i1] if individuo[1][0] == 0 else edges[i1][::-1]
    deslocamentos = []

    # Salva as coordenadas iniciais da primeira aresta
    x.append(a1[0][0])
    y.append(a1[0][1])
    x.append(a1[1][0])
    y.append(a1[1][1])

    if not fig:
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                    scale_units='xy', angles='xy', scale=1, color=colors[0])
        f1_axes.annotate(str(cutA), midPoint(*a1[0], *a1[1]))
    else:
        f1_axes.plot(x,y, 'r-', linewidth=1)

    cutA += 1

    # Itera pelas arestas subsequentes
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]  # aresta atual
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]  # proxima aresta

        # Considera a ordem normal ou reversa
        a1 = edges[i1] if individuo[1][i] == 0 else edges[i1][::-1]  # atual
        a2 = edges[i2] if individuo[1][
            i + 1 if i + 1 < len(individuo[0]) else 0] == 0 else edges[i2][::-1]  # proxima

        x1, y1, x, y = [], [], [], []

        if a1[1] != a2[0]:  # se a proxima não comecar onde a primeira termina (movimento em vazio)
            x1.append(a1[1][0])
            y1.append(a1[1][1])
            x1.append(a2[0][0])
            y1.append(a2[0][1])

            deslocamentos.append({
                'pontos': [x1[0], y1[0], x1[1] - x1[0], y1[1] - y1[0]],
                'annot': str(cutA),
                'mid': midPoint(*a1[1], *a2[0])
            })
            cutA += 1

        # plota a proxima
        x.append(a2[0][0])
        y.append(a2[0][1])
        x.append(a2[1][0])
        y.append(a2[1][1])

        if not fig:
            f1_axes.annotate(str(cutA), midPoint(*a2[0], *a2[1]))
            f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                        scale_units='xy', angles='xy', scale=1, color=colors[0])
        else:
            f1_axes.plot(x,y, 'r-', linewidth=1)
        cutA += 1

    # Desenha os vetores em vazio
    if not fig:
        for i in deslocamentos:
            f1_axes.annotate(i['annot'], (i['mid'][0] - 3, i['mid'][1]))
            f1_axes.quiver(*i['pontos'], width=.005,
                        scale_units='xy', angles='xy', scale=1, color=colors[1])

    f1_axes.set_xlim(*f1_axes.get_xlim())
    f1_axes.set_ylim(*f1_axes.get_ylim())
    plt.tight_layout()

    fig1.savefig(f'{'figura' if fig else 'plots'}/brkga/{f}.png', dpi=300)

    plt.close()


def genIndividuo(edges):
    """
    Gera um Indivíduo (Versão Padrão/Legada, não utiliza chaves aleatórias).

    Args:
        edges -> arestas do grafo a serem cortadas

    individuo[0]: ordem das arestas
    individuo[1]: ordem de corte (direção)
    """
    v = [random.randint(0, 1) for i in range(len(edges))]
    random.shuffle(v)
    return random.sample(range(len(edges)), len(edges)), v


def genIndividuoRK(edges):
    """
    Gera um Indivíduo com Chaves Aleatórias (Random Keys - Essencial para o BRKGA).

    Args:
        edges -> arestas do grafo a serem cortadas

    individuo[0]: chaves aleatórias para a ordem das arestas
    individuo[1]: chaves aleatórias para a ordem de corte (direção)
    """
    # Ambos os cromossomos são preenchidos com números reais (floats) no intervalo [0, 1)
    return [random.random() for i in range(len(edges))], [
        random.random() for i in range(len(edges))]


def decode(ind):
    """
    Decodifica as chaves aleatórias do indivíduo (vetor de floats) em uma solução real do problema.
    """
    # ind[0]: Ordena as chaves aleatórias e retorna os índices originais correspondentes (cria a permutação)
    # ind[1]: Transforma as chaves aleatórias de direção em valores binários (0 ou 1) usando 0.5 como limiar
    return [ind[0].index(i) for i in sorted(ind[0])], [0 if i < 0.5 else 1 for i in ind[1]]


def evalCut(individuo, pi=100 / 6, mi=400):
    """
    Avalia o custo do corte das arestas (Função de Fitness/Aptidão).

    Args:
        pi -> velocidade de corte
        mi -> velocidade de viagem (vazio)

    Se individuo[1][i] == 0 o corte segue a ordem da aresta.
    Caso contrário, o corte ocorre na ordem inversa da aresta.
    """
    # Decodifica o indivíduo de chaves aleatórias para uma sequência de índices
    ind = decode(individuo)
    dist = 0
    i1 = ind[0][0]
    a1 = edges[i1] if ind[1][0] == 0 else edges[i1][::-1]

    # Distância da origem até o início da primeira aresta
    if a1 != (0.0, 0.0):
        dist += dist2pt(0.0, 0.0, *a1[0]) / mi

    # Tempo para cortar a primeira aresta
    dist += (dist2pt(*a1[0], *a1[1])) / pi

    for i in range(len(ind[0]) - 1):
        i1 = ind[0][i]
        i2 = ind[0][i + 1 if i + 1 < len(ind[0]) else 0]
        a1 = edges[i1] if ind[1][i] == 0 else edges[i1][::-1]
        a2 = edges[i2] if ind[1][i + 1 if i + 1 < len(
            ind[0]
        ) else 0] == 0 else edges[i2][::-1]

        # Se as arestas estão conectadas
        if a1[1] == a2[0]:
            dist += (dist2pt(*a2[0], *a2[1])) / pi
        # Se há deslocamento em vazio
        else:
            dist += (dist2pt(*a1[1], *a2[0])) / mi + (
                dist2pt(*a2[0], *a2[1])) / pi

    # Retorno à origem
    iu = ind[0][-1]
    au = edges[iu] if ind[1][-1] == 0 else edges[iu][::-1]
    if au != (0.0, 0.0):
        dist += dist2pt(*au[1], 0.0, 0.0) / mi

    # No BRKGA, a aptidão precisa ser atribuída de volta ao objeto individuo
    individuo.fitness.values = (dist, )
    return dist,


def main(P=1000, Pe=0.2, Pm=0.3, pe=0.7, NumGenWithoutConverge=100, file=None):
    """
    Executa o Algoritmo Genético BRKGA.

    Args:
        P -> tamanho da população
        Pe -> proporção da população de elite
        Pm -> proporção da população mutante
        pe -> probabilidade de herdar um alelo do pai elite (biased crossover)
        NumGenWithoutConverge -> Número de gerações sem convergência aceitáveis (critério de parada)
        file -> arquivo onde os resultados serão escritos
    """
    tempo = timedelta(seconds=300)

    # Gera a população inicial
    pop = toolbox.population(n=P)

    # Registra o operador de cruzamento específico do BRKGA
    toolbox.register("mate", crossBRKGA, indpb=pe)

    # Define os tamanhos absolutos das partições da população
    tamElite = ceil(P * Pe)
    tamMutant = ceil(P * Pm)
    tamCrossover = P - tamElite - tamMutant

    gen, genMelhor = 0, 0

    hof = tools.HallOfFame(1)

    # Coleta de estatísticas da evolução
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Avalia a população inteira inicial
    list(toolbox.map(toolbox.evaluate, pop))

    melhor = numpy.min([i.fitness.values for i in pop])
    logbook = tools.Logbook()
    p = stats.compile(pop)
    logbook.record(gen=0, **p)
    logbook.header = "gen", 'min', 'max', "avg", "std"
    gens, inds = [], []
    gens.append(gen)
    inds.append(melhor)
    print(logbook.stream, file=file)
    hora = datetime.now()

    while gen - genMelhor <= NumGenWithoutConverge:
        # Ordena a população com base no fitness (elite ficará no topo)
        offspring = sorted(
            list(toolbox.map(toolbox.clone, pop)),
            key=lambda x: x.fitness,
            reverse=True
        )

        # Divide a população nas categorias do BRKGA
        elite = offspring[:tamElite]
        cross = offspring[tamElite:tamCrossover]
        c = []

        # Aplica o cruzamento parametrizado (Crossover viciado) na população
        for _ in range(tamCrossover):
            # Sempre seleciona um pai do conjunto Elite e outro do conjunto restante (Cross)
            e1 = random.choice(elite)
            c1 = random.choice(cross)
            ni = creator.Individual([[], []])
            ni[0] = toolbox.mate(e1[0], c1[0])
            ni[1] = toolbox.mate(e1[1], c1[1])
            c.append(ni)

        # Gera novos mutantes aleatórios do zero para manter a diversidade
        p = toolbox.population(n=tamMutant)

        # Constrói a próxima geração = Elite (intacta) + Filhos + Mutantes
        c = elite + c + p
        offspring = c

        # Avalia os indivíduos não-elite gerados nesta iteração
        list(toolbox.map(toolbox.evaluate, offspring[tamElite:]))

        # A população é substituída inteiramente pela nova geração
        pop[:] = offspring

        gen += 1
        minf = numpy.min([i.fitness.values for i in pop])
        men = False
        try:
            # Verifica se houve melhoria global nesta geração
            if minf < melhor:
                men = True
                melhor = minf
                genMelhor = gen
        except Exception:
            print(minf)

        # Grava métricas nos logs
        p = stats.compile(pop)
        logbook.record(gen=gen, **p)
        if gen - genMelhor <= NumGenWithoutConverge and not men:
            print(logbook.stream)
        else:
            print(logbook.stream, file=file)

        hof.update(pop)
        gens.append(gen)
        inds.append(minf)

        if (datetime.now() - hora) > tempo:
            break

    return pop, stats, hof, gens, inds


def crossBRKGA(ind1, ind2, indpb):
    """
    Cruzamento parametrizado/viciado do BRKGA.
    Com probabilidade `indpb` (geralmente alta, como 0.7) herda a chave do pai 1 (elite).
    Caso contrário, herda do pai 2 (não-elite).
    """
    return [ind1[i] if random.random() < indpb else ind2[i]
            for i in range(min(len(ind1), len(ind2)))]

# Configurações experimentais (grid search)
opcoes = {'pop': [5000], 'elite': [.3], 'mut': [.1]}
op = []
for i in opcoes['pop']:
    for j in opcoes['elite']:
        for k in opcoes['mut']:
            op.append((i, j, k))

# Toolbox principal do DEAP
toolbox = base.Toolbox()
# Classe de Fitness minimizadora
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Estrutura do Indivíduo
creator.create("Individual", list, fitness=creator.FitnessMin)
tipo = ['packing', 'separated']
edges = []

def brkga(f):
    # Leitura do dataset e construção da lista global de arestas
    file = open(f"../../datasets/instances/{f}").read().strip().split('\n')
    if file:
        n = int(file.pop(0))
        for i in range(len(file)):
            a = [float(j) for j in file[i].split()]
            edges.append([(a[0], a[1]), (a[2], a[3])])
    f=f.replace('.txt', '')

    # Registra o gerador que utiliza Chaves Aleatórias
    toolbox.register("indices", genIndividuoRK, edges)
    # Inicializa a estrutura do indivíduo conectando-o ao gerador de índices (chaves reais)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.indices
    )
    # Forma a população inteira
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Registra a função objetivo
    toolbox.register("evaluate", evalCut)
    # Registra a função map
    toolbox.register("map", map)

    for k in op:
        qtd = 1
        # Cria ou limpa os arquivos de log referentes a essa configuração
        with open(f"resultados/brkga/{f}_[{k[0]},{k[1]},{k[2]}].txt", mode='w+', encoding='utf-8') as \
                file_write:
            print(f"BRKGA: {f}", file=file_write)
            print(file=file_write)
            for i in range(qtd):
                print(f"Execução {i+1}:", file=file_write)
                print(
                    f"Parametros: P={k[0]}, Pe={k[1]}, Pm={k[2]}, pe=0.7, Parada=100",
                    file=file_write
                )
                iteracao = None

                # Executa o algoritmo BRKGA calculando o tempo
                with timeit(file_write=file_write):
                    iteracao = main(
                        P=k[0],
                        Pe=k[1],
                        Pm=k[2],
                        file=file_write
                    )

                # Escreve resultados após convergência ou interrupção
                print("Individuo:", decode(iteracao[2][0]), file=file_write)
                print("Fitness: ", iteracao[2][0].fitness.values[0], file=file_write)
                print("Gens: ", iteracao[3], file=file_write)
                print("Inds: ", [float(i) for i in iteracao[4]], file=file_write)
                print(file=file_write)

                # Plota a visualização da solução/rotas
                plotar(iteracao[2][0], f"{f}_[{k[0]}, {k[1]}, {k[2]}]")
                plotar(iteracao[2][0], f"{f}_[{k[0]}, {k[1]}, {k[2]}]", fig=True)

                # Cria o gráfico histórico do Fitness pelo avanço das gerações
                fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
                fig1.set_size_inches((10, 10))
                gens, inds = iteracao[3], iteracao[4]
                f1_axes.set_ylabel("Valor do Melhor Individuo")
                f1_axes.set_xlabel("Gerações")
                f1_axes.grid(True)
                f1_axes.set_xlim(0, gens[-1])
                f1_axes.set_ylim(inds[-1] - 10, inds[0] + 10)
                f1_axes.plot(gens, inds, color='blue')
                fig1.savefig(
                    f'melhora/brkga/' + f"{f}_[{k[0]}, {k[1]}, {k[2]}]" + '.png',
                    dpi=300
                )
                plt.close()
