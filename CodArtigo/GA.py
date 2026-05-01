"""Algoritmo Genético para otimização de rotas/cortes."""
import random

import numpy

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

from contextlib import contextmanager
from datetime import datetime, timedelta

from math import fabs


@contextmanager
def timeit(file_write=None):
    """Gerenciador de contexto para verificar e registrar o tempo de execução."""
    start_time = datetime.now()
    print(f"Tempo de Inicio (hh:mm:ss.ms) {start_time}", file=file_write)
    yield
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    print(f"Tempo de Termino (hh:mm:ss.ms) {end_time}", file=file_write)
    print(f"Tempo Total (hh:mm:ss.ms) {time_elapsed}", file=file_write)


def dist2pt(x1, y1, x2, y2):
    """Calcula a distância de Chebyshev entre as coordenadas de dois pontos."""
    return max(fabs(x2 - x1), fabs(y2 - y1))  # Distância de Chebyschev


def midPoint(x1, y1, x2, y2):
    """Encontra as coordenadas do ponto médio entre dois pontos."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def plotar(individuo, f, fig=False):
    """Gera gráficos exibindo o trajeto completo formado pelas arestas do indivíduo."""
    plt.close()
    fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    f1_axes.set_aspect('equal')

    x1, y1, x, y = [], [], [], []
    colors = ["red", "gray"]
    cutA = 1
    i1 = individuo[0][0]

    # Verifica a direção do corte na aresta inicial
    a1 = edges[i1] if individuo[1][0] == 0 else edges[i1][::-1]
    deslocamentos = []

    # Salva as coordenadas iniciais da primeira aresta
    x.append(a1[0][0])
    y.append(a1[0][1])
    x.append(a1[1][0])
    y.append(a1[1][1])

    # Gera a anotação visual com setas ou linhas dependendo da flag 'fig'
    if not fig:
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                    scale_units='xy', angles='xy', scale=1, color=colors[0])
        f1_axes.annotate(str(cutA), midPoint(*a1[0], *a1[1]))
    else:
        f1_axes.plot(x,y, 'r-', linewidth=1)

    cutA += 1

    # Itera sobre as arestas do indivíduo para desenhar o restante do caminho
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]

        # Considera a ordem normal ou reversa baseada no gene de direção (individuo[1])
        a1 = edges[i1] if individuo[1][i] == 0 else edges[i1][::-1]
        a2 = (
            edges[i2]
            if individuo[1][i + 1 if i + 1 < len(individuo[0]) else 0] == 0
            else edges[i2][::-1]
        )
        x1, y1, x, y = [], [], [], []

        # Se o fim de uma aresta não conectar ao início da próxima, registra um "deslocamento" (movimento em vazio)
        if a1[1] != a2[0]:
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

    # Adiciona os vetores de deslocamento em vazio no gráfico
    if not fig:
        for i in deslocamentos:
            f1_axes.annotate(i['annot'], (i['mid'][0] - 3, i['mid'][1]))
            f1_axes.quiver(*i['pontos'], width=.005,
                        scale_units='xy', angles='xy', scale=1, color=colors[1])

    f1_axes.set_xlim(*f1_axes.get_xlim())
    f1_axes.set_ylim(*f1_axes.get_ylim())
    plt.tight_layout()

    # Salva o arquivo da imagem nos diretórios respectivos
    fig1.savefig(f"{'figura' if fig else 'plots'}/ga/{f}.png", dpi=300)
    plt.close()


def genIndividuo(edges):
    """
    Gera um Indivíduo.

    Args:
        edges -> arestas do grafo a serem cortadas/percorridas

    O indivíduo possui dois cromossomos:
    individuo[0]: a ordem/sequência em que as arestas são visitadas
    individuo[1]: a direção do corte em cada aresta (0 para normal, 1 para reverso)
    """
    # Cria lista de bits aleatórios (0 ou 1) para a direção de corte
    v = [random.randint(0, 1) for i in range(len(edges))]
    random.shuffle(v)

    # Retorna uma permutação aleatória de índices das arestas e o vetor de direções
    return random.sample(range(len(edges)), len(edges)), v


def evalCut(individuo, pi=16.67, mi=400):
    """
    Avalia o custo do corte (Função de Fitness/Aptidão).

    Args:
        individuo -> cromossomo sendo avaliado
        pi -> velocidade de corte (cutting speed)
        mi -> velocidade de viagem/deslocamento vazio (travel speed)

    Se individuo[1][i] == 0, o corte segue a ordem natural da aresta.
    Caso contrário, o corte é feito na ordem inversa da aresta.
    """
    dist = 0
    i1 = individuo[0][0]
    a1 = edges[i1] if individuo[1][0] == 0 else edges[i1][::-1]

    # Adiciona o custo de viagem desde a origem até o ponto inicial da primeira aresta
    if a1 != (0.0, 0.0):
        dist += dist2pt(0.0, 0.0, *a1[0]) / mi

    # Adiciona o custo (tempo) do corte na primeira aresta
    dist += (dist2pt(*a1[0], *a1[1])) / pi

    # Itera calculando os custos de transição entre as demais arestas
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]

        a1 = edges[i1] if individuo[1][i] == 0 else edges[i1][::-1]
        a2 = (
            edges[i2]
            if individuo[1][i + 1 if i + 1 < len(individuo[0]) else 0] == 0
            else edges[i2][::-1]
        )

        # Se os vértices coincidem, apenas soma o tempo do próximo corte
        if a1[1] == a2[0]:
            dist += (dist2pt(*a2[0], *a2[1])) / pi
        # Se estão separados, soma o deslocamento vazio até a aresta e depois o corte nela
        else:
            dist += (dist2pt(*a1[1], *a2[0])) / mi + (dist2pt(*a2[0], *a2[1])) / pi

    # Retorna o cursor final de volta à origem
    iu = individuo[0][-1]
    au = edges[iu] if individuo[1][-1] == 0 else edges[iu][::-1]
    if au != (0.0, 0.0):
        dist += dist2pt(*au[1], 0.0, 0.0) / mi

    individuo.fitness.values = (dist,)
    return (dist,)


def main(pop=10000, CXPB=0.75, MUTPB=0.1, NumGenWithoutConverge=10, file=None):
    """
    Executa o Algoritmo Genético.

    Args:
        pop -> tamanho da população do AG
        CXPB -> Probabilidade de Cruzamento (Crossover Probability)
        MUTPB -> Probabilidade de Mutação (Mutation Probability)
        NumGenWithoutConverge -> Número de gerações aceitáveis sem melhoria (critério de parada)
        file -> arquivo de saída para gravar os logs da evolução
    """
    # Tempo limite máximo de execução definido para 5 minutos (300 segundos)
    tempo = timedelta(seconds=300)

    # Inicializa a população
    pop = toolbox.population(n=pop)

    gen, genMelhor = 0, 0

    # HallOfFame mantém rastreio do melhor indivíduo absoluto encontrado
    hof = tools.HallOfFame(1)

    # Configura a coleta de estatísticas (média, desvio padrão, mínimo e máximo do fitness)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Avalia a aptidão (fitness) de toda a população inicial
    list(toolbox.map(toolbox.evaluate, pop))
    melhor = min([i.fitness.values for i in pop])

    # Registra e exibe informações da primeira geração (gen=0)
    logbook = tools.Logbook()
    p = stats.compile(pop)
    logbook.record(gen=0, **p)
    logbook.header = "gen", "min", "max", "avg", "std"
    gens, inds = [], []
    gens.append(gen)
    inds.append(melhor[0])
    print(logbook.stream, file=file)
    hora = datetime.now()

    # Loop evolutivo executa enquanto não estourar o limite de gerações sem melhorias
    while gen - genMelhor <= NumGenWithoutConverge:
        # Seleciona os indivíduos da próxima geração
        offspring = toolbox.select(pop, len(pop))

        # Clona os indivíduos selecionados para evitar modificação por referência na população original
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Aplica cruzamento (crossover) e mutação nos descendentes
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                # O cromossomo 0 usa um operador focado em permutações (provavelmente PMX)
                toolbox.mate0(child1[0], child2[0])
                # O cromossomo 1 usa um crossover simples de bits
                toolbox.mate1(child1[1], child2[1])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                # Mutação por permutação no sequenciamento
                toolbox.mutate0(mutant[0])
                # Mutação com inversão de bit nas direções
                toolbox.mutate1(mutant[1])
                del mutant.fitness.values

        # Avalia apenas os indivíduos gerados que possuem uma aptidão (fitness) inválida
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        list(toolbox.map(toolbox.evaluate, invalid_ind))

        # A população atual é inteiramente substituída pelos novos descendentes
        pop[:] = offspring

        gen += 1
        minF = min([i.fitness.values for i in pop])

        # Se encontrou um valor mínimo menor, atualiza a melhor geração
        if minF < melhor:
            melhor = minF
            genMelhor = gen

        # Compila e envia métricas estatísticas da geração para os logs
        p = stats.compile(pop)
        logbook.record(gen=gen, **p)
        if gen - genMelhor <= NumGenWithoutConverge and gen != 1:
            print(logbook.stream)
        else:
            print(logbook.stream, file=file)

        hof.update(pop)
        gens.append(gen)
        inds.append(minF[0])

        # Condição de parada por tempo máximo
        if (datetime.now() - hora) > tempo:
            break

    return pop, stats, hof, gens, inds


# Grid Search de opções experimentais para o tamanho da População, Crossover e Mutação
opcoes = {'pop': [5000], 'elite': [.75], 'mut': [.15]}
op = []
for i in opcoes['pop']:
    for j in opcoes['elite']:
        for k in opcoes['mut']:
            op.append((i, j, k))

# Configuração da toolbox base do pacote DEAP
toolbox = base.Toolbox()

# Classe de Aptidão (Fitness) configurada para minimização (peso negativo)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Representação do Indivíduo (lista herdando características do criador)
creator.create("Individual", list, fitness=creator.FitnessMin)

edges = []

def ga(f):
    # Lê as instâncias dos problemas a partir de arquivos texto e constrói o formato das arestas
    file = open(f"../../datasets/instances/{f}").read().strip().split("\n")
    if file:
        n = int(file.pop(0))
        for i in range(len(file)):
            a = [float(j) for j in file[i].split()]
            edges.append([(a[0], a[1]), (a[2], a[3])])
    f=f.replace('.txt', '')

    # Registra o gerador de indivíduos, passando a lista global de edges
    toolbox.register("indices", genIndividuo, edges)

    # Inicializa a estrutura do indivíduo conectando-o à classe e ao seu respectivo gerador
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )

    # Configura como uma população é instanciada
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operador de Seleção configurado como torneio tamanho 3
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Operadores de Cruzamento (PMX para sequência, 2-pontos para flip binário)
    toolbox.register("mate0", tools.cxPartialyMatched)
    toolbox.register("mate1", tools.cxTwoPoint)

    # Operadores de Mutação (Shuffle para sequência de arestas, Flip bit para direção)
    toolbox.register("mutate0", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("mutate1", tools.mutFlipBit, indpb=0.05)

    # Função Objetivo / Avaliador mapeado e registrado
    toolbox.register("evaluate", evalCut)

    # Função para execução do map (essencial, pois pode ser substituída posteriormente caso se adote multiprocessing)
    toolbox.register("map", map)

    qtd = 1
    # Loop de testes iterando por todas combinações de hiperparâmetros
    for k in op:
        # Prepara e abre o arquivo de histórico com o nome contendo os parâmetros de entrada
        with open(f"resultados/ga/{f}_[{k[0]},{k[1]},{k[2]}].txt", mode="w+", encoding='utf-8') as file_write:
            print(f"GA: {f}", file=file_write)
            print(file=file_write)
            for i in range(qtd):
                iteracao = None
                print(f"Execução {i+1}:", file=file_write)
                print(
                    f"Parametros: Pop={k[0]}, CXPB={k[1]}, MUTPB={k[2]}, Parada=100",
                    file=file_write
                )

                # Gerenciador de contexto disparado para calcular tempo do processo
                with timeit(file_write=file_write):
                    iteracao = main(
                        pop=k[0],
                        CXPB=k[1],
                        MUTPB=k[2],
                        file=file_write
                    )

                # Extrai os dados do log
                print("Individuo:", iteracao[2][0], file=file_write)
                print("Fitness: ", iteracao[2][0].fitness.values[0], file=file_write)
                print("Gens: ", iteracao[3], file=file_write)
                print("Inds: ", [float(i) for i in iteracao[4]], file=file_write)
                print(file=file_write)

                # Chama a geração dos mapas/plots
                plotar(iteracao[2][0], f"{f}_[{k[0]}, {k[1]}, {k[2]}]")
                plotar(iteracao[2][0], f"{f}_[{k[0]}, {k[1]}, {k[2]}]", fig=True)

                # Gera o gráfico extra da convergência de aptidão por geração
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
                    f'melhora/ga/' + f"{f}_[{k[0]}, {k[1]}, {k[2]}]" + '.png',
                    dpi=300
                )
                plt.close()
