# Data Analytics II - WWU

Final project

Task II

Group 10


### Introduction

Na segunda parte do projeto final, empregamos técnicas relacionadas à redes neurais convolucionais com o objetivo de identificar paineis solares, piscinas, lagos e trampolins em imagens de satélites. 

Inicialmente, acreditamos que um modelo pré-treinado com imagens de satélite seria o mais adequado para desenvolver esta tarefa. Nesse sentido, pesquisamos por modelos pré-treinados na internet e encontramos a biblioteca Satallighte, que já continha dois modelos pré-treinados com imagens do dataset EuroSat, originalmente com X amostras divididas em 10 classes: rodovia, área urbana, área industrial, rio, floresta, e X. Além disso, esses imagens apresentavam-se em 240x240 pixels, bastante próximo ao formato 256x256 das imagens do dataset de treinamento fornecido para essa tarefa.

O modelo escolhido, denominado X, contém cerca de 2.2 milhões de parâmetros, com diversas camadas convolucionais, normalizações e dropouts. A biblioteca Satallighte foi desenvolvida usando PyTorch, e é uma mistura das palavras Satellite e Lightining (do pacote Pytoch Lightining).

Nós também utilizamos as imagens não nomeadas para treinar o modelo. Dividimos essas imagens em janelas de 256x256 com sobreposição de 50% para nomeá-las e, posteriormente, utilizá-las para treinar o modelo. As camadas densas do modelo foram substituídas por uma nova topologia que adequasse-se a 5 classes, ao invés de 10, como originalmente foram treinadas. Depois disso, todos os parâmetros anteriores às camadas densas foram congelados para que, inicialmente, apenas os pesos das camadas densas fossem ajustados. Posteriormente, nós descongelamos todos os pesos da rede e ajustamos todos os seus parâmetros. Após atingir a convergência, a equipe salvou o modelo num arquivo no formato .pt, exclusivo do PyTorch. 


![Coudn't display image final_prediction_exemple.png!](final_prediction_exemple.png "")

A fim de realizar predições nas imagens de validação fornecidas (8000x8000 pixels), nós utilizamos janelamento no tamanho 256x256 e com 50% de sobreposição. Ao final, para determinar as coordenadas de uma predição, calculamos a intersecção das predições de um objeto, como mostra a imagem. 

### Files explanation

hello world

