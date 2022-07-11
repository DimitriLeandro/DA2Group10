# Data Analytics II - WWU

Final project

Task II

Group 10


## Introduction

Na segunda parte do projeto final, empregamos técnicas relacionadas à redes neurais convolucionais com o objetivo de identificar paineis solares, piscinas, lagos e trampolins em imagens de satélites. 

Inicialmente, acreditamos que um modelo pré-treinado com imagens de satélite seria o mais adequado para desenvolver esta tarefa. Nesse sentido, pesquisamos por modelos pré-treinados na internet e encontramos a biblioteca Satallighte, que já continha dois modelos pré-treinados com imagens do dataset EuroSat, originalmente com X amostras divididas em 10 classes: rodovia, área urbana, área industrial, rio, floresta, e X. Além disso, esses imagens apresentavam-se em 240x240 pixels, bastante próximo ao formato 256x256 das imagens do dataset de treinamento fornecido para essa tarefa.

O modelo escolhido, denominado X, contém cerca de 2.2 milhões de parâmetros, com diversas camadas convolucionais, normalizações e dropouts. A biblioteca Satallighte foi desenvolvida usando PyTorch, e é uma mistura das palavras Satellite e Lightining (do pacote Pytoch Lightining).

Nós também utilizamos as imagens não nomeadas para treinar o modelo. Dividimos essas imagens em janelas de 256x256 com sobreposição de 50% para nomeá-las e, posteriormente, utilizá-las para treinar o modelo. As camadas densas do modelo foram substituídas por uma nova topologia que adequasse-se a 5 classes, ao invés de 10, como originalmente foram treinadas. Depois disso, todos os parâmetros anteriores às camadas densas foram congelados para que, inicialmente, apenas os pesos das camadas densas fossem ajustados. Posteriormente, nós descongelamos todos os pesos da rede e ajustamos todos os seus parâmetros. Após atingir a convergência, a equipe salvou o modelo num arquivo no formato .pt, exclusivo do PyTorch. 


![Coudn't display image final_prediction_exemple.png!](final_prediction_exemple.png "")

A fim de realizar predições nas imagens de validação fornecidas (8000x8000 pixels), nós utilizamos janelamento no tamanho 256x256 e com 50% de sobreposição. Ao final, para determinar as coordenadas de uma predição, calculamos a intersecção das predições de um objeto, como mostra a imagem. 

## Files explanation

### model.zip

Este arquivo contém o modelo final utilizado para fazer as predições no formato .pt, exclusivo do PyTorch. É necessário possuir o arquivo .pt para rodar o arquivo create_predictions.py, explicado abaixo.

### create_predictions.py

Este é o arquivo .py que recebe como input uma pasta contendo imagens PNG para serem preditas. Para cada imagem PNG, o algoritmo salva um arquivo CSV contendo as coordenadas (x, y) das predições realizadas.

Como mencionado anteriormente, este projeto utilizou as bibliotecas Satellighte e PyTorch. Portanto, é necessário possuir esses pacotes instalados. Um arquivo requirements.pip será fornecido contendo todas as bibliotecas utilizadas e suas respectivas versões. O algoritmo foi testado no Python 3.8.10 e consegue abrir o modelo tanto em GPU quanto em CPU. 

Os testes realizados numa CPU Intel i5 8th Generation 4-cores mostraram que uma imagem 8000x8000 pode levar entre 12 e 15 minutos para ser predita.

Para rodar este algoritmo, é necessário fornecer argumentos na linha de comando na ordem descrita abaixo. Caso não sejam fornecidos, valores padrão serão utilizados. Além disso, o código contém comentários que explicam brevemente suas funções e procedimentos.

	Args:

		python3 predict_images.py imgs_path path_to_save_csvs path_to_model plot_predictions

		imgs_path (str)			-> 	Absolute path to folder containing PNG images to be predicted. 
									Default is the same path this python file is located.
		
		path_to_save_csvs (str)	->	Absolute path where to save the output CSVs. 
									Default is the same as imgs_path.
		
		path_to_model (str)		->	Absolute path to model checkpoint (suffix .pt).
									Default is the same path this python file is located
									plus model_ckpt.pt.
		
		plot_predictions (int)	->	Whether to plot images with predictions or not. 
									This value must be 0 (default) or 1. 

	Exemple:

			python3 
			predict_images.py 
			/home/dimi/validation_data/02_validation_data_images/
			/home/dimi/validation_data/02_validation_data_images/csv_results/ 
			/home/dimi/DA2Group10/task_2/results/model_checkpoints/model_checkpoint_all_layers_round_5-v1.pt 
			1

### model.zip

aaa

### model.zip

aaa

### model.zip

aaa

