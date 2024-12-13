# IMTS Forecasting - Experimentos TCC do Porto de Paranaguá

Esse repositório contém o código, datos e modelos usados no experimento para defesa de Trabalho de Conclusão de Curso de Enzo Bustos Da Silva entitulado **"Irregular Multivariate Time Series e o Paranaguá Port Meteorological and Oceanographic Dataset"**. Trabalho focado na divulgação do P2MOD com alguns experimentos provando que o uso de técnicas que abordam diretamente as irregularidades são superiores à tecnicas de Processamento de Sinais que necessitam a regularização dos dados.

## Introdução

Este trabalho propõe uma contribuição significativa para o estudo das Séries Temporais Multivariadas Irregulares (IMTS), com foco no monitoramento do Porto de Paranaguá, maior porto graneleiro da América Latina. O conjunto de dados Paranaguá Port Meteorological and Oceanographic Dataset (P²MOD) preenche uma lacuna na literatura ao oferecer dados reais que refletem as irregularidades comuns em sensoriamento ambiental, permitindo o desenvolvimento de modelos mais robustos para tarefa de IMTS-forecasting usando AI/ML.

## Dataset

O Dataset usado para os experimentos é o **Paranaguá Port Meteorological and Oceanographic Dataset (P2MOD)**. Esse *dataset* consiste em:

- Medições Meteorológicas no Terminal Cattalini e Boia ODAS
- Medições de Correntes no Terminal Cattalini e Boia ODAS
- Altura da superfície do Mar no Terminal Cattalini e no Porto de Paranaguá
- Dados de Maré Astronômica e Maré Harmônica

O conjunto de dados exibe diversas irregularidades tornando-o uma boa referência para avaliar modelos em IMTS-forecasting.

## Modelos Testados

Os modelos avaliados foram feitos a partir de técnicas de AI/ML que incluem:

- GRU
- Gap-Ahead


## Resultados

Os resultados dos experimentos são medidos usando a Métrica de Index of Agreement (IoA) que mede a concordância entre os valores de dados previstos e reais.