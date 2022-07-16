
## Objetivo

Com o avanço dos dispositivos movéis no que tange a captação de áudio tornou-se possíıvel coletar e armazenar dados sonoros e com o advento da
inteligencia artificial se torna possivel a criação de interfaces de reconhecimento de voz que facilitar a vida das pessoas no dia á dia. 
Este trabalho propõem a criação de uma base de dados sonora com vozes humanas utilizando 150 arquivos de  áudio de 15 pessoas diferentes e um modelo de 
classificação automatica de digitos que são pronunciados por pessoas ao receberem ligações de empresas de telemarketing.

## Fonte de dados

Foi criado um repositorio com arquivos de audio gravados com tratamento de ruídos externo para cada número de 0 á 9 de 2 segundos gravados por
15 academicos advindos do curso bacharelado em ciência da computação da UNIFAP cada voluntario gravou dez arquivos totalizando 150 arquivos ao todo

## Metodologia

Para criar um classificador iremos usar a abordagem de recuperação de informações musicais, comummente referenciada pelo termo em inglês Music Information
Retrieval (MIR), é um emergente campo de pesquisa que trata da recuperação e organização de grandes coleções ou informações musicais, de acordo com sua 
relevância para consultas específicas. Esta prática tem se tornado extremamente relevante, dada a vasta quantidade de informações e serviços relacionados a 
música que existem atualmente mas podemos utilizar a mesma ideia para capturar informações do canto de cada espécie de pássaros as informações que são capturadas
nada mais são do que chromogramas, tempogramas e são baseados em frequências sonora mais informações na documentação do librosa biblioteca que usamos para esta 
finalidade: https://librosa.org/doc/main/feature.html

Uma vez que temos os nossos dados estruturados de forma tabular podemos aplicar feature engineer, feature select, técnicas de machine learning e validações robusta 
através de cross validate e k-fold, tunnin de hiperparametros e métricas de classificação. Segue os modelo de machine learning e outros algoritmos
- Bayes OPT
- RFE
- KNN
- MLP
- Decision Tree
- Logistc Regressor
- Random Forest
- XGBOOST
- LITHGBM
- CATBOOST

## Conclusão 

Após os primeiros experimentos nesta pesquisa não obtivemos resultados satisfatóios no entanto podemos observar que após selecionar as features as 35 
features mais importantes e aplicar bayes otimization para tunning de hiperparametros podemos ver uma pequena melhora nos resultados e constatar também 
que o fato de ter poucos dados na base contribuiu para resultados ruins.
