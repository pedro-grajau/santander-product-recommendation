# Recomendação de produtos Santander

Projeto de conclusão de Engenharia de Computação de Pedro Lucas Grajaú Farias, UEMA (2023). A pergunta é quais produtos um cliente tende a contratar no próximo mês, dado seu perfil e sua carteira anterior. O trabalho compara XGBoost multiclasse com uma combinação de popularidade, similaridade entre carteiras e árvores de decisão. A aplicação Streamlit apresenta o ranking do XGBoost.

## Documentos do TCC

- [Monografia completa, versão assinada (PDF)](docs/monografia.pdf)
- [Slides da apresentação (PDF)](docs/apresentacao.pdf)

## Executar

Na raiz do repositório, em um ambiente Python com as dependências instaladas:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Para habilitar a previsão, coloque os arquivos mensais da base Santander Product Recommendation em `data/train_AAAA_MM_28.csv` e execute `python train.py`. Para o cenário do TCC, são necessários os meses de maio de 2015 a abril de 2016. O script detecta aquisições pela passagem de 0 a 1 entre meses consecutivos do mesmo cliente. Treina com as aquisições de junho de 2015 a abril de 2016 e usa a carteira de abril para recomendar em maio de 2016. Os indicadores com posse desconhecida ficam fora da geração dos rótulos. O cadastro e os indicadores de cada exemplo vêm do mês anterior. As categorias são codificadas com o vocabulário de treino.

O treinamento grava na raiz `xgb_model.json`, `processed_test.csv` e `added_products.csv`. A aplicação também aceita o `xgb_model.pkl` produzido localmente no fluxo original, acompanhado dos dois CSVs da mesma execução. Use apenas pickles de origem confiável. O catálogo precisa conter os rótulos usados no treinamento. Para excluir produtos já contratados, a aplicação consulta as colunas de posse com o mesmo nome do rótulo. O novo treino preserva os códigos Santander, inclusive os dois produtos de previdência que os notebooks traduziam como `pensions`.

## Leitura dos resultados

| Resultado reportado | XGBoost | Híbrido |
| --- | ---: | ---: |
| Precisão na primeira posição | 70% | 45% |
| Acurácia para k = 1 | 43% | 22% |
| Recall entre sete sugestões | 98% | 86% |
| Precisão entre sete sugestões | 18% | 18% |

Fonte: tabelas 3 e 4 da monografia e slide 37 da defesa. O slide 47 atribui 70% à acurácia. A tabela identifica esse valor como precisão. Com sete sugestões, a cobertura das compras aumenta, enquanto a precisão por sugestão cai. O slide 33 também merece correção: F1 é a média harmônica de precisão e recall.

O trabalho reporta MAP@7 de aproximadamente 0,80 para o XGBoost. O notebook de avaliação parte de uma lista de aquisições e agrupa as previsões correspondentes. Isso condiciona a interpretação do resultado aos compradores avaliados. Para medir desempenho na base inteira, é preciso incluir clientes sem aquisição e explicitar o denominador.

O híbrido foi limitado a uma amostra de até 50 mil clientes no cálculo da similaridade e usou uma janela histórica diferente. Além disso, no notebook histórico, a contribuição da árvore está comentada na soma ponderada. A comparação publicada descreve esse experimento específico. Para isolar o efeito do algoritmo, seria necessário reavaliar ambos na mesma população e janela temporal.

## Validação da refatoração com dados reais

Usamos o arquivo `train_ver2.csv` da [competição Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation/data). A amostra inclui os clientes com `ncodpers % 20 == 0` e preserva o histórico de cada um. O XGBoost foi treinado por 400 rodadas com aquisições de junho de 2015 a abril de 2016 e aprendeu 22 classes presentes nessa amostra. Para o teste, usamos a carteira de abril e comparamos as sete primeiras sugestões de produtos ainda não contratados com as aquisições observadas em maio de 2016.

| População de teste | Clientes | MAP@7 |
| --- | ---: | ---: |
| Clientes presentes em abril e maio com carteira completa nos dois meses | 46.211 | 0,02589 |
| Clientes desse grupo que adquiriram algum produto em maio | 1.392 | 0,85942 |

O ranking cobriu 1.757 das 1.775 aquisições observadas (recall@7 de 98,99%). Na primeira linha, clientes sem aquisição recebem AP@7 igual a zero. Na segunda, a média considera apenas compradores. A diferença entre as linhas mostra por que o denominador importa ao comparar esse teste amostral com o resultado histórico do TCC.

## Código da pesquisa

Os seis notebooks em `notebooks/` registram a exploração, os modelos, os testes e a avaliação originais. As saídas salvas foram removidas. Eles usam caminhos relativos e dependências do ambiente de pesquisa. A instalação acima atende à aplicação e ao novo treinamento. Os números das tabelas são históricos. A refatoração usa apenas informações disponíveis antes do mês-alvo e requer uma nova avaliação para obter métricas próprias.
