# SkimLit

An NLP (Natural Language Processing) model that analyzes a research paper and categorizing it into objective, methods, results, etc, providing researchers to skim through the research paper easily with brief details.

![alt-text](https://github.com/gautamvr/SkimLit/blob/main/assets/skimlit.png)

NLP model created by analyzing different models:
- LSTM
- GRU
- CONV1D

Folowing are the various architectures used to analyze the accuracy of various models:

- **Baseline model (Model 0)** - TF-IDF Multinomial Naive Bayes
- **Model 1** - Conv1D with token embeddings
- **Model 2** - Feature extraction with pretrained token embeddings
- **Model 3** - Conv1D with character embeddings
- **Model 4** - Combining pretrained token embeddings + character embeddings (hybrid embedding layer)
- **Model 5** - Transfer Learning with pretrained token embeddings + character embeddings + positional embeddings

### Model 5 Architecture summary:

![alt-text](https://github.com/gautamvr/SkimLit/blob/main/assets/modelArchitecture.PNG)


#### The results for the above 6 models:

![alt-text](https://github.com/gautamvr/SkimLit/blob/main/assets/ResultsData.PNG)

![alt-text](https://github.com/gautamvr/SkimLit/blob/main/assets/ResultsDataPlot.PNG)

> As we can see from the plots, the tribrid model (Model 5) gives the highest accuracy - 82% even higher than the naive bayes model.

 These models have been created using best practices in NLP modelling, such as ending up with the efficient model by working the way up from a baseline model, by adjusting what is needed. To process the data faster, we have used `tf.data` API that enables faster Data processing with the help of (`batch`, `prefetch()`, `tf.data.AUTOTUNE` )

