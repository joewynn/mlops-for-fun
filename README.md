

 [Beyond Jupyter Notebooks](https://youtu.be/C7fBf33nQ7E)

setup instructions from 

[software development for mlops](https://github.com/FourthBrain/software-dev-for-mlops-101)

[x] Deploy a wine classifier with Docker and FastAPI

- The data
- train the model
- create the classifier
- serve with Docker and FastAPI
[x] Notes
- used a `continuum docker` file with `python 3.10` installed
- used that to ensure that `pickle` protocol for the model matches
- updated `pydantic` and used the new version 
- combined the batch and the non-batched models
- the results of the pickle file are monotonous, something that is not expected, so need to re-train the model and find something that will give better results, 
- wish we could change the model on the test different results

[] Train the other wine quality data and serve the model

[] build a pipeline of data, feature engineering, hyperperameter tuning, serving the model, managing the version of the model. 

[] Train and serve the wine recommendation dataset




https://www.askpython.com/python/wine-classification

https://github.com/RogueNPC/wine-classifier

https://jonathonbechtel.com/blog/2018/02/06/wines/

https://www.alldatascience.com/classification/wine-dataset-analysis-with-python/

well motivated from the business point of view more humane

[How to Build a Wine Quality Prediction Model Using Machine Learning?](https://labelyourdata.com/articles/machine-learning-for-wine-quality-prediction)