# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model predicts whether a person's annual income exceeds $50,000 based on demographic and employment attributes from the U.S. Census Bureau. It is a scikit-learn `RandomForestClassifier` with 100 trees, a minimum of 5 samples per leaf, and a fixed random seed of 42. Categorical features are one-hot encoded with a `OneHotEncoder` (unknown categories are ignored), and the label is binarized with a `LabelBinarizer`. The trained model and encoder are saved as pickle files in the `model/` folder. The project was developed with Python 3.10, scikit-learn 1.5.1, and FastAPI 0.112.0.

## Intended Use
The model is intended for educational purposes, as a demonstration of a reproducible machine learning pipeline with continuous integration and a deployed REST API. It should not be used to make real decisions about individuals, such as decisions on lending, hiring, housing, or insurance.

## Training Data
The training data is the Census Income (Adult) dataset from the UCI Machine Learning Repository, provided as `data/census.csv`. It has 32,561 rows and 14 input features plus the `salary` label (`<=50K` or `>50K`). About 76% of the rows are labeled `<=50K`. The features include age, workclass, final weight, education, education-num, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. The data was split 80/20 with `train_test_split` (random seed 42), so about 26,048 rows were used for training.

## Evaluation Data
The remaining 20% of the data (about 6,513 rows) was held out as the test set. It was processed with the encoder and label binarizer fitted on the training data only, so no information from the test set leaked into training.

## Metrics
The model is evaluated with precision, recall, and F1 score (beta = 1) on the positive class (`>50K`). Precision is the share of predicted high earners who actually earn more than $50K, and recall is the share of actual high earners that the model finds. On the test set the model scored precision **0.77**, recall **0.62**, and F1 **0.69**. The same three metrics were computed for every value of every categorical feature and are stored in `slice_output.txt`. Performance varies by slice. For example, F1 was higher for Federal-gov workers than for Self-emp-not-inc workers. Slices with very few rows, such as rare native countries, produce unstable metrics.

## Ethical Considerations
The data includes sensitive attributes such as race, sex, and native country, and it reflects historical income inequalities in the United States. The model can learn and reproduce these patterns, so its predictions may be less accurate or less fair for some groups. The slice metrics show uneven performance across groups, and the data comes from a 1994 census sample, so it may not reflect today's population or economy. The dataset also contains personal information about individuals, although it is public and anonymized.

## Caveats and Recommendations
The model's recall is modest, so it misses a considerable share of people who earn more than $50K, and the class imbalance affects this. Users should not rely on it for individual decisions. Possible improvements include hyperparameter tuning, class weighting or resampling, cross-validation, and a fairness analysis on sensitive features. The model should be retrained and re-evaluated on current data before any real-world use. Metrics on small slices should be interpreted with caution because of the small sample sizes.