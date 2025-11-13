## sonar rock Vs Mine End_to_End ML project

in this project we will train a ML model to detect whether an object is a mine or a rock

dataset used: [Sonar data (Rock vs Mine) - Kaggle]: https://www.kaggle.com/datasets/mahmudulhaqueshawon/sonar-data

Each row in the CSV file likely represents a single measurement or observation, while the columns 
represent different parameters or features.

# 1-Features / Key Components

- **Data Ingestion:** get data and ingest it into artifacts folder
- **Data Transformation:** scaling numeric features.
- **Model Training:** Logistic Regression,K-Nearest Neighbors,SVC,Random Forest,Gradient Boosting, Naive Bayes,Neural Network (MLPClassifier)
- **Predict Pipeline:** Takes sonar features, and predicts label.
- **Evaluation:** Accuracy, precision, recall, F1-score.

# 2-Project structure
Fake_News_MLProject/
- │
- ├── artifacts/          # Preprocessed data, trained model, preprocessor
- ├── src/                # Source code
- │   ├── components/     # Data ingestion, transformation, model trainer, pipelines
- │   ├── utils.py        # Helper functions
- │   ├── exception.py    # Custom exceptions
- │   └── logger.py       # Logging setup
- ├── notebooks/          # Exploratory data analysis / experimentation
- ├── requirements.txt    # Project dependencies
- ├── setup.py            # for package building
- └── README.md           # This file


# 3-Installation & Setup
### Clone the repo
git clone <repo_url>

cd SONARMLPROJECT

### Create a virtual environment
conda create -p venv python=3.8 -y

conda activate venv/

### Install dependencies
pip install -r requirements.txt

# 4-Train the model
python src/pipelines/train_pipeline.py

# 5- run the app
python app.py 

- This launches a Flask app on http://localhost:5000 where you can paste news articles and get real-time predictions.
