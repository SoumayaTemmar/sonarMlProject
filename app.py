from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


application = Flask(__name__)
app = application


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/predictData', methods=['GET', 'POST'])
def predict_datapoint():
   if request.method == 'GET':
      return render_template('home.html')
   else:

      # Get text input and convert to float list
      raw_features = request.form['features']
      features = [float(x.strip()) for x in raw_features.split(',')]

      if len(features) != 60:
         return "Error: Expected 60 features!", 400

      data = CustomData(*features)
      data_as_df = data.get_data_as_data_frame()

      predict_pipeline = PredictPipeline()
      prediction = predict_pipeline.predict(data_as_df)
      return render_template('home.html', prediction_text=f"Predicted Class: {prediction[0]}")
   
if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
