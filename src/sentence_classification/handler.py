import pandas as pd
import os
from flask import Flask, request, Response
from keras.models                       import load_model
from text_classifier import SentenceClassifier
from tensorflow.keras                   import backend as K
from keras                              import saving

app = Flask(__name__)

@saving.register_keras_serializable()
def f2_score(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    fbeta_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    return K.mean(fbeta_score)


model = load_model('data/06_models/lstm_model.keras')


@app.route('/predict', methods=['POST'])
def sentence_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = SentenceClassifier()

        cleaned_data = pipeline.data_cleaning(test_raw)

        prepared_data = pipeline.data_preparation(cleaned_data)

        predicted_categories_list = pipeline.get_prediction(model, prepared_data)

        predicted_response = pd.DataFrame({'predicted_categories': predicted_categories_list})

        df_response = pd.concat([test_raw.reset_index(drop=True), predicted_response], axis=1)
        json_response = df_response.to_json(orient='records', date_format='iso')

        return json_response
    
    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
