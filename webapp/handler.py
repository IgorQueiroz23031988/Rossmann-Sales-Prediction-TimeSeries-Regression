import pickle
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import os

# loading model
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))

# initialize API
app = Flask(__name__)


@app.route('/rossmann/predict', methods=['POST'])  # endpoint, where the original data with the forecast will be sent
def rossmann_predict():
    test_json = request.get_json()  # here pulls up the original csv files for both train and store

    if test_json:  # there is data
        # json to dataframe conversion
        if isinstance(test_json, dict):  # Unique example
            test_raw = pd.DataFrame(test_json, index=[0])

        else:  # Multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instantiate Rossmann class # get the information from rossmann class
        pipeline = Rossmann()

        # data cleaning # model preparation 1
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering # model preparation 2
        df2 = pipeline.feature_engineering(df1)

        # data preparation # model preparation 3
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw,
                                              df3)  # test raw is the original data and df3 is the prepared data

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)