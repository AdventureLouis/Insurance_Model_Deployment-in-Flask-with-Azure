from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # loading the model

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    bmi = float(request.form["bmi"])
    New_Smoker= int(request.form["New_Smoker"])
    age= int(request.form["age"])
    prediction = model.predict([[bmi, New_Smoker,age]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2) 

    return render_template('index.html', prediction_text=f'A policy holder with {bmi} bmi,{New_Smoker} smoker  and {age} age will incure insurance cost of  $ {output}K')

if __name__ == "__main__":
    app.run()












# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.errorhandler(404)
# def invalid_route(e):
#     return "invalid_route."



# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the feature values from the form
# #     feature1 = float(request.form['bmi'])
# #     feature2 = float(request.form['New_Smoker'])

# #     # Perform the prediction
# #     prediction = model.predict([[feature1, feature2]])

# #     # Render the prediction result on the result.html template
# #     return render_template('result.html', prediction=prediction[0])



# @app.route('/predict', methods=['GET','POST'])
# def predictandoutput():
#     """
#     docstring
#     """
#     print("Inside predictandoutput function")
#     # model.train()
#     bmi = request.form["bmithin"]
#     New_Smoker = request.form["New_Smokerthin"]
#     # bmi,New_Smoker= (float(bmi),float(New_Smoker))
#     result = model.predict(bmi, New_Smoker)
#     return render_template("result.html",prediction=int(result[1]*100))


# @app.route('/prediction')
# def predict():

#     # int_features = [eval(x) for x in request.form.values()]
#     # final_features = [np.array(int_features)]
#     # prediction = model.predict(final_features)
    

#     # output = np.round(prediction[0], 2)

#     return render_template('index.html')

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True)