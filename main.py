from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        my_dict = request.form
        cough = int(my_dict['cough'])
        fever = int(my_dict['fever'])
        sore_throat = int(my_dict['sore_throat'])
        shortness_of_breath = int(my_dict['shortness_of_breath'])
        head_ache = int(my_dict['head_ache'])
        age_60_and_above = int(my_dict['age_60_and_above'])
        gender = int(my_dict['gender'])
        test_indication = int(my_dict['test_indication'])
        # Code for inference
        input_features = [[cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, gender, test_indication]]
        print(clf.predict(input_features))
        corona_prob = clf.predict_proba(input_features)[0][1]
        print(corona_prob)
        return render_template('show.html', corona=round(corona_prob*100))
    #return 'Hello, World! ' + str(corona_prob)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
