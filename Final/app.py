from flask import Flask, render_template, request
import pickle


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
       
        transformed_message = vectorizer.transform([message])
        
        prediction = model.predict(transformed_message)
        
        
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        
        return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
