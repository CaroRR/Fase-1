from flask import Flask, render_template, request
import joblib

# Crear aplicación Flask

app = Flask(__name__)

# Cargar modelo y vectorizador

modelo = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    resultado = None
    if request.method == "POST":
        review = request.form["review"]
        vector = vectorizer.transform([review])

    # Predicción
        pred = modelo.predict(vector)[0]

        if pred == 1:
            resultado = "😊 POSITIVA"

        else:
            resultado = "☹️ NEGATIVA"
    return render_template(
        "index.html",
        resultado=resultado
)
if __name__ == "__main__":
    app.run(debug=True)
