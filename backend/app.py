from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_stock import predict_stock_price
from query_rag import ask_question
from stock_news import stock_health_report

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing for frontend-backend interaction

# Route for stock price prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock_ticker = data.get("stock_ticker")
    days = int(data.get("days", 1))

    if not stock_ticker or days < 1:
        return jsonify({"error": "Invalid input"}), 400

    predictions = predict_stock_price(stock_ticker, days)
    return jsonify({"predictions": [float(p) for p in predictions]})

# Route for RAG-based question answering
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']

    answer = ask_question(question)
    return jsonify({"answer": answer})

@app.route('/check',methods=['GET'])
def check():
    data=request.get_json()
    return data

@app.route("/health_report", methods=["POST"])
def get_health_report():
    data = request.get_json()
    company = data.get("company")
    ticker = data.get("ticker")

    if not company or not ticker:
        return jsonify({"error": "Missing company or ticker"}), 400

    result = stock_health_report(company, ticker)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
