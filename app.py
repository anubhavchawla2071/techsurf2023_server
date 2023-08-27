from flask import Flask,request,jsonify
from models.gpt2_large import generate_gpt2_large
from models.lstm import generate_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/lstm", methods=["POST"])
def generate_text_endpoint():
    if request.method == "POST":
        try:
            data = request.get_json()
            input_text = data.get("input_text")

            # Generate text using the function
            generated_text = generate_text(input_text)

            # Create the response JSON
            response_data = {"generated_text": generated_text}

            return jsonify(response_data)
        except Exception as e:
            # Log the error for debugging purposes
            print("Error generating text:", e)
            return jsonify({"error": "An error occurred during text generation"}), 500


@app.route("/gpt2",methods=["POST"])
def members():
    if request.method == "POST":
        try:
            data = request.get_json()
            input_text = data.get("input_text")
            # print(input_text)
            # Use the generate_paragraph function from text_generator.py
            generated_text = generate_gpt2_large(input_text)

            # Create the response JSON
            response_data = {"generated_text": generated_text}

            return jsonify(response_data)
        except Exception as e:
            # Log the error for debugging purposes
            print("Error processing JSON data:", e)
            return jsonify({"error": "Invalid JSON data"}), 400

if __name__=="__main__":
    app.run(debug=True)