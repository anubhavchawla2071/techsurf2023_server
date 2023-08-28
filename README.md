# Techsurf 2023
## Running the Flask Server

To run the Flask server and interact with the LSTM model, follow these steps:

1. **Place the LSTM Model and Tokenizer Files**:
   - Ensure you have the LSTM model pkl file and tokenizer ready.
   - Place both files in the `models` directory within this repository.

   For your model:
   - [Download model.pkl](https://drive.google.com/drive/folders/1Ah79uvYHBLQ0yzDc1JkFmQbjBpfUmcni?usp=sharing)
   - [Download tokenizer.pkl](https://drive.google.com/drive/folders/1Ah79uvYHBLQ0yzDc1JkFmQbjBpfUmcni?usp=sharing)

2. **Start the Flask Server**:
   - Open your terminal.
   - Navigate to the project directory using the following command:

     ```bash
     cd /path/to/your-flask-project
     ```

   - Start the Flask server by running:

     ```bash
     python app.py
     ```

   - The server will start running, and you'll see output indicating that the server is listening for requests. By default, it will be accessible at `http://localhost:5000`.

     **Note**: If you need to use a different port, you can modify the `app.py` file accordingly.

3. **Interact with the Server**:
   - Once the server is running, you can use the [TechSurf2023 Client](https://github.com/anubhavchawla2071/techsurf2023_client) to send POST requests to the server and generate text using the LSTM model or GPT-2.

   - The client provides an easy-to-use interface to input prompts and receive generated text from the server.

   - To use the client, follow the instructions provided in its repository.

## Important Notes

- Ensure that you have the required packages installed using the `requirements.txt` file.
- The server runs on port 5000 by default. Modify the port in the `app.py` file if necessary.
- Remember to place the LSTM model pkl file and tokenizer in the `models` directory before starting the server and changing the path to them in lstm.py file.

## License

This project is licensed under the [MIT License](LICENSE).
