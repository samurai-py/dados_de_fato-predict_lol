from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf

# Carregue o modelo treinado
model = tf.keras.models.load_model('/opt/render/project/src/models/model.h5')

with open('/opt/render/project/src/models/model2.csv', 'rb') as csv_file:
    df_model = pd.read_csv(csv_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def process_payload():
    try:
        # Recebe o payload
        data = request.json

        # Cria o DataFrame inicial
        df = df_model.copy()
    
        # Processa os picks
        for pick in data['picks']:
            df.loc[df['cols'] == f"pick_{pick}", '5465313'] = True
        
        # Processa a position
        df.loc[df['cols'] == f"position_{data['positions']}", '5465313'] = True
        
        # Processa os bans
        for ban in data['bans']:
            df.loc[df['cols'] == f"ban_{ban}.0", '5465313'] = True
            
        print(df.columns)
        
        # Converta o DataFrame para o formato correto para a previsão
        df_for_prediction = df['5465313'].astype(float).values.reshape(1, -1)

        # Faz a previsão
        prediction = model.predict(df_for_prediction)

        output = str(prediction[0][0])
        print(output)
        result = {
            "output": output}

        # Retorne a previsão como JSON
        return jsonify(result)  # Use tolist() para garantir que o resultado seja serializável

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
