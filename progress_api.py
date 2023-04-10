from flask import Flask, jsonify
from opendota_fetcher import get_current_progress

app = Flask(__name__)

@app.route('/progress', methods=['GET'])
def get_progress():
    progress = get_current_progress()
    return jsonify({'progress': progress})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
