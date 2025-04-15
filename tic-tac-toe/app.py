"""
uv run flask --app app run --host=0.0.0.0 --port=8050
"""

from flask import Flask, jsonify

app = Flask(__name__)
books = [{'id': 1, 'title': 'Python Essentials', 'author': 'Jane Doe'}]

@app.route('/books', methods=['GET'])
def get_books():
    return jsonify({'books': books})
