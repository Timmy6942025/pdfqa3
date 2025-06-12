# main.py
import os
from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings)

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    result = qa_chain({'question': question, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((question, answer))
    chat_history = chat_history[-5:]
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
