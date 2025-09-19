from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import uvicorn

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Models


class ConnectionModel(BaseModel):
    from_word: str
    to_word: str
    from_idx: int
    to_idx: int


class TrainingModel(BaseModel):
    container_id: int
    connections: List[ConnectionModel]

# Simple Neural Network for Word Alignment


class AlignmentNet(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, eng_ids, grk_ids):
        eng_emb = self.embedding(eng_ids)
        grk_emb = self.embedding(grk_ids)
        combined = torch.cat([eng_emb, grk_emb], dim=1)
        return self.fc(combined)

# Data Manager


class AlignmentDataManager:
    def __init__(self, data_path="C:/Users/dcurl/Desktop/Input"):
        self.data_path = Path(data_path)
        self.english_verses = []
        self.greek_verses = []
        self.vocab = {}
        self.vocab_counter = 0
        self.model = None
        self.optimizer = None
        self.training_data = []

        self.load_data()
        self.init_model()

    def load_data(self):
        with open(self.data_path / "english.txt", 'r', encoding='utf-8') as f:
            self.english_verses = [line.strip() for line in f.readlines()]

        with open(self.data_path / "greek.txt", 'r', encoding='utf-8') as f:
            self.greek_verses = [line.strip() for line in f.readlines()]

        print(
            f"Loaded {len(self.english_verses)} English and {len(self.greek_verses)} Greek verses")

    def get_word_id(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.vocab_counter
            self.vocab_counter += 1
        return self.vocab[word]

    def init_model(self):
        self.model = AlignmentNet(vocab_size=5000).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        print("Model initialized")

    def get_container(self, container_id: int):
        if container_id >= len(self.english_verses) or container_id >= len(self.greek_verses):
            return None

        english_words = self.english_verses[container_id].split()
        greek_words = self.greek_verses[container_id].split()

        return {
            'container_id': container_id,
            'english_verse': self.english_verses[container_id],
            'greek_verse': self.greek_verses[container_id],
            'english_words': english_words,
            'greek_words': greek_words
        }

    def add_training_data(self, container_id: int, connections: List[Dict]):
        container = self.get_container(container_id)
        if not container:
            return False

        training_example = {
            'container_id': container_id,
            'english_words': container['english_words'],
            'greek_words': container['greek_words'],
            'connections': connections
        }

        self.training_data.append(training_example)
        return True

    def train_model(self, epochs=20):
        if not self.training_data:
            return False

        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            for example in self.training_data:
                eng_words = example['english_words']
                grk_words = example['greek_words']
                connections = example['connections']

                # Create positive examples
                pos_eng_ids = []
                pos_grk_ids = []
                for conn in connections:
                    eng_id = self.get_word_id(eng_words[conn['from_idx']])
                    grk_id = self.get_word_id(grk_words[conn['to_idx']])
                    pos_eng_ids.append(eng_id)
                    pos_grk_ids.append(grk_id)

                if not pos_eng_ids:
                    continue

                # Create negative examples
                neg_eng_ids = []
                neg_grk_ids = []
                import random
                for _ in range(len(pos_eng_ids) * 2):
                    eng_idx = random.randint(0, len(eng_words)-1)
                    grk_idx = random.randint(0, len(grk_words)-1)
                    if not any(c['from_idx'] == eng_idx and c['to_idx'] == grk_idx for c in connections):
                        neg_eng_ids.append(
                            self.get_word_id(eng_words[eng_idx]))
                        neg_grk_ids.append(
                            self.get_word_id(grk_words[grk_idx]))

                if not neg_eng_ids:
                    continue

                # Training batch
                all_eng_ids = torch.tensor(
                    pos_eng_ids + neg_eng_ids, device=device)
                all_grk_ids = torch.tensor(
                    pos_grk_ids + neg_grk_ids, device=device)
                labels = torch.tensor(
                    [1.0] * len(pos_eng_ids) + [0.0] * len(neg_eng_ids), device=device)

                # Forward pass
                predictions = self.model(all_eng_ids, all_grk_ids).squeeze()
                loss = F.binary_cross_entropy(predictions, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            total_loss += epoch_loss
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        return True

    def predict_alignments(self, container_id: int, top_k=10):
        container = self.get_container(container_id)
        if not container:
            return []

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i, eng_word in enumerate(container['english_words']):
                for j, grk_word in enumerate(container['greek_words']):
                    eng_id = self.get_word_id(eng_word)
                    grk_id = self.get_word_id(grk_word)

                    eng_tensor = torch.tensor([eng_id], device=device)
                    grk_tensor = torch.tensor([grk_id], device=device)

                    prob = self.model(eng_tensor, grk_tensor).item()

                    predictions.append({
                        'from_idx': i,
                        'to_idx': j,
                        'from_word': eng_word,
                        'to_word': grk_word,
                        'confidence': prob
                    })

        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:top_k]


# Initialize the system
alignment_system = AlignmentDataManager()

# FastAPI app
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Word Alignment System", "game": "/game"}


@app.get("/api/container/{container_id}")
def get_container(container_id: int):
    container = alignment_system.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    return container


@app.post("/api/train")
def train_model(training_data: TrainingModel):
    connections = [conn.dict() for conn in training_data.connections]
    success = alignment_system.add_training_data(
        training_data.container_id, connections)

    if not success:
        raise HTTPException(status_code=400, detail="Invalid container")

    training_success = alignment_system.train_model()

    return {
        "success": training_success,
        "message": "Model trained successfully",
        "training_examples": len(alignment_system.training_data)
    }


@app.get("/api/predictions/{container_id}")
def get_predictions(container_id: int):
    predictions = alignment_system.predict_alignments(container_id)
    return {"predictions": predictions}


@app.get("/game")
def serve_game():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Word Alignment Game</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary { background: #667eea; color: white; }
        .btn-success { background: #56ab2f; color: white; }
        .btn-warning { background: #f093fb; color: white; }
        .btn-info { background: #4ecdc4; color: white; }
        
        button:hover { transform: translateY(-2px); }
        
        .verse-display {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .verse {
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .game-area {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .word-list {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 15px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .word-item {
            background: rgba(255,255,255,0.1);
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .word-item:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.02);
        }
        
        .word-item.selected {
            border: 2px solid #ffd700;
            background: rgba(255,215,0,0.3);
        }
        
        .word-item.connected {
            border: 2px solid #00ff00;
            background: rgba(0,255,0,0.2);
        }
        
        .connections-panel {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .connection-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .connection-item {
            background: rgba(255,255,255,0.1);
            padding: 8px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .remove-btn {
            background: #ff4757;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .stats {
            display: flex;
            justify-content: space-around;
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #ffd700;
        }
        
        .predictions {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            display: none;
        }
        
        .prediction-item {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            cursor: pointer;
        }
        
        .confidence {
            background: rgba(255,215,0,0.3);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        
        #container-input {
            padding: 8px;
            border: none;
            border-radius: 5px;
            width: 80px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Active Learning Word Alignment</h1>
            <p>Connect English and Greek words to train the AI!</p>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="container-id">0</div>
                <div>Container</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="connections-count">0</div>
                <div>Connections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="training-count">0</div>
                <div>Trained</div>
            </div>
        </div>
        
        <div class="controls">
            <input type="number" id="container-input" value="0" min="0" max="7999">
            <button class="btn-primary" onclick="loadContainer()">📖 Load</button>
            <button class="btn-info" onclick="showPredictions()">🤖 Predictions</button>
            <button class="btn-success" onclick="trainModel()">🧠 Train</button>
            <button class="btn-warning" onclick="clearConnections()">🗑️ Clear</button>
        </div>
        
        <div class="verse-display">
            <div class="verse"><strong>English:</strong> <span id="english-text">Load a container first</span></div>
            <div class="verse"><strong>Greek:</strong> <span id="greek-text">Load a container first</span></div>
        </div>
        
        <div class="game-area">
            <div class="word-list">
                <h3>English Words</h3>
                <div id="english-words"></div>
            </div>
            
            <div class="connections-panel">
                <h3>Connections</h3>
                <div class="connection-list" id="connections"></div>
            </div>
            
            <div class="word-list">
                <h3>Greek Words</h3>
                <div id="greek-words"></div>
            </div>
        </div>
        
        <div class="predictions" id="predictions-panel">
            <h3>AI Predictions</h3>
            <div id="predictions-list"></div>
        </div>
    </div>

    <script>
        class WordAlignmentGame {
            constructor() {
                this.currentContainer = 0;
                this.connections = [];
                this.selectedWord = null;
                this.containerData = null;
                this.trainingCount = 0;
            }
            
            async loadContainer() {
                const containerId = parseInt(document.getElementById('container-input').value);
                this.currentContainer = containerId;
                
                try {
                    const response = await fetch(`/api/container/${containerId}`);
                    this.containerData = await response.json();
                    
                    document.getElementById('english-text').textContent = this.containerData.english_verse;
                    document.getElementById('greek-text').textContent = this.containerData.greek_verse;
                    
                    this.displayWords('english-words', this.containerData.english_words, 'eng');
                    this.displayWords('greek-words', this.containerData.greek_words, 'grk');
                    
                    this.clearConnections();
                    this.updateStats();
                    
                } catch (error) {
                    alert('Error loading container: ' + error.message);
                }
            }
            
            displayWords(containerId, words, prefix) {
                const container = document.getElementById(containerId);
                container.innerHTML = '';
                
                words.forEach((word, index) => {
                    const div = document.createElement('div');
                    div.className = 'word-item';
                    div.textContent = `${index}: ${word}`;
                    div.onclick = () => this.selectWord(prefix, index, word);
                    div.id = `${prefix}-${index}`;
                    container.appendChild(div);
                });
            }
            
            selectWord(side, index, word) {
                // Clear previous selection
                if (this.selectedWord) {
                    document.getElementById(`${this.selectedWord.side}-${this.selectedWord.index}`)
                        .classList.remove('selected');
                }
                
                // If selecting from opposite side, create connection
                if (this.selectedWord && this.selectedWord.side !== side) {
                    this.createConnection(this.selectedWord, {side, index, word});
                    this.selectedWord = null;
                    return;
                }
                
                // Select this word
                this.selectedWord = {side, index, word};
                document.getElementById(`${side}-${index}`).classList.add('selected');
            }
            
            createConnection(word1, word2) {
                let engWord, grkWord;
                if (word1.side === 'eng') {
                    engWord = word1;
                    grkWord = word2;
                } else {
                    engWord = word2;
                    grkWord = word1;
                }
                
                // Check if connection exists
                const exists = this.connections.some(conn => 
                    conn.from_idx === engWord.index && conn.to_idx === grkWord.index);
                if (exists) return;
                
                // Add connection
                this.connections.push({
                    from_idx: engWord.index,
                    to_idx: grkWord.index,
                    from_word: engWord.word,
                    to_word: grkWord.word
                });
                
                // Update visuals
                document.getElementById(`eng-${engWord.index}`).classList.add('connected');
                document.getElementById(`grk-${grkWord.index}`).classList.add('connected');
                
                this.updateConnections();
                this.updateStats();
            }
            
            updateConnections() {
                const container = document.getElementById('connections');
                container.innerHTML = '';
                
                if (this.connections.length === 0) {
                    container.innerHTML = '<div style="color: #ccc;">No connections</div>';
                    return;
                }
                
                this.connections.forEach((conn, i) => {
                    const div = document.createElement('div');
                    div.className = 'connection-item';
                    div.innerHTML = `
                        <span>${conn.from_word} ↔ ${conn.to_word}</span>
                        <button class="remove-btn" onclick="game.removeConnection(${i})">×</button>
                    `;
                    container.appendChild(div);
                });
            }
            
            removeConnection(index) {
                const conn = this.connections[index];
                document.getElementById(`eng-${conn.from_idx}`).classList.remove('connected');
                document.getElementById(`grk-${conn.to_idx}`).classList.remove('connected');
                this.connections.splice(index, 1);
                this.updateConnections();
                this.updateStats();
            }
            
            clearConnections() {
                this.connections.forEach(conn => {
                    const engEl = document.getElementById(`eng-${conn.from_idx}`);
                    const grkEl = document.getElementById(`grk-${conn.to_idx}`);
                    if (engEl) engEl.classList.remove('connected');
                    if (grkEl) grkEl.classList.remove('connected');
                });
                
                if (this.selectedWord) {
                    document.getElementById(`${this.selectedWord.side}-${this.selectedWord.index}`)
                        .classList.remove('selected');
                    this.selectedWord = null;
                }
                
                this.connections = [];
                this.updateConnections();
                this.updateStats();
            }
            
            async trainModel() {
                if (this.connections.length === 0) {
                    alert('Create some connections first!');
                    return;
                }
                
                try {
                    const response = await fetch('/api/train', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            container_id: this.currentContainer,
                            connections: this.connections
                        })
                    });
                    
                    const result = await response.json();
                    this.trainingCount = result.training_examples;
                    
                    alert(`Training successful! Total examples: ${this.trainingCount}`);
                    this.updateStats();
                    
                    // Load next container
                    setTimeout(() => {
                        document.getElementById('container-input').value = this.currentContainer + 1;
                        this.loadContainer();
                    }, 1000);
                    
                } catch (error) {
                    alert('Training failed: ' + error.message);
                }
            }
            
            async showPredictions() {
                try {
                    const response = await fetch(`/api/predictions/${this.currentContainer}`);
                    const data = await response.json();
                    
                    const panel = document.getElementById('predictions-panel');
                    const list = document.getElementById('predictions-list');
                    
                    list.innerHTML = '';
                    data.predictions.forEach(pred => {
                        const div = document.createElement('div');
                        div.className = 'prediction-item';
                        div.innerHTML = `
                            <span>${pred.from_word} ↔ ${pred.to_word}</span>
                            <span class="confidence">${(pred.confidence * 100).toFixed(1)}%</span>
                        `;
                        div.onclick = () => this.applyPrediction(pred);
                        list.appendChild(div);
                    });
                    
                    panel.style.display = 'block';
                    
                } catch (error) {
                    alert('Error getting predictions: ' + error.message);
                }
            }
            
            applyPrediction(pred) {
                const exists = this.connections.some(conn => 
                    conn.from_idx === pred.from_idx && conn.to_idx === pred.to_idx);
                if (exists) return;
                
                this.connections.push(pred);
                document.getElementById(`eng-${pred.from_idx}`).classList.add('connected');
                document.getElementById(`grk-${pred.to_idx}`).classList.add('connected');
                this.updateConnections();
                this.updateStats();
            }
            
            updateStats() {
                document.getElementById('container-id').textContent = this.currentContainer;
                document.getElementById('connections-count').textContent = this.connections.length;
                document.getElementById('training-count').textContent = this.trainingCount;
            }
        }
        
        const game = new WordAlignmentGame();
        
        function loadContainer() { game.loadContainer(); }
        function showPredictions() { game.showPredictions(); }
        function trainModel() { game.trainModel(); }
        function clearConnections() { game.clearConnections(); }
        
        // Load first container
        window.onload = () => game.loadContainer();
    </script>
</body>
</html>
    '''

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
