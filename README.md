# AI-Powered-Chatbot-for-College-queries

An intelligent chatbot built with Flask and TensorFlow for answering college-related queries. This project demonstrates NLP-based intent classification using TF-IDF vectorization and deep learning.

## Features

- 🤖 **AI-Powered**: Uses TF-IDF + Keras neural network for intent classification
- 💬 **Conversational**: Natural language understanding with 11 different intent categories
- 🎨 **Modern UI**: Beautiful, responsive chat interface with typing animations
- 📊 **Chat Logs**: SQLite database for storing conversation history
- ⚡ **Real-time**: Instant responses with typing indicators
- 📱 **Responsive**: Works seamlessly on desktop and mobile devices

## Tech Stack

- **Backend**: Flask (Python)
- **ML/NLP**: TensorFlow, Keras, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **Vectorization**: TF-IDF

## Project Structure

```
college_chatbot/
├── app.py                 # Flask backend application
├── training.py            # Model training script
├── intents.json          # Intent patterns and responses
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Chat interface
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
├── models/               # Generated after training
│   ├── chat_model.h5    # Trained model
│   ├── vectorizer.pkl   # TF-IDF vectorizer
│   ├── tag_mappings.pkl # Intent tag mappings
│   └── intents_data.pkl # Intents data
└── README.md            # This file
```

## Installation & Setup

### 1. Clone or Download the Project

```bash
cd chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

**Important**: You must train the model before running the application!

```bash
python training.py
```

This will:
- Load intents from `intents.json`
- Train a neural network model
- Save model files in the `models/` directory
- Display training accuracy

**Note**: Training may take 5-10 minutes depending on your hardware.

### 5. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5001`

Open your browser and navigate to the URL to start chatting!

## Usage

### Supported Intents

The chatbot can answer questions about:

1. **Greetings**: Hello, Hi, Hey
2. **Admissions**: Admission process, requirements, eligibility
3. **Courses**: Available programs, departments
4. **Fees**: Fee structure, scholarships, payment
5. **Facilities**: Infrastructure, hostels, labs, library
6. **Placements**: Placement assistance, companies, records
7. **Contact**: Phone, email, address, location
8. **Academic Calendar**: Semester schedules, exam dates
9. **Faculty**: Professors, teaching staff
10. **Thanks**: Thank you responses
11. **Goodbye**: Farewell messages

### Example Queries

- "What courses are offered?"
- "Tell me about the admission process"
- "What are the fees?"
- "Which companies visit for placements?"
- "Do you have hostel facilities?"
- "What is the contact information?"

## Model Architecture

- **Input**: TF-IDF vectors (max 1000 features, unigrams + bigrams)
- **Architecture**: Fully connected neural network
  - Input Layer: 1000 dimensions
  - Hidden Layer 1: 128 neurons (ReLU)
  - Dropout: 0.5
  - Hidden Layer 2: 64 neurons (ReLU)
  - Dropout: 0.5
  - Output Layer: 11 neurons (Softmax) - one for each intent
- **Optimizer**: Adam
- **Loss**: Categorical Cross-entropy
- **Epochs**: 200

## Customization

### Adding New Intents

1. Edit `intents.json`
2. Add a new intent object with:
   - `tag`: Unique identifier
   - `patterns`: List of example queries
   - `responses`: List of possible responses
3. Retrain the model: `python training.py`
4. Restart the app: `python app.py`

### Modifying UI

- Edit `templates/index.html` for structure
- Edit `static/style.css` for styling
- Edit `static/script.js` for functionality

## Chat Logs

All conversations are automatically logged to `chat_logs.db` SQLite database with:
- User message
- Bot response
- Predicted intent
- Confidence score
- Timestamp

## API Endpoints

- `GET /`: Render chat interface
- `POST /chat`: Send message and get response
- `GET /health`: Check application health

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model files not found
```

**Solution**: Run `python training.py` first to generate model files.

### Low Confidence Scores

If the bot gives incorrect or low-confidence responses:
- Add more training patterns to `intents.json`
- Retrain the model
- Increase training epochs in `training.py`

### Port Already in Use

```
Address already in use
```

**Solution**: Change port in `app.py` (last line) from 5000 to another port.

## Future Enhancements

 Add more intents and responses
 Implement sentiment analysis
 Add multi-language support
 Integrate with college database APIs
 Add voice input/output
 Implement user authentication
 Create admin dashboard for logs
 Add context awareness in conversations

## License

This project is created for educational purposes (5th Semester College Project).

## Author

Developed as a medium-complexity AI chatbot project demonstrating:
- Flask web development
- NLP and ML with TensorFlow
- Frontend development
- Full-stack integration

---

**Note**: Make sure to train the model before running the application. Without trained model files, the app will not work!

