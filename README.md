# Text Reversal API

A simple API that reverses text input. Built with Flask and optimized for AWS Elastic Beanstalk deployment.

## Setup and Usage

### Server Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

### Client Usage

1. Make sure the server is running first
2. In a new terminal window, run the client:
```bash
python client.py
```

3. Follow the interactive prompts to enter text and see it reversed

### Example Client Usage
```python
from client import reverse_text

# Reverse a single text
result = reverse_text("Hello World")
print(result['reversed_text'])  # Output: dlroW olleH

# Use a different server URL
result = reverse_text("Hello World", server_url="http://your-server-url:5000")
```

## API Endpoints

### Reverse Text
- **URL**: `/reverse`
- **Method**: `POST`
- **Body**:
```json
{
    "text": "Hello World"
}
```
- **Response**:
```json
{
    "original_text": "Hello World",
    "reversed_text": "dlroW olleH"
}
```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
```json
{
    "status": "healthy"
}
```

## AWS Elastic Beanstalk Deployment

1. Install the EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB application:
```bash
eb init -p python-3.8 your-app-name
```

3. Create an environment:
```bash
eb create your-environment-name
```

4. Deploy:
```bash
eb deploy
```

5. Open the application:
```bash
eb open
```

## Project Structure
```
.
├── app.py              # Server application
├── client.py           # Client application
├── requirements.txt    # Python dependencies
├── Procfile           # EB process file
└── README.md          # This file
```

## Requirements
- Python 3.8+
- Flask
- gunicorn (for production)
- python-dotenv
- requests (for client) 