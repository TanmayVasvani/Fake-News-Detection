# Fake-News-Detection
# Fake News Detection Program

This repository contains a program to detect fake news using Python, TensorFlow, and Flask. The model is trained on the [Fake and Real News Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv), and it provides a simple API for predictions.

## Features
- Preprocesses text data to remove noise and irrelevant content.
- Uses TF-IDF vectorization for feature extraction.
- Implements a neural network for classification.
- Provides a Flask API for deploying the model.

---

## Getting Started

### Prerequisites
1. **Python 3.7+**
2. Install required dependencies using:
```python
   ```bash
````   pip install -r requirements.txt

## Dataset
The program uses the Fake and Real News Dataset. The dataset will be loaded automatically. If the URL is inaccessible, download the dataset manually and replace the path in app.py:
```python
url = "path/to/local/dataset.csv"

```
## Troubleshooting

## Dataset Issues
If the dataset URL is unavailable, download it manually and update the path in the code:
```python
`url = `path/to/local/dataset.csv`

```
## Port Conflicts
If the Flask app doesn’t start because port 5000 is in use, change the port:
```python
app.run(debug=True, port=5001)

```
## TensorFlow Compatibility
Ensure you have TensorFlow installed:
pip install tensorflow==2.10.0

## API Testing Tools
Use tools like Postman or curl to test the API.

## Model Details
Algorithm: Neural Network
Framework: TensorFlow
Training Data: Fake and Real News Dataset
Accuracy: ~85% (may vary based on dataset and epochs)

## Contributions
Contributions are welcome! If you encounter issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
