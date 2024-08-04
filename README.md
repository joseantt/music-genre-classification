# README

## Audio Genre Classification Web App

This Django-based web application allows users to upload an audio file, processes it, and predicts the genre of the music. The genres include Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, and Rock. The application uses a pre-trained Keras model to classify the genre of the audio. The model was trained on the GTZAN dataset and achieved a test accuracy of 69%.

### Features
- **Upload Audio File**: Users can upload an audio file in the application.
- **Audio Processing**: The application processes the audio file to extract a mel-spectrogram.
- **Genre Prediction**: Uses a pre-trained Keras model to predict the genre of the music.
- **Top 3 Predictions**: Displays the top 3 predicted genres for the uploaded audio file.

### Technologies Used
- **Django**: Web framework for building the application.
- **Librosa**: Library for audio and music analysis.
- **Matplotlib**: Library for creating visualizations.
- **Pillow**: Imaging library to handle image processing.
- **OpenCV**: Library for image processing.
- **Keras**: Deep learning framework for building and using the pre-trained model.

### How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/joseantt/music-genre-classification.git
   cd music-genre-classification
   ```

2. **Install Dependencies**
   Make sure you have Python and pip installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**
   ```bash
   python manage.py runserver
   ```

4. **Access the Application**
   Open your web browser and go to `http://127.0.0.1:8000/`.

### Template Files

- **home.html**: The home page template.
- **result.html**: Template to display the prediction results.

### Model

The model used for genre prediction is pre-trained and saved as `modelo_definitivo2.keras`. The model predicts the genre based on the mel-spectrogram of the uploaded audio file.