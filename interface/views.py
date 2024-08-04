from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import cv2
from keras.models import load_model
from io import BytesIO

model = load_model('modelo_definitivo2.keras')
label_dict = {0: 'Blues', 1: 'Classical', 2: 'Country',
              3: 'Disco', 4: 'Hiphop', 5: 'Jazz',
              6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'}


# path: /
def home(request):
    return render(request, 'home.html')


# path: /upload
@csrf_exempt
def upload(request):
    if request.method == 'POST':
        if request.method == 'POST' and request.FILES.get('audio'):
            audio_file = request.FILES['audio']

        try:
            y, sr = librosa.load(audio_file, sr=44100)  # Load audio file from request
        except:
            return JsonResponse({'error': 'Error loading audio file'}, status=400)

        # Get the duration of the audio
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 30:
            return JsonResponse({'error': 'Audio is too short. Minimum duration is 30 seconds.'}, status=400)

        if duration > 30:
            y = y[:int(30 * sr)]  # Trim the audio to 30 seconds

        y, _ = librosa.effects.trim(y)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=11000)
        spect = librosa.amplitude_to_db(mel, ref=np.max)

        # Apply MinMaxScaler to the spectrogram
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm_spect = scaler.fit_transform(spect)

        # Create the plot
        fig, ax = plt.subplots(figsize=(4.48, 2.91))  # This ratio approximates 336x218 when saved
        librosa.display.specshow(norm_spect, cmap='magma', sr=sr, hop_length=512, fmax=11000, ax=ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot as an image in memory
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=75, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        img_resized = img.resize((336, 218), Image.LANCZOS)

        # Convert image to numpy array
        img_array = np.array(img_resized.convert('L'), dtype=np.float32)
        img_array = cv2.resize(img_array, (336, 218))
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_array, verbose=0)
        top3_indices = predictions[0].argsort()[-3:][::-1]
        top3_labels = [label_dict[index] for index in top3_indices]
        # Create a dictionary to store the labels and their corresponding probabilities
        result = {'first': top3_labels[0],
                  'second': top3_labels[1],
                  'third': top3_labels[2]}
        return render(request, 'result.html', result)
    else:
        return redirect('/')
