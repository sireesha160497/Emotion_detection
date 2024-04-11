import tkinter as tk
from tkinter import filedialog
import requests
from sklearn.metrics import classification_report, accuracy_score
import time
import csv
from EmotionDetectionModel import EmotionDetectionModel


def predict_emotion():
    text = text_entry.get()
    true_emotion = true_emotion_var.get()  # Get the true emotion from the dropdown menu
    start_time = time.time()
    response = requests.post('http://127.0.0.1:5000/predict', json={'text': text})
    end_time = time.time()
    try:
        data = response.json()
        predicted_emotion = data.get('predicted_emotion')
        if predicted_emotion:
            emotions.append((text, true_emotion, predicted_emotion, end_time - start_time))
            update_results()
        else:
            result_label.config(text="Error: Unable to predict emotion")
    except ValueError:
        result_label.config(text="Error: Invalid response from server")


def upload_csv():
   # global model  # Ensure that the 'model' variable is accessed from the global scope

    # Open a file dialog to select the CSV file
    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if csv_file_path:
        # Read the CSV file and extract text and true emotion columns
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            if 'Text' in reader.fieldnames and 'True Emotion' in reader.fieldnames:
                data = [(row['Text'], row['True Emotion']) for row in reader]

                # Predict emotions for each row of data
                predicted_emotions = [requests.post('http://127.0.0.1:5000/predict', json={'text': text}).json().get('predicted_emotion') for text, _ in data]

                # Update the data with predicted emotions
                data_with_predictions = [(text, true_emotion, predicted_emotion) for
                                         (text, true_emotion), predicted_emotion in zip(data, predicted_emotions)]

                # Write the updated data with predicted emotions back to the CSV file
                with open(csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Text', 'True Emotion', 'Predicted Emotion'])  # Write header
                    writer.writerows(data_with_predictions)  # Write data with predicted emotions

                result_label.config(text="Predicted emotions added to the CSV file")
                # Calculate accuracy
                true_emotions = [true_emotion for _, true_emotion, _ in data_with_predictions]
                predicted_emotions = [predicted_emotion for _, _, predicted_emotion in data_with_predictions]
                accuracy = accuracy_score(true_emotions, predicted_emotions)
                accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
            else:
                result_label.config(text="Error: 'Text' or 'True Emotion' columns not found in the CSV file")
    else:
        result_label.config(text="Error: No file selected")


def update_results():
    result_text = ""
    for text, true_emotion, predicted_emotion, response_time in emotions:
        response_time_str = f"{response_time:.4f} seconds" if response_time is not None else "N/A"
        result_text += f"Text: {text}\tTrue Emotion: {true_emotion}\tPredicted Emotion: {predicted_emotion}\tResponse Time: {response_time_str}\n"
    result_label.config(text=result_text)


def calculate_performance():
    y_true = [true_emotion for _, true_emotion, _, _ in emotions]
    y_pred = [predicted_emotion for _, _, predicted_emotion, _ in emotions]
    report = classification_report(y_true, y_pred, zero_division='warn')
    result_label.config(text="Performance Metrics:\n" + report)


def save_to_csv():
    with open('emotions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for emotion in emotions:
            writer.writerow(emotion[:2])  # Write only the first three values of each tuple
    result_label.config(text="Saved inputs to emotions.csv")
    train_model()
    # Reload the GUI after training
    root.destroy()  # Destroy the current GUI window
    create_gui()  # Create a new GUI window


def train_model():
    global model
    model = EmotionDetectionModel('emotions.csv')
    model.train_model()
    result_label.config(text="Training model with updated dataset...")


def create_gui():
    global root, text_entry, true_emotion_var, emotions, result_label, accuracy_label  # Add accuracy_label to global variables

    root = tk.Tk()
    root.title("Emotion Detection GUI")

    label = tk.Label(root, text="Enter Text:")
    label.pack()

    text_entry = tk.Entry(root, width=40)
    text_entry.pack()

    true_emotion_label = tk.Label(root, text="Select True Emotion:")
    true_emotion_label.pack()

    true_emotion_var = tk.StringVar(root)
    true_emotion_var.set("Select Emotion")  # Default option
    true_emotion_options = ["sadness", "anger", "love", "surprise", "joy", "fear"]
    true_emotion_dropdown = tk.OptionMenu(root, true_emotion_var, *true_emotion_options)
    true_emotion_dropdown.pack()

    predict_button = tk.Button(root, text="Predict Emotion", command=predict_emotion)
    predict_button.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    analyze_button = tk.Button(root, text="Calculate Performance", command=calculate_performance)
    analyze_button.pack()

    save_button = tk.Button(root, text="Save Inputs to CSV", command=save_to_csv)
    save_button.pack()

    upload_button = tk.Button(root, text="Upload Test CSV", command=upload_csv)
    upload_button.pack()

    accuracy_label = tk.Label(root, text="")
    accuracy_label.pack()  # Add accuracy_label to the GUI layout

    emotions = []  # List to store user inputs, true emotions, predicted emotions, and response times
    root.mainloop()


# Create the GUI when the script is executed
create_gui()
