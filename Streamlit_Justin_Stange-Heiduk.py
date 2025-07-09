import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from tensorflow.keras.applications.vgg16 import preprocess_input
import openai

# Erlaubte Klassen:
ALLOWED_CLASSES = [
    "Abraham Grampa Simpson", "Bart Simpson", "Charles Montgomery Burns",
    "Chief Wiggum", "Homer Simpson", "Krusty The Clown", "Lisa Simpson",
    "Marge Simpson", "Milhouse Van Houten", "Moe Szyslak", "Ned Flanders",
    "Principal Skinner", "Sideshow Bob"
]

@st.cache_resource
def load_my_model():
    return load_model("best_model_trainiert_simpsons.h5")

# Funktionen für Bildverarbeitung
def resize_and_pad_image(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    old_size = image.size
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_img

# Funktionen um das Bild in das richtige Format zu bringen
def preprocess_image(image: Image.Image):
    image = resize_and_pad_image(image, target_size=(224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

# Funktion zur Klassifizierung des Bildes
def classify_image(model, image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    if preds.size == 0:
        return None, None
    class_idx = np.argmax(preds, axis=1)[0]
    probability = preds[0][class_idx]
    return class_idx, probability

# Funktion zur Kodierung des Bildes in Base64
def encode_image(image: Image.Image) -> str:
    """
    Konvertiert das Bild in Base64 für GPT-4o.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Hauptfunktion
def main():
    st.title("Simpsons Character Classifier & Chatbot")
    st.write("Lade ein Simpsons-Bild hoch, klassifiziere es und stelle Fragen dazu im Chatbot-Stil.")

    # Sidebar mit den erlaubten Klassen
    with st.sidebar.expander(" Zu klassifizierende Simpsons-Charaktere"):
        for character in ALLOWED_CLASSES:
            st.write(f"- {character}")

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
    
    # Entferne vorherige Klassifikation, wenn Bild gelöscht wurde
    if uploaded_file is None:
        st.session_state.pop("classification", None)
        st.session_state.pop("probability", None)

    # Bild anzeigen und Klassifikation durchführen
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        if st.button("Classify"):
            model = load_my_model()
            class_idx, probability = classify_image(model, image)

            if class_idx is not None:
                predicted_class = ALLOWED_CLASSES[class_idx]
                st.write(f"**Vorhersage:** {predicted_class}")
                st.write(f"**Wahrscheinlichkeit:** {probability * 100:.2f}%")
                st.session_state["classification"] = predicted_class
                st.session_state["probability"] = probability
            else:
                st.error("Fehler bei der Klassifikation. Bitte versuche es mit einem anderen Bild.") 

        st.markdown("---")

    # Chatbot-Interaktion nur anzeigen, wenn Bild hochgeladen wurde und eine Klassifikation existiert
    if "classification" in st.session_state and uploaded_file is not None:
        st.subheader("Chatbot: Frage zum Bild")

        # Benutzereingabe
        user_input = st.text_input("Deine Frage:", key="user_question")
        
        # Senden der Frage
        if st.button("Send"):
            if user_input:
                classification = st.session_state.get("classification", "Unbekannt")
                # Systemnachricht für den Chatbot
                system_message = (
                    "Du bist ein hilfreicher Assistent, der Fragen zu Simpsons-Bildern beantwortet. "
                    "Das KNN kann folgende Charaktere klassifizieren: " + ", ".join(ALLOWED_CLASSES) + "."
                )
                # Prompt: Klassifikationsergebnis, Nutzerfrage und das verkleinerte Bild als Base64-Daten
                prompt = f"Das Bild wurde vom KNN als '{classification}' klassifiziert. " \
                            f"Beantworte bitte die folgende Frage: {user_input}"
                
                # Bild in Base64 umwandeln
                image_data = encode_image(image)

                # OpenAI API-Schlüssel prüfen
                if "general" in st.secrets and "openai_key" in st.secrets["general"]:
                    client = openai.OpenAI(api_key=st.secrets["general"]["openai_key"])
                else:
                    st.error("API-Schlüssel für OpenAI nicht gefunden.")
                    return

                try:
                    # Anfrage an GPT-4o
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": [{"type": "text", "text": prompt}]},
                            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data}}]}
                        ]
                    )

                    # Antwort des Chatbots
                    answer = response.choices[0].message.content
                    st.write("**Antwort:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Fehler bei der Anfrage: {e}")

if __name__ == "__main__":
    main()
