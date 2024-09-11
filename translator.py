import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import os

# Disable warnings about symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="Language Translator", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextArea>textarea {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Define available language codes
language_codes = {
    'English': 'en',
    'Hindi': 'hi',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Italian': 'it',
    'Dutch': 'nl',
    'Russian': 'ru',
    'Chinese': 'zh',
}

# Load the appropriate MarianMT model and tokenizer based on source and target languages
def load_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Translate the input text using the loaded model and tokenizer
def translate(text, model, tokenizer):
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return translated_text

# Main app function
def main():
    st.title("Language Translator")
    st.markdown("Translate text between different languages")

    # Layout: Input selection and fields
    with st.form(key='translator_form'):
        cols = st.columns(2)
        with cols[0]:
            src_lang = st.selectbox("Select source language:", list(language_codes.keys()), index=0)
        with cols[1]:
            tgt_lang = st.selectbox("Select target language:", list(language_codes.keys()), index=1)

        text = st.text_area("Enter text to translate:", height=150, placeholder="Type your text here...")

        submitted = st.form_submit_button("Translate")

    # Ensure source and target languages are different
    if src_lang == tgt_lang:
        st.warning("Source and target languages must be different.")
        return

    # Translation logic when the form is submitted
    if submitted:
        if text.strip() == "":
            st.error("Please enter some text to translate.")
        else:
            src_code = language_codes[src_lang]
            tgt_code = language_codes[tgt_lang]

            try:
                st.info(f"Translating from {src_lang} to {tgt_lang}...")
                model, tokenizer = load_model(src_code, tgt_code)
                translated_text = translate(text, model, tokenizer)
                st.success(f"Translated Text ({tgt_lang}):")
                st.write(f"{translated_text}")
            except Exception as e:
                st.error(f"Error in translation: {str(e)}")

    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()
