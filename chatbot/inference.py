# app.py
import streamlit as st
import torch
import time
from model import ChatBotModel
from utils import ConversationManager


st.set_page_config(
    page_title="ChatBot Contextuel NLP",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_model(model_type='gpt2', vocab_size=None):
    return ChatBotModel(model_type=model_type, vocab_size=vocab_size)


def initialize_session_state():
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    if 'model' not in st.session_state:
        with st.spinner('Chargement du modèle...'):
            st.session_state.model = load_model()

# -------------------
# Fonction principale
# -------------------
def main():
    st.title("🤖 ChatBot Contextuel NLP")
    st.markdown("""
    Ce chatbot utilise des modèles NLP avancés (Seq2Seq, Attention, GPT) 
    pour générer des réponses contextuelles basées sur les données DailyDialog.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Paramètres")
        max_length = st.slider("Longueur maximale des réponses", 50, 200, 100)
        temperature = st.slider("Température", 0.1, 1.0, 0.7)

        st.markdown("---")
        if st.button("Effacer l'historique"):
            st.session_state.conversation_manager.clear_history()
            st.experimental_rerun()

        st.markdown("### Modèles supportés:")
        st.write("- GPT-2 (par défaut)")
        st.write("- Seq2Seq avec Attention")
        st.write("- Transformers personnalisés")

    # Zone de chat principale
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Conversation")

        # Affichage de l'historique
        chat_container = st.container()
        with chat_container:
            for speaker, text in st.session_state.conversation_manager.conversation_history:
                if speaker == 'user':
                    with st.chat_message("user"):
                        st.write(text)
                else:
                    with st.chat_message("assistant"):
                        st.write(text)

        # Input utilisateur
        user_input = st.chat_input("Tapez votre message ici...")
        if user_input:
            # Afficher le message utilisateur
            with st.chat_message("user"):
                st.write(user_input)

            # Générer la réponse
            with st.chat_message("assistant"):
                with st.spinner("Le chatbot réfléchit..."):
                    context = st.session_state.conversation_manager.get_context()
                    response = st.session_state.model.generate_response(
                        context,
                        max_length=max_length,
                        temperature=temperature
                    )

                    # Affichage progressif
                    message_placeholder = st.empty()
                    full_response = ""
                    for word in response.split():
                        full_response += word + " "
                        time.sleep(0.03)
                        message_placeholder.write(full_response + "▌")
                    message_placeholder.write(full_response.strip())

            # Mettre à jour l'historique
            st.session_state.conversation_manager.add_interaction(user_input, full_response.strip())

    with col2:
        st.subheader("Informations")
        num_messages = len(st.session_state.conversation_manager.conversation_history)
        st.metric("Messages échangés", num_messages // 2)

        st.markdown("### Contexte actuel:")
        context_preview = st.session_state.conversation_manager.get_context()
        if context_preview:
            st.text_area("Contexte", context_preview, height=200, label_visibility="collapsed")
        else:
            st.info("Aucune conversation en cours")

        with st.expander("Détails techniques"):
            st.write("**Modèle:** GPT-2 / Seq2Seq")
            st.write("**Architecture:** Transformer")
            st.write("**Données:** DailyDialog ou custom")
            st.write("**Tokenization:** Byte-pair encoding ou custom tokenizer")


if __name__ == "__main__":
    initialize_session_state()
    main()
