import streamlit as st
import time
import uuid
from assistant import get_answer
from db import save_conversation, save_feedback, get_recent_conversations, get_feedback_stats

def main():
    st.title("Benefits & Claims Assistant")

    # Session state initialization
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "conversation_saved" not in st.session_state:
        st.session_state.conversation_saved = False
    if "count" not in st.session_state:
        st.session_state.count = 0

    # Claims selection
    section = st.selectbox("Select a claims type:", ["general claim benefits", "nhs claim benefits"])

    # Model selection
    model_choice = st.selectbox("Select a model:", ["ollama/phi3", "openai/gpt-3.5-turbo", "openai/gpt-4o", "openai/gpt-4o-mini"])

    # Search type selection
    search_type = st.radio("Select search type:", ["Text", "Vector"])

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        if user_input:  # Only process if user has input something
            with st.spinner("Processing..."):
                start_time = time.time()
                answer_data = get_answer(user_input, section, model_choice, search_type)
                end_time = time.time()

                st.write(answer_data["answer"])
                st.write(f"Response time: {answer_data['response_time']:.2f} seconds")
                st.write(f"Relevance: {answer_data['relevance']}")
                st.write(f"Model used: {answer_data['model_used']}")
                st.write(f"Total tokens: {answer_data['total_tokens']}")
                if answer_data["openai_cost"] > 0:
                    st.write(f"OpenAI cost: ${answer_data['openai_cost']:.4f}")

                # Save conversation to the database
                save_conversation(st.session_state.conversation_id, user_input, answer_data, section)
                st.session_state.conversation_saved = True  # Set flag to True once saved

    # Feedback buttons (now work on the saved conversation)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+1"):
            if st.session_state.conversation_saved:
                st.session_state.count += 1
                save_feedback(st.session_state.conversation_id, 1)
                st.success("Positive feedback saved")
            else:
                st.error("Please ask a question before giving feedback.")

    with col2:
        if st.button("-1"):
            if st.session_state.conversation_saved:
                st.session_state.count -= 1
                save_feedback(st.session_state.conversation_id, -1)
                st.success("Negative feedback saved")
            else:
                st.error("Please ask a question before giving feedback.")

    # Relevance buttons
    st.subheader("Relevance Feedback")
    col3, col4, col5 = st.columns(3)
    with col3:
        if st.button("Relevant"):
            if st.session_state.conversation_saved:
                save_feedback(st.session_state.conversation_id, "RELEVANT")
                st.success("Marked as relevant")
            else:
                st.error("Please ask a question first.")

    with col4:
        if st.button("Partly Relevant"):
            if st.session_state.conversation_saved:
                save_feedback(st.session_state.conversation_id, "PARTLY_RELEVANT")
                st.success("Marked as partly relevant")
            else:
                st.error("Please ask a question first.")

    with col5:
        if st.button("Non-Relevant"):
            if st.session_state.conversation_saved:
                save_feedback(st.session_state.conversation_id, "NON_RELEVANT")
                st.success("Marked as non-relevant")
            else:
                st.error("Please ask a question first.")

    # Recent conversations
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox("Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"])
    recent_conversations = get_recent_conversations(limit=5, relevance=relevance_filter if relevance_filter != "All" else None)
    for conv in recent_conversations:
        st.write(f"Q: {conv['question']}")
        st.write(f"A: {conv['answer']}")
        st.write(f"Relevance: {conv['relevance']}")
        st.write(f"Model: {conv['model_used']}")
        st.write("---")

    # Feedback statistics
    feedback_stats = get_feedback_stats()
    st.subheader("Feedback Statistics")
    st.write(f"Thumbs up: {feedback_stats['thumbs_up']}")
    st.write(f"Thumbs down: {feedback_stats['thumbs_down']}")


if __name__ == "__main__":
    main()
