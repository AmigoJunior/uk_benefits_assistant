import streamlit as st
import time
import uuid

from assistant import get_answer
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)


def print_log(message):
    print(message, flush=True)


def main():
    print_log("Starting the Benefit Claims application")
    st.title("Benefits & Claims Assistant")

    # Session state initialization
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        print_log(
            f"New conversation started with ID: {st.session_state.conversation_id}"
        )
    if "count" not in st.session_state:
        st.session_state.count = 0
        print_log("Feedback count initialized to 0")

    # Claims selection
    section = st.selectbox(
        "Select a claims type:",
        ["general claim benefits", "nhs claim benefits"],
    )
    print_log(f"User selected course: {section}")

    # Model selection
    model_choice = st.selectbox(
        "Select a model:",
        ["ollama/phi3", "openai/gpt-3.5-turbo", "openai/gpt-4o", "openai/gpt-4o-mini"],
    )
    print_log(f"User selected model: {model_choice}")

    # Search type selection
    search_type = st.radio("Select search type:", ["Text", "Vector"])
    print_log(f"User selected search type: {search_type}")

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        print_log(f"User asked: '{user_input}'")
    with st.spinner("Processing..."):
        print_log(f"Getting answer from assistant using {model_choice} model and {search_type} search")
        start_time = time.time()
        answer_data = get_answer(user_input, section, model_choice, search_type)
        end_time = time.time()
        print_log(f"Answer received in {end_time - start_time:.2f} seconds")
        st.success("Completed!")
        
        # Display the category (section) in the output
        # st.write(f"Category: {category}")
        st.write(answer_data["answer"])

        # Display monitoring information
        st.write(f"Response time: {answer_data['response_time']:.2f} seconds")
        st.write(f"Relevance: {answer_data['relevance']}")
        st.write(f"Model used: {answer_data['model_used']}")
        st.write(f"Total tokens: {answer_data['total_tokens']}")
        if answer_data["openai_cost"] > 0:
            st.write(f"OpenAI cost: ${answer_data['openai_cost']:.4f}")

        # Save conversation to database (including category)
        print_log("Saving conversation to database")
        save_conversation(st.session_state.conversation_id, user_input, answer_data, section)
        print_log("Conversation saved successfully")
        st.session_state.conversation_saved = True
    # Feedback buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+1"):
            if st.session_state.conversation_saved:
                st.session_state.count += 1
                print_log(
                    f"Positive feedback received. New count: {st.session_state.count}"
                )
                save_feedback(st.session_state.conversation_id, 1)
                print_log("Positive feedback saved to database")
            else:
                st.error("Please ask a question before giving feedback.")
    with col2:
        if st.button("-1"):
            if st.session_state.conversation_saved:
                st.session_state.count -= 1
                print_log(
                    f"Negative feedback received. New count: {st.session_state.count}"
                )
                save_feedback(st.session_state.conversation_id, -1)
                print_log("Negative feedback saved to database")
            else:
                st.error("Please ask a question before giving feedback.")

    st.write(f"Current count: {st.session_state.count}")

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

    # Display recent conversations
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox(
        "Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    )
    recent_conversations = get_recent_conversations(
        limit=5, relevance=relevance_filter if relevance_filter != "All" else None
    )
    for conv in recent_conversations:
        st.write(f"Q: {conv['question']}")
        st.write(f"A: {conv['answer']}")
        st.write(f"Relevance: {conv['relevance']}")
        st.write(f"Model: {conv['model_used']}")
        st.write("---")

    # Display feedback stats
    feedback_stats = get_feedback_stats()
    st.subheader("Feedback Statistics")
    st.write(f"Thumbs up: {feedback_stats['thumbs_up']}")
    st.write(f"Thumbs down: {feedback_stats['thumbs_down']}")


print_log("Streamlit app loop completed")


if __name__ == "__main__":
    print_log("Claims and Benefits application started")
    main()
