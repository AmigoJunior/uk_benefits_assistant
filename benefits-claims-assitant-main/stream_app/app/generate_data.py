import time
import random
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from db import save_conversation, save_feedback, get_db_connection

# Set the timezone to CET (Europe/Berlin)
tz = ZoneInfo("Europe/London")

# List of sample questions and answers for benefits and claims application

SAMPLE_QUESTIONS = [
    "What is a deductible in insurance?",
    "How do I file a claim for medical expenses?",
    "What's the difference between in-network and out-of-network providers?",
    "What is a copayment?",
    "How long does it typically take to process a claim?",
]

SAMPLE_ANSWERS = [
    "A deductible is the amount you pay for covered health care services before your insurance plan starts to pay. For example, with a $2,000 deductible, you pay the first $2,000 of covered services yourself before insurance begins to pay.",
    "To file a claim for medical expenses, first gather all necessary documents such as medical bills and receipts. Then, obtain a claim form from your insurance provider, fill it out completely, and submit it along with the supporting documents either online through the insurer's portal or via mail to the address provided by your insurance company.",
    "In-network providers are healthcare providers who have contracted with your insurance company to provide services at pre-negotiated rates. Out-of-network providers don't have this agreement. Using in-network providers usually results in lower out-of-pocket costs for you, while out-of-network providers may lead to higher costs and may not be covered by your insurance plan.",
    "A copayment, often called a copay, is a fixed amount you pay for a covered health care service, usually when you receive the service. For example, you might pay $25 for a doctor visit or $10 for a prescription. The amount can vary by the type of covered health care service.",
    "The time to process a claim can vary depending on the complexity of the claim and the insurance provider. Typically, simple claims might be processed within a few days to a couple of weeks. More complex claims could take 30 to 60 days. If additional information is required or if there are discrepancies, the process may take longer. Always check with your specific insurance provider for their estimated processing times.",
]

SECTION = ["general claim benefits", "nhs claim benefits"]
MODELS = ["ollama/phi3", "openai/gpt-3.5-turbo", "openai/gpt-4o", "openai/gpt-4o-mini"]
RELEVANCE = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]


def generate_synthetic_data(start_time, end_time):
    current_time = start_time
    conversation_count = 0
    print(f"Starting historical data generation from {start_time} to {end_time}")
    while current_time < end_time:
        conversation_id = str(uuid.uuid4())
        question = random.choice(SAMPLE_QUESTIONS)
        answer = random.choice(SAMPLE_ANSWERS)
        section = random.choice(SECTION)
        model = random.choice(MODELS)
        relevance = random.choice(RELEVANCE)

        openai_cost = 0

        if model.startswith("openai/"):
            openai_cost = random.uniform(0.001, 0.1)

        answer_data = {
            "answer": answer,
            "response_time": random.uniform(0.5, 5.0),
            "relevance": relevance,
            "relevance_explanation": f"This answer is {relevance.lower()} to the question.",
            "model_used": model,
            "prompt_tokens": random.randint(50, 200),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(100, 500),
            "eval_prompt_tokens": random.randint(50, 150),
            "eval_completion_tokens": random.randint(20, 100),
            "eval_total_tokens": random.randint(70, 250),
            "openai_cost": openai_cost,
        }

        save_conversation(conversation_id, question, answer_data, section, current_time)
        print(
            f"Saved conversation: ID={conversation_id}, Time={current_time}, Section={section}, Model={model}"
        )

        if random.random() < 0.7:
            feedback = 1 if random.random() < 0.8 else -1
            save_feedback(conversation_id, feedback, current_time)
            print(
                f"Saved feedback for conversation {conversation_id}: {'Positive' if feedback > 0 else 'Negative'}"
            )

        current_time += timedelta(minutes=random.randint(1, 15))
        conversation_count += 1
        if conversation_count % 10 == 0:
            print(f"Generated {conversation_count} conversations so far...")

    print(
        f"Historical data generation complete. Total conversations: {conversation_count}"
    )


def generate_live_data():
    conversation_count = 0
    print("Starting live data generation...")
    while True:
        current_time = datetime.now(tz)
        # current_time = None
        conversation_id = str(uuid.uuid4())
        question = random.choice(SAMPLE_QUESTIONS)
        answer = random.choice(SAMPLE_ANSWERS)
        section = random.choice(SECTION)
        model = random.choice(MODELS)
        relevance = random.choice(RELEVANCE)

        openai_cost = 0

        if model.startswith("openai/"):
            openai_cost = random.uniform(0.001, 0.1)

        answer_data = {
            "answer": answer,
            "response_time": random.uniform(0.5, 5.0),
            "relevance": relevance,
            "relevance_explanation": f"This answer is {relevance.lower()} to the question.",
            "model_used": model,
            "prompt_tokens": random.randint(50, 200),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(100, 500),
            "eval_prompt_tokens": random.randint(50, 150),
            "eval_completion_tokens": random.randint(20, 100),
            "eval_total_tokens": random.randint(70, 250),
            "openai_cost": openai_cost,
        }

        save_conversation(conversation_id, question, answer_data, section, current_time)
        print(
            f"Saved live conversation: ID={conversation_id}, Time={current_time}, Section={section}, Model={model}"
        )

        if random.random() < 0.7:
            feedback = 1 if random.random() < 0.8 else -1
            save_feedback(conversation_id, feedback, current_time)
            print(
                f"Saved feedback for live conversation {conversation_id}: {'Positive' if feedback > 0 else 'Negative'}"
            )

        conversation_count += 1
        if conversation_count % 10 == 0:
            print(f"Generated {conversation_count} live conversations so far...")

        time.sleep(1)


if __name__ == "__main__":
    print(f"Script started at {datetime.now(tz)}")
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(hours=6)
    print(f"Generating historical data from {start_time} to {end_time}")
    generate_synthetic_data(start_time, end_time)
    print("Historical data generation complete.")

    print("Starting live data generation... Press Ctrl+C to stop.")
    try:
        generate_live_data()
    except KeyboardInterrupt:
        print(f"Live data generation stopped at {datetime.now(tz)}.")
    finally:
        print(f"Script ended at {datetime.now(tz)}")