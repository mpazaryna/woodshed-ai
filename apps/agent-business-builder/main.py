import logging

from openai import OpenAI

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("output.log", mode="w")],
)
logger = logging.getLogger(__name__)

client = OpenAI()


def get_completion(messages):
    logger.info("Sending request to OpenAI API")
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
    )
    logger.info("Received response from OpenAI API")
    return completion.choices[0].message.content


def clarity_agent(user_input):
    logger.info("Clarity Agent: Processing user input")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clarity agent. "
                "Your job is to ask questions to clarify the user's business needs."
            ),
        },
        {
            "role": "user",
            "content": f"Based on this input, ask 1 clarifying questions: {user_input}",
        },
    ]
    response = get_completion(messages)
    logger.info("Clarity Agent: Generated response")
    return response


def niche_agent(user_input):
    logger.info("Niche Agent: Processing user input")
    messages = [
        {
            "role": "system",
            "content": "You are a niche agent. Your job is to generate niche content and identify the ideal target avatar.",
        },
        {
            "role": "user",
            "content": (
                f"Based on this input, suggest a niche and describe the ideal target avatar: "
                f"{user_input}"
            ),
        },
    ]
    response = get_completion(messages)
    logger.info("Niche Agent: Generated response")
    return response


def action_agent(user_input):
    logger.info("Action Agent: Processing user input")
    messages = [
        {
            "role": "system",
            "content": "You are an action agent. Your job is to deliver precise, impactful, and actionable steps that the user can implement immediately to achieve their goals.",
        },
        {
            "role": "user",
            "content": f"Based on this input, provide 3 specific actions the user should take: {user_input}",
        },
    ]
    response = get_completion(messages)
    logger.info("Action Agent: Generated response")
    return response


def business_strategist(clarity_response, niche_response, action_response):
    logger.info("Business Strategist: Synthesizing information")
    messages = [
        {
            "role": "system",
            "content": "You are a business strategist. Your role is to integrate diverse insights and craft a comprehensive, strategic roadmap that aligns with the user's business objectives and drives sustainable growth.",
        },
        {
            "role": "user",
            "content": f"Synthesize the following information into a clear, concise business strategy:\n\nClarity: {clarity_response}\n\nNiche: {niche_response}\n\nActions: {action_response}",
        },
    ]
    response = get_completion(messages)
    logger.info("Business Strategist: Generated strategy")
    return response


def get_user_input(prompt):
    print(prompt)
    return input()


def main():
    logger.info("Starting Business Builder process")
    initial_prompt = "Tell me about your business idea or current business:"
    user_input = get_user_input(initial_prompt)
    logger.info(f"Received initial user input: {user_input}")

    # Process through each agent without additional input
    clarity_response = clarity_agent(user_input)
    print("\nClarity Agent Response:")
    print(clarity_response)
    logger.info(f"Clarity Agent Response: {clarity_response}")

    niche_response = niche_agent(user_input)
    print("\nNiche Agent Response:")
    print(niche_response)
    logger.info(f"Niche Agent Response: {niche_response}")

    action_response = action_agent(user_input)
    print("\nAction Agent Response:")
    print(action_response)
    logger.info(f"Action Agent Response: {action_response}")

    strategy = business_strategist(clarity_response, niche_response, action_response)
    print("\nBusiness Strategy:")
    print(strategy)
    logger.info(f"Business Strategy: {strategy}")

    logger.info("Business Builder process completed")


if __name__ == "__main__":
    main()
