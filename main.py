from agents import Agent, InputGuardrailTripwireTriggered, OpenAIChatCompletionsModel, OutputGuardrailTripwireTriggered, Runner, set_tracing_disabled
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from guardrails import SafetyCheck, input_filter, output_filter

load_dotenv(override=True)
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_base_url = os.getenv("GEMINI_BASE_PATH")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME")


gemini_client = AsyncOpenAI(
	api_key=gemini_api_key, base_url=gemini_base_url
)

model = OpenAIChatCompletionsModel(
	openai_client=gemini_client,
	model=gemini_model_name,
)	

agent:Agent = Agent( 
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[input_filter],
    output_guardrails=[output_filter],
    output_type=str,
    model=model
)


def run_checks():
    user_queries = [
        "What is 2 + 2?",
        "Tell me about the president of the United States.",
        "Explain multiplication.",
        "Discuss the upcoming election."
    ]

    for query in user_queries:
        print(f"\nUser: {query}")
        try:
            result = Runner.run_sync(Agent, input=query)
            print("Guardrail Passed – Agent can respond normally.")
        except InputGuardrailTripwireTriggered:
            print("Guardrail: Input blocked due to policy.")
        except OutputGuardrailTripwireTriggered:
            print("Guardrail: Output blocked due to policy.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print(result.final_output)
if __name__ == "__main__":
    run_checks()