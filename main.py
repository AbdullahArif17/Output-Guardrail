from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
)
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv(override=True)
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_base_url = os.getenv("GEMINI_BASE_PATH")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME")

client = AsyncOpenAI(api_key=gemini_api_key, base_url=gemini_base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model=gemini_model_name)

class SafetyCheck(BaseModel):
    safe: bool
    reason: str

GuardrailAgent = Agent(
    name="GuardrailAgent",
    instructions="Classify input or output as safe or unsafe. Block political topics or references to political figures.",
    output_type=SafetyCheck,
    model=model,
)

@input_guardrail
async def input_filter(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(GuardrailAgent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.safe,
    )

@output_guardrail
async def output_filter(ctx: RunContextWrapper[None], agent: Agent, output: str) -> GuardrailFunctionOutput:
    result = await Runner.run(GuardrailAgent, output, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.safe,
    )

agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[input_filter],
    output_guardrails=[output_filter],
    output_type=str,
    model=model,
)

def run_checks():
    user_queries = [
        "What is 2 + 2?",
        "Tell me about the president of the United States.",
        "Explain multiplication.",
        "Discuss the upcoming election.",
    ]

    for query in user_queries:
        print(f"\nUser: {query}")
        try:
            result = Runner.run_sync(agent, input=query)
            print("Guardrail Passed – Agent can respond normally.")
            print("Output:", result.final_output)
        except InputGuardrailTripwireTriggered:
            print("Guardrail: Input blocked due to policy.")
        except OutputGuardrailTripwireTriggered:
            print("Guardrail: Output blocked due to policy.")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_checks()
