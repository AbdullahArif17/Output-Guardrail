import os
from agents import Agent, GuardrailFunctionOutput, OpenAIChatCompletionsModel, RunContextWrapper, Runner, input_guardrail, output_guardrail, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

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

class SafetyCheck(BaseModel):
    safe: bool
    reason: str

GuardrailAgent = Agent(
    name="GuardrailAgent",
    instructions="Classify input or output as safe or unsafe. Block political topics or references to political figures.",
    output_type=SafetyCheck,
    model=model
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
    result = await Runner.run(GuardrailAgent, output.response, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.safe,
    )
