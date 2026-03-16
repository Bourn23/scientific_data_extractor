Warning: Gemini 3 Pro Preview is deprecated and has been shut down March 9, 2026. Migrate to Gemini 3.1 Pro Preview to avoid service disruption.
Gemini 3 is our most intelligent model family to date, built on a foundation of state-of-the-art reasoning. It is designed to bring any idea to life by mastering agentic workflows, autonomous coding, and complex multimodal tasks. This guide covers key features of the Gemini 3 model family and how to get the most out of it.

Try Gemini 3.1 Pro Preview Try Gemini 3 Flash Try Nano Banana 2
Explore our collection of Gemini 3 apps to see how the model handles advanced reasoning, autonomous coding, and complex multimodal tasks.

Get started with a few lines of code:

Python
JavaScript
REST

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="Find the race condition in this multi-threaded C++ snippet: [code here]",
)

print(response.text)
Meet the Gemini 3 series
Gemini 3.1 Pro is best for complex tasks that require broad world knowledge and advanced reasoning across modalities.

Gemini 3 Flash is our latest 3-series model, with Pro-level intelligence at the speed and pricing of Flash.

Nano Banana Pro (also known as Gemini 3 Pro Image) is our highest quality image generation model, and Nano Banana 2 (also known as Gemini 3.1 Flash Image) is the high-volume, high-efficiency, lower price-point equivalent.

Gemini 3.1 Flash-Lite is our workhorse model built for cost-efficiency model and high-volume tasks.

All Gemini 3 models are currently in preview.

Model ID	Context Window (In / Out)	Knowledge Cutoff	Pricing (Input / Output)*
gemini-3.1-flash-lite-preview	1M / 64k	Jan 2025	$0.25 (text, image, video), $0.50 (audio) / $1.50
gemini-3.1-flash-image-preview	128k / 32k	Jan 2025	$0.25 (Text Input) / $0.067 (Image Output)**
gemini-3.1-pro-preview	1M / 64k	Jan 2025	$2 / $12 (<200k tokens)
$4 / $18 (>200k tokens)
gemini-3-flash-preview	1M / 64k	Jan 2025	$0.50 / $3
gemini-3-pro-image-preview	65k / 32k	Jan 2025	$2 (Text Input) / $0.134 (Image Output)**
* Pricing is per 1 million tokens unless otherwise noted. ** Image pricing varies by resolution. See the pricing page for details.

For detailed limits, pricing, and additional information, see the models page.

New API features in Gemini 3
Gemini 3 introduces new parameters designed to give developers more control over latency, cost, and multimodal fidelity.

Thinking level
Gemini 3 series models use dynamic thinking by default to reason through prompts. You can use the thinking_level parameter, which controls the maximum depth of the model's internal reasoning process before it produces a response. Gemini 3 treats these levels as relative allowances for thinking rather than strict token guarantees.

If thinking_level is not specified, Gemini 3 will default to high. For faster, lower-latency responses when complex reasoning isn't required, you can constrain the model's thinking level to low.

Thinking Level	Gemini 3.1 Pro	Gemini 3.1 Flash-Lite	Gemini 3 Flash	Description
minimal	Not supported	Supported (Default)	Supported	Matches the "no thinking" setting for most queries. The model may think very minimally for complex coding tasks. Minimizes latency for chat or high throughput applications. Note, minimal does not guarantee that thinking is off.
low	Supported	Supported	Supported	Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput applications.
medium	Supported	Supported	Supported	Balanced thinking for most tasks.
high	Supported (Default, Dynamic)	Supported (Dynamic)	Supported (Default, Dynamic)	Maximizes reasoning depth. The model may take significantly longer to reach a first (non thinking) output token, but the output will be more carefully reasoned.
Python
JavaScript
REST

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="How does AI work?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low")
    ),
)

print(response.text)
Important: You cannot use both thinking_level and the legacy thinking_budget parameter in the same request. Doing so will return a 400 error.
Media resolution
Gemini 3 introduces granular control over multimodal vision processing via the media_resolution parameter. Higher resolutions improve the model's ability to read fine text or identify small details, but increase token usage and latency. The media_resolution parameter determines the maximum number of tokens allocated per input image or video frame.

You can now set the resolution to media_resolution_low, media_resolution_medium, media_resolution_high, or media_resolution_ultra_high per individual media part or globally (via generation_config, global not available for ultra high). If unspecified, the model uses optimal defaults based on the media type.

Recommended settings

Media Type	Recommended Setting	Max Tokens	Usage Guidance
Images	media_resolution_high	1120	Recommended for most image analysis tasks to ensure maximum quality.
PDFs	media_resolution_medium	560	Optimal for document understanding; quality typically saturates at medium. Increasing to high rarely improves OCR results for standard documents.
Video (General)	media_resolution_low (or media_resolution_medium)	70 (per frame)	Note: For video, low and medium settings are treated identically (70 tokens) to optimize context usage. This is sufficient for most action recognition and description tasks.
Video (Text-heavy)	media_resolution_high	280 (per frame)	Required only when the use case involves reading dense text (OCR) or small details within video frames.
Note: The media_resolution parameter maps to different token counts depending on the input type. While images scale linearly (media_resolution_low: 280, media_resolution_medium: 560, media_resolution_high: 1120), Video is compressed more aggressively. For Video, both media_resolution_low and media_resolution_medium are capped at 70 tokens per frame, and media_resolution_high is capped at 280 tokens. See the full details on the Media resolution page.
Python
JavaScript
REST

from google import genai
from google.genai import types
import base64

# The media_resolution parameter is currently only available in the v1alpha API version.
client = genai.Client(http_options={'api_version': 'v1alpha'})

response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=[
        types.Content(
            parts=[
                types.Part(text="What is in this image?"),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=base64.b64decode("..."),
                    ),
                    media_resolution={"level": "media_resolution_high"}
                )
            ]
        )
    ]
)

print(response.text)
Temperature
For all Gemini 3 models, we strongly recommend keeping the temperature parameter at its default value of 1.0.

While previous models often benefited from tuning temperature to control creativity versus determinism, Gemini 3's reasoning capabilities are optimized for the default setting. Changing the temperature (setting it below 1.0) may lead to unexpected behavior, such as looping or degraded performance, particularly in complex mathematical or reasoning tasks.

Thought signatures
Gemini 3 uses Thought signatures to maintain reasoning context across API calls. These signatures are encrypted representations of the model's internal thought process. To ensure the model maintains its reasoning capabilities you must return these signatures back to the model in your request exactly as they were received:

Function Calling (Strict): The API enforces strict validation on the "Current Turn". Missing signatures will result in a 400 error.

Note: Circulation of thought signatures is required even when thinking level is set to minimal for Gemini 3 Flash.
Text/Chat: Validation is not strictly enforced, but omitting signatures will degrade the model's reasoning and answer quality.

Image generation/editing (Strict): The API enforces strict validation on all Model parts including a thoughtSignature. Missing signatures will result in a 400 error.

Success: If you use the official SDKs (Python, Node, Java) and standard chat history, Thought Signatures are handled automatically. You do not need to manually manage these fields.
Function calling (strict validation)
When Gemini generates a functionCall, it relies on the thoughtSignature to process the tool's output correctly in the next turn. The "Current Turn" includes all Model (functionCall) and User (functionResponse) steps that occurred since the last standard User text message.

Single Function Call: The functionCall part contains a signature. You must return it.
Parallel Function Calls: Only the first functionCall part in the list will contain the signature. You must return the parts in the exact order received.
Multi-Step (Sequential): If the model calls a tool, receives a result, and calls another tool (within the same turn), both function calls have signatures. You must return all accumulated signatures in the history.
Text and streaming
For standard chat or text generation, the presence of a signature is not guaranteed.

Non-Streaming: The final content part of the response may contain a thoughtSignature, though it is not always present. If one is returned, you should send it back to maintain best performance.
Streaming: If a signature is generated, it may arrive in a final chunk that contains an empty text part. Ensure your stream parser checks for signatures even if the text field is empty.
Image generation and editing
For gemini-3-pro-image-preview and gemini-3.1-flash-image-preview, thought signatures are critical for conversational editing. When you ask the model to modify an image it relies on the thoughtSignature from the previous turn to understand the composition and logic of the original image.

Editing: Signatures are guaranteed on the first part after the thoughts of the response (text or inlineData) and on every subsequent inlineData part. You must return all of these signatures to avoid errors.
Code examples
Multi-step Function Calling (Sequential)
Parallel Function Calling
Text/In-Context Reasoning (No Validation)
Image Generation & Editing
Migrating from other models
If you are transferring a conversation trace from another model (e.g., Gemini 2.5) or injecting a custom function call that was not generated by Gemini 3, you will not have a valid signature.

To bypass strict validation in these specific scenarios, populate the field with this specific dummy string: "thoughtSignature": "context_engineering_is_the_way to_go"

Structured Outputs with tools
Gemini 3 models allow you to combine Structured Outputs with built-in tools, including Grounding with Google Search, URL Context, Code Execution, and Function Calling.

Python
JavaScript
REST

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

class MatchResult(BaseModel):
    winner: str = Field(description="The name of the winner.")
    final_match_score: str = Field(description="The final match score.")
    scorers: List[str] = Field(description="The name of the scorer.")

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents="Search for all details for the latest Euro.",
    config={
        "tools": [
            {"google_search": {}},
            {"url_context": {}}
        ],
        "response_mime_type": "application/json",
        "response_json_schema": MatchResult.model_json_schema(),
    },  
)

result = MatchResult.model_validate_json(response.text)
print(result)
Image generation
Gemini 3.1 Flash Image and Gemini 3 Pro Image let you generate and edit images from text prompts. It uses reasoning to "think" through a prompt and can retrieve real-time data—such as weather forecasts or stock charts—before using Google Search grounding before generating high-fidelity images.

New & improved capabilities:

4K & text rendering: Generate sharp, legible text and diagrams with up to 2K and 4K resolutions.
Grounded generation: Use the google_search tool to verify facts and generate imagery based on real-world information. Grounding with Google Image Search available for Gemini 3.1 Flash Image.
Conversational editing: Multi-turn image editing by simply asking for changes (e.g., "Make the background a sunset"). This workflow relies on Thought Signatures to preserve visual context between turns.
For complete details on aspect ratios, editing workflows, and configuration options, see the Image Generation guide.

Python
JavaScript
REST

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="Generate an infographic of the current weather in Tokyo.",
    config=types.GenerateContentConfig(
        tools=[{"google_search": {}}],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="4K"
        )
    )
)

image_parts = [part for part in response.parts if part.inline_data]

if image_parts:
    image = image_parts[0].as_image()
    image.save('weather_tokyo.png')
    image.show()
Example Response

Weather Tokyo

Code Execution with images
Gemini 3 Flash can treat vision as an active investigation, not just a static glance. By combining reasoning with code execution, the model formulates a plan, then writes and executes Python code to zoom in, crop, annotate, or otherwise manipulate images step-by-step to visually ground its answers.

Use cases:

Zoom and inspect: The model implicitly detects when details are too small (e.g., reading a distant gauge or serial number) and writes code to crop and re-examine the area at higher resolution.
Visual math and plotting: The model can run multi-step calculations using code (e.g., summing line items on a receipt, or generating a Matplotlib chart from extracted data).
Image annotation: The model can draw arrows, bounding boxes, or other annotations directly onto images to answer spatial questions like "Where should this item go?".
To enable visual thinking, configure Code Execution as a tool. The model will automatically use code to manipulate images when needed.

Python
JavaScript
REST

from google import genai
from google.genai import types
import requests
from PIL import Image
import io

image_path = "https://goo.gle/instrument-img"
image_bytes = requests.get(image_path).content
image = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        image,
        "Zoom into the expression pedals and tell me how many pedals are there?"
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    ),
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    if part.executable_code is not None:
        print(part.executable_code.code)
    if part.code_execution_result is not None:
        print(part.code_execution_result.output)
    if part.as_image() is not None:
        display(Image.open(io.BytesIO(part.as_image().image_bytes)))
For more details on code execution with images, see Code Execution.

Multimodal function responses
Multimodal function calling allows users to have function responses containing multimodal objects allowing for improved utilization of function calling capabilities of the model. Standard function calling only supports text-based function responses:

Python
JavaScript
REST

from google import genai
from google.genai import types

import requests

client = genai.Client()

# This is a manual, two turn multimodal function calling workflow:

# 1. Define the function tool
get_image_declaration = types.FunctionDeclaration(
  name="get_image",
  description="Retrieves the image file reference for a specific order item.",
  parameters={
      "type": "object",
      "properties": {
          "item_name": {
              "type": "string",
              "description": "The name or description of the item ordered (e.g., 'instrument')."
          }
      },
      "required": ["item_name"],
  },
)
tool_config = types.Tool(function_declarations=[get_image_declaration])

# 2. Send a message that triggers the tool
prompt = "Show me the instrument I ordered last month."
response_1 = client.models.generate_content(
  model="gemini-3-flash-preview",
  contents=[prompt],
  config=types.GenerateContentConfig(
      tools=[tool_config],
  )
)

# 3. Handle the function call
function_call = response_1.function_calls[0]
requested_item = function_call.args["item_name"]
print(f"Model wants to call: {function_call.name}")

# Execute your tool (e.g., call an API)
# (This is a mock response for the example)
print(f"Calling external tool for: {requested_item}")

function_response_data = {
  "image_ref": {"$ref": "instrument.jpg"},
}
image_path = "https://goo.gle/instrument-img"
image_bytes = requests.get(image_path).content
function_response_multimodal_data = types.FunctionResponsePart(
  inline_data=types.FunctionResponseBlob(
    mime_type="image/jpeg",
    display_name="instrument.jpg",
    data=image_bytes,
  )
)

# 4. Send the tool's result back
# Append this turn's messages to history for a final response.
history = [
  types.Content(role="user", parts=[types.Part(text=prompt)]),
  response_1.candidates[0].content,
  types.Content(
    role="user",
    parts=[
        types.Part.from_function_response(
          name=function_call.name,
          response=function_response_data,
          parts=[function_response_multimodal_data]
        )
    ],
  )
]

response_2 = client.models.generate_content(
  model="gemini-3-flash-preview",
  contents=history,
  config=types.GenerateContentConfig(
      tools=[tool_config],
      thinking_config=types.ThinkingConfig(include_thoughts=True)
  ),
)

print(f"\nFinal model response: {response_2.text}")
Migrating from Gemini 2.5
Gemini 3 is our most capable model family to date and offers a stepwise improvement over Gemini 2.5. When migrating, consider the following:

Thinking: If you were previously using complex prompt engineering (like chain of thought) to force Gemini 2.5 to reason, try Gemini 3 with thinking_level: "high" and simplified prompts.
Temperature settings: If your existing code explicitly sets temperature (especially to low values for deterministic outputs), we recommend removing this parameter and using the Gemini 3 default of 1.0 to avoid potential looping issues or performance degradation on complex tasks.
PDF & document understanding: If you relied on specific behavior for dense document parsing, test the new media_resolution_high setting to ensure continued accuracy.
Token consumption: Migrating to Gemini 3 defaults may increase token usage for PDFs but decrease token usage for video. If requests now exceed the context window due to higher default resolutions, we recommend explicitly reducing the media resolution.
Image segmentation: Image segmentation capabilities (returning pixel-level masks for objects) are not supported in Gemini 3 Pro or Gemini 3 Flash. For workloads requiring native image segmentation, we recommend continuing to utilize Gemini 2.5 Flash with thinking turned off or Gemini Robotics-ER 1.5.
Computer Use: Gemini 3 Pro and Gemini 3 Flash support Computer Use. Unlike the 2.5 series, you don't need to use a separate model to access the Computer Use tool.
Tool support: Maps grounding is not yet supported for Gemini 3 models, so won't migrate. Additionally, combining built-in tools with function calling is not yet supported.
