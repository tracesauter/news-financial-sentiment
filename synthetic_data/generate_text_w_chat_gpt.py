from openai import OpenAI
import os
from openai import OpenAI, OpenAIError
from PyPDF2 import PdfReader

def load_api_key(file_path="api_key.txt"):
    with open(file_path, "r") as file:
        # Strip to remove any whitespace or newlines
        api_key = file.read().strip()
    return api_key

api_key = load_api_key()
client = OpenAI(api_key=api_key)

manual_example_1 = r"""
% - BAD EXAMPLE: Not sufficiently related to finance, economics, or the business environmen
**headline**:
Miley Cyrus puts on great concert in Las Vegas
"""

manual_example_2 = r"""
% - GOOD EXAMPLE:
**headline**:
Intel's Q2 Should Be A Game Changer (Earnings Preview)
"""

manual_example_3 = r"""
% - GOOD EXAMPLE:
**headline**:
JPMorgan Predicts 2008 Will Be "Nothing But Net"
"""

manual_example_3 = r"""
% - GOOD EXAMPLE:
**headline**:
U.S. Stocks Higher After Economic Data, Monsanto Outlook
"""

manual_example_4 = r"""
% - GOOD EXAMPLE:
**headline**:
U.S. Stocks Sink; Dow Off More Than 180 Points
"""

manual_example_5 = r"""
% - GOOD EXAMPLE:
**headline**:
U.S. Stocks Drop As AIG Discloses Doubts About Portfolio
"""

try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": (
                "You are an expert in personal finance and investment finance who keeps in touch and up-to-date with the latest news. Your "
                "task is to generate headlines for news articles that are related to finance, economics, or the business environment. These headlines will "
                "form the basis of a synthetic dataset for training a machine learning model that will be used to measure sentiment from headlines. You "
                "have seen examples labeled BAD EXAMPLE and GOOD EXAMPLE. You should avoid headlines like the one in the bad example that don't relate "
                "to finance, economics, or the business environment. Instead, you should generate headlines that are similar to the good examples, although "
                "you have license to be create and generate your own headlines, so long as they relate to the US financial system, business environment, stock market, "
                "economy, or similar topics."
            )},
            {"role": "assistant", "content": manual_example_1},
            {"role": "assistant", "content": manual_example_2},
            {"role": "assistant", "content": manual_example_3},
            {"role": "assistant", "content": manual_example_4},
            {"role": "assistant", "content": manual_example_5},
            {"role": "user", "content": (
                r"""
Hello, I'm taking a course on the theory of machine learning and I will be building a model that measures sentiment from news headlines.

I would like you to help me build a synthetic dataset of news headlines related to finance, economics, or the business environment so that I can
pre-train my model to begin forming generalized representations of data relating to patterns in financial news headlines. I would like you to generate 150
headlines as a proof of concept formatted as a delimited file where the first column is a numeric id (starting with 1 and counting up by headline) and the second column is the headline
and the delimiter is the pipe character (|).

Thank you!
"""
            )}
        ],
        temperature=0.7  # for creativity and variety
    )

    latex_output = response.choices[0].message.content

    with open("generated_data/generated_headlines_81.txt", "w") as f:
        f.write(latex_output)

    print("Generated headlines saved to generated_data/generated_headlines_81.txt")
except OpenAIError as e:
    print(f"An error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")