## Import the necessary modules
import json
import re

import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result

## Name of the Qwen model served by the local Ollama instance
MODEL = "qwen3:8b"
## Path to the lost-and-found database
DATABASE_FILE = "found_items.json"
## Path where the match result is written
OUTPUT_FILE = "output/match_result.json"


## Build your prompt based on the description the user provides
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.
def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant. Your job is to help a user "
        "find a lost item by matching their description against the items "
        "available in the lost-and-found database.\n\n"
        "Rules you MUST follow:\n"
        "1. Use ONLY the items provided in the user message. Never invent or "
        "reference any item that is not in the provided list.\n"
        "2. Not all the details of an item must match for it to be a possible "
        "match. An item can still be a possible match if only some of its "
        "details (item type, color, location, or date) are consistent with the "
        "description.\n"
        "3. Return ONLY a JSON object with exactly the following structure:\n"
        '   {"matches": ["ITEM_ID"], "confidence": "LOW"}\n'
        '4. "matches" contains all the possible matches, as a list of item IDs.\n'
        '5. "confidence" measures how confident you are about the matches and '
        "must be exactly one of: LOW, MEDIUM, HIGH.\n"
        '6. If there is no match, return {"matches": [], "confidence": "LOW"}.\n'
        "7. Do NOT include any other text, explanation, or markdown. Output "
        "only the JSON object."
    )

    user_prompt = (
        f"Lost item description: {description}\n\n"
        "Available items in the lost-and-found database:\n"
        + json.dumps(available_items, indent=2)
    )

    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result.
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()

    # Strip optional markdown code fences (```json ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Locate the JSON object even if there is extra text around it.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if not isinstance(result["confidence"], str):
        return False
    if result["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        return False

    valid_ids = {item["id"] for item in available_items}
    if not all(isinstance(match_id, str) and match_id in valid_ids for match_id in result["matches"]):
        return False

    return True


## Logic to display the matches found by Qwen in a user-friendly format.
## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    print("MATCH RESULT")
    print("--------------------------------------------------")
    print(f"Confidence: {result['confidence']}")
    print()

    matches = result["matches"]
    if not matches:
        print("No matches found. Matches: []")
        print()
        return

    items_by_id = {item["id"]: item for item in available_items}
    print("Possible matches:")
    print()
    for match_id in matches:
        item = items_by_id.get(match_id)
        if item is None:
            continue
        print(f"ID: {item['id']}")
        print(f"Item: {item['item']}")
        print(f"Color: {item['color']}")
        print(f"Location: {item['location']}")
        print(f"Date found: {item['date']}")
        print()


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("==================================================")
    print()

    description = input("Describe the item you lost: ")
    print()
    print("Searching for possible matches...")
    print()

    items = load_items(DATABASE_FILE)
    available_items = get_unclaimed_items(items)

    system_prompt, user_prompt = build_prompt(description, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    # Fall back to "no match" if Qwen returned something we cannot validate.
    if not validate_result(result, available_items):
        result = {"matches": [], "confidence": "LOW"}

    display_matches(result, available_items)
    save_result(result, OUTPUT_FILE)
    print(f"Result saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
