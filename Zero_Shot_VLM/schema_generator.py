"""
Schema Generator — takes a task description, uses an LLM to produce
the object-state schema and process steps.
"""
 
import json
from backends.base import LLMBackend
from utils.prompts import SCHEMA_SYSTEM_PROMPT, SCHEMA_USER_PROMPT
 
 
def generate_schema(llm: LLMBackend, task_description: str) -> dict:
    prompt = SCHEMA_USER_PROMPT.format(task_description=task_description)
    raw_response = llm.generate(prompt, system_prompt=SCHEMA_SYSTEM_PROMPT)
 
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
 
    schema = json.loads(cleaned)
 
    assert "objects" in schema, "Schema missing 'objects' key"
    for obj_name, obj_data in schema["objects"].items():
        assert "states" in obj_data, f"Object '{obj_name}' missing 'states'"
        assert "initial_state" in obj_data, f"Object '{obj_name}' missing 'initial_state'"
        assert len(obj_data["states"]) >= 2, f"Object '{obj_name}' needs at least 2 states"
        assert obj_data["initial_state"] in obj_data["states"], \
            f"initial_state '{obj_data['initial_state']}' not in states for '{obj_name}'"
 
    if "process_steps" not in schema:
        schema["process_steps"] = []
 
    print(f"[Schema] Generated schema for task: {schema.get('task', task_description)}")
    if schema["process_steps"]:
        print(f"[Schema] Process steps:")
        for step in schema["process_steps"]:
            print(f"  {step}")
    for obj_name, obj_data in schema["objects"].items():
        print(f"  {obj_name}: {' -> '.join(obj_data['states'])}")
 
    return schema