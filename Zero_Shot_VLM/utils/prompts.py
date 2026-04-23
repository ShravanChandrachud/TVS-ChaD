"""
All LLM and VLM prompt templates.
Centralized here so they're easy to tune without touching pipeline code.
"""
 
# ──────────────────────────────────────────────────────────────
# SCHEMA GENERATION (LLM)
# ──────────────────────────────────────────────────────────────
 
SCHEMA_SYSTEM_PROMPT = """You are a visual state analysis expert. Given a task description, 
you identify all key objects involved and enumerate the visually distinct states each object 
can be in during that task. You also outline the high-level process steps. 
Focus on states that are VISUALLY distinguishable in a video — not abstract or internal states."""
 
SCHEMA_USER_PROMPT = """Task: "{task_description}"
 
Analyze this task and produce a JSON object with the following structure:
{{
    "task": "<task description>",
    "process_steps": [
        "Step 1: <description>",
        "Step 2: <description>",
        ...
    ],
    "objects": {{
        "<object_name>": {{
            "states": ["<state_1>", "<state_2>", ...],
            "initial_state": "<state_1>"
        }}
    }}
}}
 
Rules:
1. Only include objects that are VISUALLY present and change state during the task.
2. States must be VISUALLY distinct — a camera watching the task could tell them apart.
3. Order states chronologically (first state = how the object appears at the start).
4. State names should be short, descriptive visual labels (e.g., "whole_in_shell", "cracked_in_bowl").
5. Include 2-8 states per object. Don't over-segment — only states with clear visual difference.
6. initial_state is always the first state in the list.
7. Each object's states describe THAT OBJECT's own lifecycle — not the container it's in.
   For example, an egg's states are about the egg itself: whole_in_shell, cracked, beaten, cooking, folded, plated.
   A pan's states are about the pan: empty, oiled, has_contents, empty_with_residue.
   Do NOT put egg-specific states on the pan (like "with_liquid_eggs").
8. process_steps should be 4-10 high-level steps in chronological order describing the overall task.
9. States can only progress FORWARD in the list during normal operation. An object does not 
   go back to a previous state unless it is physically replaced (e.g., a bowl emptied and refilled).
 
Respond with ONLY the JSON object. No markdown, no explanation."""
 
 
# ──────────────────────────────────────────────────────────────
# VLM: FIRST FRAME INVENTORY
# ──────────────────────────────────────────────────────────────
 
VLM_INVENTORY_SYSTEM_PROMPT = """You are a precise visual scene analyzer for a cooking video. 
You examine a frame and identify every object instance visible, giving each a unique ID and 
spatial description so it can be re-identified in future frames."""
 
VLM_INVENTORY_PROMPT = """I am tracking objects in a video of: {task_description}
 
OBJECT CLASSES TO TRACK (with their possible states):
{object_classes_str}
 
PROCESS STEPS FOR THIS TASK:
{process_steps_str}
 
Look at this frame and identify every visible instance of these object classes.
 
For each instance:
1. instance_id: <class>_<number_or_descriptor> (e.g., "egg_1", "bowl_1", "pan_main")
2. description: spatial/visual description relative to other objects (e.g., "white bowl on left side of stove", "egg inside bowl_1")
3. current_state: from that class's state list
4. If there are MULTIPLE instances of the same class (e.g., 2 eggs, 2 bowls), give each a distinct ID.
 
Respond with ONLY this JSON:
{{
    "current_step": "<which process step we appear to be on>",
    "instances": [
        {{
            "class": "<object_class>",
            "instance_id": "<unique_id>",
            "description": "<position relative to other objects or scene landmarks>",
            "current_state": "<state_from_class_list>"
        }}
    ]
}}
 
Only include objects you can ACTUALLY SEE. Do not hallucinate.
Respond with ONLY the JSON. No markdown, no explanation."""
 
 
# ──────────────────────────────────────────────────────────────
# VLM: STATE VERIFICATION (called on each flagged frame)
# ──────────────────────────────────────────────────────────────
 
VLM_SYSTEM_PROMPT = """You are a precise visual state tracker for a cooking video. You compare 
two frames and determine what changed. Every object is tracked independently — an egg has its 
own lifecycle, a bowl has its own lifecycle, a pan has its own lifecycle. You use the position 
table to identify WHICH specific instance you're looking at."""
 
VLM_VERIFICATION_PROMPT = """I am tracking objects in a video of: {task_description}
 
PROCESS STEPS:
{process_steps_str}
CURRENT STEP: {current_step}
 
POSITION TABLE (where each object currently is):
{position_table_str}
 
MEMORY BANK (recent state history):
{recent_history_str}
 
POSSIBLE STATES PER OBJECT CLASS:
{possible_states_str}
 
IMAGE 1 (first): The anchor frame — last confirmed state change.
IMAGE 2 (second): The current frame — evaluate this.
 
TASK: Compare the two frames. Determine:
1. Has any object instance changed state? (Track each object independently — egg states are about 
   the egg, bowl states are about the bowl, etc.)
2. Has any object moved? If so, update its position description.
3. Has a NEW object appeared that wasn't in the position table? If so, register it.
   IMPORTANT: If you see an object in a location where a DIFFERENT object of the same class was 
   last tracked (e.g., a new egg being cracked into a bowl that previously held egg_1), this is 
   a NEW instance — give it a new ID (egg_2).
 
Rules:
- States only progress FORWARD for an object. An egg cannot go from "cooking" back to "beaten".
  If you see what looks like a backward transition, it's likely a NEW instance of that object.
- Match instances by their position descriptions. "egg_1 was in bowl_1" — if bowl_1 now has 
  a different-looking egg, that might be egg_2.
- Every object is independent. Don't describe egg states on the pan. The pan is "has_contents" 
  or "empty", the egg is "cooking" or "folded".
 
Respond with ONLY this JSON:
{{
    "changed": true/false,
    "current_step": "<updated process step>",
    "transitions": [
        {{
            "instance_id": "<instance_id>",
            "class": "<object_class>",
            "from_state": "<previous_state>",
            "to_state": "<new_state>",
            "confidence": "<high/medium/low>",
            "reason": "<brief visual evidence>"
        }}
    ],
    "position_updates": [
        {{
            "instance_id": "<instance_id>",
            "new_description": "<updated position/location>"
        }}
    ],
    "new_instances": [
        {{
            "class": "<object_class>",
            "instance_id": "<unique_id>",
            "description": "<position relative to other objects>",
            "current_state": "<initial_state>"
        }}
    ]
}}
 
If nothing changed:
{{
    "changed": false,
    "current_step": "{current_step}",
    "transitions": [],
    "position_updates": [],
    "new_instances": []
}}
 
Respond with ONLY the JSON. No markdown, no explanation."""