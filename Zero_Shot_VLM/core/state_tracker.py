"""
State Tracker — the main orchestrator.
Autoregressive: position table and memory bank grow as new objects appear.
Every object tracked independently. VLM gets full context each call.
"""
 
import json
 
from backends.base import EmbeddingBackend, VLMBackend
from core.change_detector import ChangeDetector
from core.memory_bank import MemoryBank
from utils.video import Frame
from utils.prompts import (
    VLM_INVENTORY_SYSTEM_PROMPT, VLM_INVENTORY_PROMPT,
    VLM_SYSTEM_PROMPT, VLM_VERIFICATION_PROMPT,
)
import config
 
 
class StateTracker:
 
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vlm_backend: VLMBackend,
        schema: dict,
        threshold: float = None,
    ):
        self.vlm = vlm_backend
        self.schema = schema
        self.change_detector = ChangeDetector(embedding_backend, threshold)
        self.memory_bank = MemoryBank(task=schema.get("task", "unknown"), schema=schema)
        self.anchor_frame: Frame | None = None
 
        self.frames_processed = 0
        self.changes_detected = 0
        self.vlm_calls = 0
        self.transitions_confirmed = 0
 
    def _build_object_classes_str(self) -> str:
        lines = []
        for obj_name, obj_data in self.schema["objects"].items():
            states_str = ", ".join(obj_data["states"])
            lines.append(f"  {obj_name}: [{states_str}]")
        return "\n".join(lines)
 
    def _build_process_steps_str(self) -> str:
        steps = self.schema.get("process_steps", [])
        return "\n".join(f"  {s}" for s in steps) if steps else "  (not available)"
 
    def _build_position_table_str(self) -> str:
        table = self.memory_bank.get_position_table()
        if not table:
            return "  (no objects registered yet)"
        lines = []
        for iid, info in table.items():
            lines.append(
                f"  {iid} ({info['class']}): state={info['state']}, "
                f"position=\"{info['description']}\""
            )
        return "\n".join(lines)
 
    def _build_recent_history_str(self) -> str:
        recent = self.memory_bank.get_recent_history(10)
        if not recent:
            return "  (no transitions yet)"
        lines = []
        for t in recent:
            mins = int(t.timestamp // 60)
            secs = int(t.timestamp % 60)
            lines.append(f"  {mins}:{secs:02d} | {t.instance_id}: {t.from_state} -> {t.to_state}")
        return "\n".join(lines)
 
    def _build_possible_states_str(self) -> str:
        lines = []
        for obj_name, obj_data in self.schema["objects"].items():
            states_str = ", ".join(obj_data["states"])
            lines.append(f"  {obj_name}: [{states_str}]")
        return "\n".join(lines)
 
    def _parse_json(self, raw: str) -> dict | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
 
    def _run_inventory(self, frame: Frame):
        print("[Tracker] Running scene inventory on first frame...")
        self.vlm_calls += 1
 
        prompt = VLM_INVENTORY_PROMPT.format(
            task_description=self.schema.get("task", ""),
            object_classes_str=self._build_object_classes_str(),
            process_steps_str=self._build_process_steps_str(),
        )
 
        for attempt in range(config.VLM_MAX_RETRIES + 1):
            raw = self.vlm.query(
                images=[frame.image],
                prompt=prompt,
                system_prompt=VLM_INVENTORY_SYSTEM_PROMPT,
            )
            result = self._parse_json(raw)
 
            if result and "instances" in result:
                self.memory_bank.current_step = result.get("current_step", self.memory_bank.current_step)
 
                for inst in result["instances"]:
                    obj_class = inst.get("class", "")
                    if obj_class not in self.schema["objects"]:
                        continue
                    state = inst.get("current_state", "")
                    if state not in self.schema["objects"][obj_class]["states"]:
                        state = self.schema["objects"][obj_class]["initial_state"]
 
                    self.memory_bank.register_instance(
                        instance_id=inst.get("instance_id", ""),
                        object_class=obj_class,
                        description=inst.get("description", ""),
                        initial_state=state,
                    )
 
                print(f"[Tracker] Inventory complete: {len(self.memory_bank.instances)} instances.\n")
                return
 
            if attempt < config.VLM_MAX_RETRIES:
                print(f"  [Inventory] Parse error (attempt {attempt + 1}), retrying...")
 
        print("[Tracker] Inventory failed. Using one instance per class.")
        for obj_name, obj_data in self.schema["objects"].items():
            self.memory_bank.register_instance(
                instance_id=f"{obj_name.lower().replace(' ', '_')}_main",
                object_class=obj_name,
                description=f"main {obj_name.lower()}",
                initial_state=obj_data["initial_state"],
            )
 
    def _call_vlm(self, anchor_frame: Frame, current_frame: Frame) -> dict | None:
        prompt = VLM_VERIFICATION_PROMPT.format(
            task_description=self.schema.get("task", ""),
            process_steps_str=self._build_process_steps_str(),
            current_step=self.memory_bank.current_step,
            position_table_str=self._build_position_table_str(),
            recent_history_str=self._build_recent_history_str(),
            possible_states_str=self._build_possible_states_str(),
        )
 
        for attempt in range(config.VLM_MAX_RETRIES + 1):
            try:
                raw = self.vlm.query(
                    images=[anchor_frame.image, current_frame.image],
                    prompt=prompt,
                    system_prompt=VLM_SYSTEM_PROMPT,
                )
                result = self._parse_json(raw)
                if result is not None:
                    return result
                if attempt < config.VLM_MAX_RETRIES:
                    print(f"  [VLM] Parse error (attempt {attempt + 1}), retrying...")
            except Exception as e:
                if attempt < config.VLM_MAX_RETRIES:
                    print(f"  [VLM] Error (attempt {attempt + 1}): {e}")
                else:
                    print(f"  [VLM] Failed after {config.VLM_MAX_RETRIES + 1} attempts: {e}")
                    return None
        return None
 
    def _validate_forward_only(self, instance_id: str, to_state: str) -> bool:
        inst = self.memory_bank.instances.get(instance_id)
        if not inst:
            return False
        obj_class = inst.object_class
        states = self.schema["objects"].get(obj_class, {}).get("states", [])
        current_idx = states.index(inst.current_state) if inst.current_state in states else -1
        new_idx = states.index(to_state) if to_state in states else -1
        if new_idx < current_idx:
            print(f"  [Tracker] BLOCKED backward: {instance_id} "
                  f"{inst.current_state} -> {to_state}")
            return False
        return True
 
    def process_frames(self, frames: list[Frame]):
        if not frames:
            print("[Tracker] No frames to process.")
            return
 
        self._run_inventory(frames[0])
 
        self.change_detector.set_anchor(frames[0].image)
        self.anchor_frame = frames[0]
        print(f"[Tracker] Anchor set to frame 0 @ {frames[0].timestamp}s")
        print(f"[Tracker] Processing {len(frames)} frames...\n")
 
        for frame in frames[1:]:
            self.frames_processed += 1
 
            changed, similarity = self.change_detector.check_change(frame.image)
            if not changed:
                continue
 
            self.changes_detected += 1
            self.vlm_calls += 1
            print(f"[Frame {frame.index}] Change @ {frame.timestamp:.1f}s "
                  f"(sim={similarity:.3f}). Calling VLM...")
 
            vlm_result = self._call_vlm(self.anchor_frame, frame)
 
            if vlm_result is None:
                print(f"  [Frame {frame.index}] Unparseable response. Skipping.")
                continue
 
            if not vlm_result.get("changed", False):
                print(f"  [Frame {frame.index}] NO_CHANGE.")
                continue
 
            self.memory_bank.current_step = vlm_result.get(
                "current_step", self.memory_bank.current_step
            )
 
            for new_inst in vlm_result.get("new_instances", []):
                obj_class = new_inst.get("class", "")
                if obj_class not in self.schema["objects"]:
                    continue
                state = new_inst.get("current_state", "")
                if state not in self.schema["objects"][obj_class]["states"]:
                    state = self.schema["objects"][obj_class]["initial_state"]
                self.memory_bank.register_instance(
                    instance_id=new_inst.get("instance_id", ""),
                    object_class=obj_class,
                    description=new_inst.get("description", ""),
                    initial_state=state,
                )
 
            for pos in vlm_result.get("position_updates", []):
                self.memory_bank.update_position(
                    pos.get("instance_id", ""),
                    pos.get("new_description", ""),
                )
 
            for t in vlm_result.get("transitions", []):
                iid = t.get("instance_id", "")
                to_state = t.get("to_state", "")
 
                if iid not in self.memory_bank.instances:
                    print(f"  [Frame {frame.index}] Unknown '{iid}'. Skipping.")
                    continue
 
                inst_class = self.memory_bank.instances[iid].object_class
                valid_states = self.schema["objects"].get(inst_class, {}).get("states", [])
                if to_state not in valid_states:
                    print(f"  [Frame {frame.index}] Invalid state '{to_state}' for {inst_class}.")
                    continue
 
                if not self._validate_forward_only(iid, to_state):
                    continue
 
                self.memory_bank.record_transition(
                    instance_id=iid,
                    object_class=inst_class,
                    from_state=t.get("from_state", ""),
                    to_state=to_state,
                    timestamp=frame.timestamp,
                    frame_number=frame.frame_number,
                    confidence=t.get("confidence", "medium"),
                    reason=t.get("reason", ""),
                )
                self.transitions_confirmed += 1
 
            self.change_detector.set_anchor(frame.image)
            self.anchor_frame = frame
            print(f"  [Frame {frame.index}] Anchor updated to {frame.timestamp:.1f}s\n")
 
        print(f"\n[Tracker] Complete.")
        print(f"  Frames processed: {self.frames_processed}")
        print(f"  Changes detected: {self.changes_detected}")
        print(f"  VLM calls made:   {self.vlm_calls}")
        print(f"  Transitions:      {self.transitions_confirmed}")
 
    def get_memory_bank(self) -> MemoryBank:
        return self.memory_bank