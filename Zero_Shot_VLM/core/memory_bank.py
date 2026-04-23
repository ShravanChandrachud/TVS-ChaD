"""
Memory Bank — tracks every object independently with its own lifecycle.
Position table tracks where each instance currently is.
"""
 
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
 
 
@dataclass
class Transition:
    """A single state transition event."""
    instance_id: str
    object_class: str
    from_state: str
    to_state: str
    timestamp: float
    frame_number: int
    confidence: str
    reason: str
 
 
@dataclass
class ObjectInstance:
    """A single tracked object instance."""
    instance_id: str
    object_class: str
    description: str
    current_state: str
    transitions: list[Transition] = field(default_factory=list)
 
 
class MemoryBank:
 
    def __init__(self, task: str, schema: dict):
        self.task = task
        self.schema = schema
        self.instances: dict[str, ObjectInstance] = {}
        self.current_step: str = schema.get("process_steps", ["Unknown"])[0]
 
    def register_instance(self, instance_id: str, object_class: str,
                          description: str, initial_state: str):
        if instance_id in self.instances:
            return
        self.instances[instance_id] = ObjectInstance(
            instance_id=instance_id,
            object_class=object_class,
            description=description,
            current_state=initial_state,
        )
        print(f"  [Registry] + {instance_id} ({object_class}) = {initial_state}")
        print(f"             @ {description}")
 
    def update_position(self, instance_id: str, new_description: str):
        if instance_id in self.instances:
            old = self.instances[instance_id].description
            self.instances[instance_id].description = new_description
            if old != new_description:
                print(f"  [Position] {instance_id}: \"{old}\" -> \"{new_description}\"")
 
    def get_position_table(self) -> dict[str, dict]:
        return {
            iid: {
                "class": inst.object_class,
                "description": inst.description,
                "state": inst.current_state,
            }
            for iid, inst in self.instances.items()
        }
 
    def get_recent_history(self, n: int = 10) -> list[Transition]:
        all_t = []
        for inst in self.instances.values():
            all_t.extend(inst.transitions)
        all_t.sort(key=lambda t: t.timestamp)
        return all_t[-n:]
 
    def record_transition(self, instance_id: str, object_class: str,
                          from_state: str, to_state: str,
                          timestamp: float, frame_number: int,
                          confidence: str = "high", reason: str = ""):
        if instance_id not in self.instances:
            print(f"  [MemoryBank] Unknown instance '{instance_id}', skipping.")
            return
        if from_state == to_state:
            return
 
        transition = Transition(
            instance_id=instance_id,
            object_class=object_class,
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp,
            frame_number=frame_number,
            confidence=confidence,
            reason=reason,
        )
        self.instances[instance_id].transitions.append(transition)
        self.instances[instance_id].current_state = to_state
 
        print(f"  [MemoryBank] {instance_id}: {from_state} -> {to_state} "
              f"@ {timestamp:.1f}s (frame {frame_number}) [{confidence}]")
 
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "generated_at": datetime.now().isoformat(),
            "schema": self.schema,
            "current_step": self.current_step,
            "instances": {
                iid: {
                    "instance_id": inst.instance_id,
                    "object_class": inst.object_class,
                    "description": inst.description,
                    "current_state": inst.current_state,
                    "transitions": [asdict(t) for t in inst.transitions],
                }
                for iid, inst in self.instances.items()
            },
            "summary": {
                "total_transitions": sum(len(i.transitions) for i in self.instances.values()),
                "instances_tracked": len(self.instances),
            },
        }
 
    def save(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[MemoryBank] JSON saved to {output_path}")
 
    def save_table(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        all_transitions = []
        for inst in self.instances.values():
            for t in inst.transitions:
                all_transitions.append(t)
        all_transitions.sort(key=lambda t: t.timestamp)
 
        if not all_transitions:
            print("[MemoryBank] No transitions to write.")
            return
 
        w_time = 8
        w_obj = max(max(len(t.instance_id) for t in all_transitions), len("Object"))
        w_new = max(max(len(t.to_state) for t in all_transitions), len("New State"))
        w_prev = max(max(len(t.from_state) for t in all_transitions), len("Prev State"))
        w_frame = 8
        w_conf = 10
 
        header = (f"{'Time':<{w_time}} | {'Object':<{w_obj}} | "
                  f"{'New State':<{w_new}} | {'Prev State':<{w_prev}} | "
                  f"{'Frame':<{w_frame}} | {'Confidence':<{w_conf}}")
        sep = "-" * len(header)
 
        lines = []
        lines.append(f"MEMORY BANK - {self.task}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Instances tracked: {len(self.instances)}")
        lines.append(f"Total transitions: {len(all_transitions)}")
        lines.append("")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
 
        for t in all_transitions:
            mins = int(t.timestamp // 60)
            secs = int(t.timestamp % 60)
            time_str = f"{mins}:{secs:02d}"
            lines.append(
                f"{time_str:<{w_time}} | {t.instance_id:<{w_obj}} | "
                f"{t.to_state:<{w_new}} | {t.from_state:<{w_prev}} | "
                f"{t.frame_number:<{w_frame}} | {t.confidence:<{w_conf}}"
            )
 
        lines.append(sep)
 
        lines.append("")
        lines.append("POSITION TABLE (final snapshot)")
        lines.append("-" * 60)
        for iid, inst in self.instances.items():
            lines.append(f"  {iid:<20} | state: {inst.current_state}")
            lines.append(f"  {'':<20} | at: \"{inst.description}\"")
        lines.append("-" * 60)
 
        table_text = "\n".join(lines)
 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table_text)
 
        print(f"[MemoryBank] Table saved to {output_path}")
        print()
        print(table_text)