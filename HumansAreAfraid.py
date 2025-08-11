"""
AGICore.py

Upgraded AGICore that merges a BabyAGI-style task loop (fast autonomous iterate) with
AIGodUltima's modular pillars (memory, planner, world-model, safety, tools, self-model).

- Still sandboxed: any external/hardware/network tool requires explicit human approval.
- Includes a demo LLM adapter stub, a deterministic embedder, an in-memory vector DB,
  and a BabyAGI-like task queue that produces/consumes tasks while consulting memory.

Run as a demo: `python AGICore.py` — it will run a few simulated cycles and show logs.

Security: Do NOT wire in network/hardware tools until you perform a security review.
"""

from __future__ import annotations
import time
import uuid
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque, defaultdict

logger = logging.getLogger("AGICore")
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Types
# ----------------------------

@dataclass
class Objective:
    id: str
    desc: str
    priority: float
    created_at: float
    metadata: Dict[str, Any]

@dataclass
class MemoryItem:
    id: str
    ts: float
    content: str
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    priority: float = 0.5
    summary: Optional[str] = None

# ----------------------------
# Embedding & Vector DB (in-memory pluggable)
# ----------------------------

class Embedder:
    """Deterministic toy embedder. Replace with API/local model for production."""
    def embed(self, text: str) -> List[float]:
        # deterministic simple hash -> small vector
        h = sum(ord(c) for c in text) % 1009
        base = [(h + i * 7) % 997 / 997.0 for i in range(16)]
        return base

class InMemoryVectorDB:
    def __init__(self):
        self.store: Dict[str, Tuple[List[float], MemoryItem]] = {}

    def add(self, mem: MemoryItem):
        if mem.embedding is None:
            raise ValueError("embedding required")
        self.store[mem.id] = (mem.embedding, mem)

    def search(self, embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        def cosine(a,b):
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            if na==0 or nb==0: return 0.0
            return dot/(na*nb)
        scored = [(cosine(embedding, emb), mem) for emb, mem in ( (v[0], v[1]) for v in self.store.values() )]
        scored.sort(key=lambda x: -x[0])
        return [m for s,m in scored[:top_k]]

# ----------------------------
# Simple LLM adapter (stub)
# ----------------------------

class LLMStub:
    """Toy LLM replacement. Deterministic pseudo-responses based on prompt contents.
    Replace with OpenAI/other LLM wrapper in production.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        # deterministic-ish mapping: pick words that appear, echo structure
        random.seed(hash(prompt) ^ self.seed)
        keywords = [w for w in prompt.split() if len(w)>4][:6]
        out = "Response:
"
        if not keywords:
            out += "I need more context."
        else:
            for i,k in enumerate(keywords):
                out += f"- Insight about {k}: score {random.random():.2f}
"
            # also produce a toy subtask suggestion
            if 'objective' in prompt.lower() or 'task' in prompt.lower():
                sub = f"Investigate_{keywords[0]}_details"
                out += f"
Suggested_subtask: {sub}"
        return out

# ----------------------------
# Perception & Memory
# ----------------------------

class Perception:
    def __init__(self, embedder: Embedder, vdb: InMemoryVectorDB):
        self.embedder = embedder
        self.vdb = vdb

    def perceive_text(self, text: str, tags: Optional[List[str]] = None, priority: float = 0.5) -> MemoryItem:
        mem = MemoryItem(id=str(uuid.uuid4()), ts=time.time(), content=text, tags=tags or [], priority=priority)
        mem.embedding = self.embedder.embed(text)
        self.vdb.add(mem)
        logger.info(f"[Perception] Stored memory {mem.id} tags={mem.tags}")
        return mem

# ----------------------------
# Memory manager (working + episodic)
# ----------------------------

class MemoryManager:
    def __init__(self, vdb: InMemoryVectorDB, embedder: Embedder):
        self.vdb = vdb
        self.embedder = embedder
        self.working: deque[MemoryItem] = deque(maxlen=50)
        self.episodic_ids: deque[str] = deque(maxlen=5000)

    def store(self, content: str, tags: Optional[List[str]] = None, priority: float = 0.5) -> MemoryItem:
        mem = MemoryItem(id=str(uuid.uuid4()), ts=time.time(), content=content, tags=tags or [], priority=priority)
        mem.embedding = self.embedder.embed(content)
        self.vdb.add(mem)
        self.working.append(mem)
        self.episodic_ids.append(mem.id)
        return mem

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        emb = self.embedder.embed(query)
        return self.vdb.search(emb, top_k=top_k)

# ----------------------------
# Planner + WorldModel (toy)
# ----------------------------

class WorldModel:
    def predict(self, state: Dict[str,Any], action: Dict[str,Any]) -> Tuple[Dict[str,Any], float, Dict[str,Any]]:
        nxt = dict(state)
        if 'inc' in action:
            nxt[action['inc']] = nxt.get(action['inc'], 0) + 1
        reward = random.uniform(0.0, 1.0)
        return nxt, reward, {'conf':0.5}

class Planner:
    def __init__(self, world_model: WorldModel, llm: LLMStub):
        self.world = world_model
        self.llm = llm

    def decompose(self, objective: Objective) -> List[Dict[str,Any]]:
        prompt = f"Decompose objective: {objective.desc}"
        resp = self.llm.generate(prompt)
        # parse suggested subtask if present
        subtasks = []
        for line in resp.splitlines():
            if line.strip().startswith('Suggested_subtask:'):
                name = line.split(':',1)[1].strip()
                subtasks.append({'inc': name})
        if not subtasks:
            # fallback to toy decomposition
            for i in range(random.randint(2,4)):
                subtasks.append({'inc': f"{objective.id}_step_{i}"})
        return subtasks

    def plan(self, objective: Objective, state: Dict[str,Any]) -> List[Dict[str,Any]]:
        parts = self.decompose(objective)
        # imagine and choose best ordering (toy)
        cand = parts.copy()
        random.shuffle(cand)
        return cand

# ----------------------------
# Safety, Tools, SelfModel, Evaluator
# ----------------------------

class Safety:
    def __init__(self, banned_tokens: Optional[List[str]]=None):
        self.banned = set(banned_tokens or ['exec','rm -rf','ssh','net','crawl'])

    def vet(self, action: Dict[str,Any]) -> Tuple[bool,str]:
        s = str(action).lower()
        for b in self.banned:
            if b in s:
                return False, f"banned token {b}"
        return True, 'ok'

class Tool:
    def __init__(self, name:str, fn:Callable[...,Any], requires_approval:bool=True):
        self.name = name
        self.fn = fn
        self.requires_approval = requires_approval

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, t: Tool):
        self.tools[t.name] = t

    def call(self, name:str, *args, **kwargs):
        if name not in self.tools:
            raise ValueError('unknown tool')
        tool = self.tools[name]
        if tool.requires_approval:
            raise PermissionError('tool requires human approval')
        return tool.fn(*args, **kwargs)

class SelfModel:
    def __init__(self):
        self.history: List[str] = []

    def update(self, summary: str):
        self.history.append(summary)
        if len(self.history) > 500:
            self.history.pop(0)

class Evaluator:
    def __init__(self):
        self.episodes: List[Dict[str,Any]] = []
    def record(self, obj:Objective, plan:List[Dict[str,Any]], outcome:Dict[str,Any]):
        self.episodes.append({'ts':time.time(), 'obj':obj, 'plan_len':len(plan), 'outcome':outcome})
    def summarize(self):
        if not self.episodes: return {'episodes':0}
        recent = self.episodes[-20:]
        succ = sum(1 for e in recent if e['outcome'].get('success'))/len(recent)
        avg_len = sum(e['plan_len'] for e in recent)/len(recent)
        return {'episodes': len(recent), 'success_rate':succ, 'avg_len': avg_len}

# ----------------------------
# Task queue + Runner (BabyAGI style)
# ----------------------------

class TaskQueue:
    def __init__(self):
        self.q: deque[Objective] = deque()

    def add(self, desc: str, priority: float = 0.5, metadata: Optional[Dict[str,Any]]=None) -> Objective:
        obj = Objective(id=str(uuid.uuid4()), desc=desc, priority=priority, created_at=time.time(), metadata=metadata or {})
        self.q.append(obj)
        logger.info(f"[TaskQueue] Added {obj.id} : {desc}")
        return obj

    def pop_top(self) -> Optional[Objective]:
        if not self.q: return None
        # simple priority selection
        best = max(self.q, key=lambda o: o.priority)
        self.q.remove(best)
        return best

    def is_empty(self) -> bool:
        return len(self.q)==0

class AGIRunner:
    def __init__(self):
        self.embedder = Embedder()
        self.vdb = InMemoryVectorDB()
        self.perception = Perception(self.embedder, self.vdb)
        self.memory = MemoryManager(self.vdb, self.embedder)
        self.llm = LLMStub()
        self.world = WorldModel()
        self.planner = Planner(self.world, self.llm)
        self.safety = Safety()
        self.tools = ToolManager()
        self.self_model = SelfModel()
        self.evaluator = Evaluator()
        self.taskq = TaskQueue()
        self.state: Dict[str,Any] = {'counters':{}}

    def bootstrap_demo(self):
        # seed memories and a starting task
        self.perception.perceive_text('Startup: user asked to research solar micro-harvesters', tags=['user','request'])
        self.taskq.add('Research solar micro energy harvesters', priority=0.8)

    def run_cycle(self, human_cb: Optional[Callable[[Dict[str,Any]],bool]] = None):
        obj = self.taskq.pop_top()
        if obj is None:
            logger.info('[AGIRunner] No tasks to run')
            return {'success': False, 'reason': 'no_tasks'}

        logger.info(f"[AGIRunner] Working on {obj.id} -> {obj.desc}")

        # recall relevant memory
        context = self.memory.recall(obj.desc, top_k=5)
        ctx_text = '
'.join(m.content for m in context)

        # plan
        plan = self.planner.plan(obj, self.state)
        logger.info(f"[AGIRunner] Generated plan with {len(plan)} steps")

        # execute (simulate or call tools with approval)
        for step in plan:
            ok, reason = self.safety.vet(step)
            if not ok:
                logger.warning(f"[AGIRunner] Safety veto: {reason}")
                self.evaluator.record(obj, plan, {'success': False, 'reason': reason})
                return {'success': False, 'reason': reason}

            if 'tool' in step:
                tname = step['tool']
                tool = self.tools.tools.get(tname)
                if tool is None:
                    logger.warning('[AGIRunner] Missing tool. Simulating result.')
                    self.state, r, info = self.world.predict(self.state, {'inc': 'sim_missing_tool'})
                    continue
                # require approval
                approved = False
                if human_cb:
                    approved = human_cb(step)
                if not approved:
                    logger.info('[AGIRunner] Human denied tool use; simulating instead')
                    self.state, r, info = self.world.predict(self.state, {'inc': 'sim_tool_denied'})
                else:
                    try:
                        res = self.tools.call(tname, **(step.get('args') or {}))
                        logger.info(f"[AGIRunner] Tool {tname} executed -> {res}")
                    except Exception as e:
                        logger.exception('Tool failed')
                        self.evaluator.record(obj, plan, {'success': False, 'reason': str(e)})
                        return {'success': False, 'reason': str(e)}
            else:
                self.state, r, info = self.world.predict(self.state, step)
                logger.info(f"[AGIRunner] Sim step -> r={r:.3f}")

        # postprocessing: write memory, suggest new tasks
        summary = f"Completed objective {obj.id}: {obj.desc}"
        self.memory.store(summary, tags=['summary'])
        self.self_model.update(summary)
        self.evaluator.record(obj, plan, {'success': True})

        # ask LLM for followups
        follow_prompt = f"Given result of task '{obj.desc}', propose 0-3 followup tasks. Context:
{ctx_text}
Summary:
{summary}"
        llm_out = self.llm.generate(follow_prompt)
        # parse toy suggestions
        for line in llm_out.splitlines():
            if line.strip().startswith('Suggested_subtask:'):
                desc = line.split(':',1)[1].strip()
                self.taskq.add(desc, priority=0.6)

        logger.info(f"[AGIRunner] Cycle complete for {obj.id}")
        return {'success': True}

# ----------------------------
# Demo run
# ----------------------------

if __name__ == '__main__':
    runner = AGIRunner()
    runner.bootstrap_demo()

    # register a safe local tool (no network/hardware)
    def local_calc(x=0,y=0):
        return {'sum': x+y, 'product': x*y}
    runner.tools.register(Tool('local_calc', local_calc, requires_approval=False))

    def human_cb(action: Dict[str,Any]) -> bool:
        # demo approval: reject anything with banned tokens
        s = str(action).lower()
        if any(tok in s for tok in ['net','ssh','exec']):
            return False
        return True

    # run a few cycles
    for i in range(3):
        res = runner.run_cycle(human_cb)
        print(f'Cycle {i+1} result:', res)
        time.sleep(0.2)

    print('Evaluator summary:', runner.evaluator.summarize())

# End of AGICore.py
