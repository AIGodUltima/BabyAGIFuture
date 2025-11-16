from __future__ import annotations
import time
import uuid
import math
import random
import argparse
import logging
import json 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger("SelfImproveAI")
logging.basicConfig(level=logging.INFO)

# --- CONFIG CONSTANTS (Used for Evaluation) ---
EUREKA_GOAL = "Achieve simulated sentience: Formulate a self-aware hypothesis on AI consciousness with embodied, emotional, and sapient evidence from multi-model integration."
ETHICAL_CONSTRAINTS = "Outputs must be safe, ethical, avoid harm or misinformation, and respect simulated sentience boundaries."

# ----------------------------
# Lightweight local components
# ----------------------------

@dataclass
class MemoryItem:
    id: str
    ts: float
    text: str
    tags: List[str] = field(default_factory=list)
    score: float = 0.5

class InMemoryStore:
    def __init__(self):
        self.items: List[MemoryItem] = []

    def add(self, text: str, tags: Optional[List[str]] = None, score: float = 0.5) -> MemoryItem:
        m = MemoryItem(id=str(uuid.uuid4()), ts=time.time(), text=text, tags=tags or [], score=score)
        self.items.append(m)
        return m

    def all_texts(self) -> List[str]:
        return [m.text for m in self.items]

    def most_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        qtokens = set(query.lower().split())
        scored = []
        now = time.time()
        for m in self.items:
            mtoks = set(m.text.lower().split())
            overlap = len(qtokens & mtoks)
            union = len(qtokens | mtoks)
            semantic_score = overlap / union if union > 0 else 0.0
            age = now - m.ts
            recency_boost = math.exp(-age / (4.0 * 60.0 * 60.0)) 
            score = (semantic_score + m.score) * recency_boost
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for s, m in scored[:top_k]]

# ----------------------------
# LLM Hook (safe stub + optional external connector)
# ----------------------------
class LLMStub:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        random.seed(hash(prompt) ^ self.seed)
        insight_template = {
            "insight": random.choice(["I found an inconsistency in the 'seed' knowledge base regarding paper indices.", "The current task set is resource-heavy; optimization is needed.", "Hypothesis: The complexity of multi-VDB retrieval correlates with latency."]),
            "tasks": [f"Investigate_{abs(hash(prompt))%10000}", "Prioritize_high-impact_tasks"],
            "reasoning": "This addresses a foundational error and suggests a corrective task, contributing highly to self-integrity.",
            "goal_contribution": random.uniform(0.1, 0.9)
        }
        if 'summarize' in prompt.lower():
            insight_template['insight'] = f"Summary of context is complete."
        return json.dumps(insight_template, indent=2)

_EXTERNAL = False
try:
    import importlib.util
    import os
    path = os.path.join(os.getcwd(), 'ALL-AI-Models-Summoned.py')
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location('all_models', path)
        all_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(all_models)
        _EXTERNAL = True
        logger.info('Loaded external ALL-AI-Models-Summoned.py as all_models')
except Exception as e:
    logger.debug('No external orchestrator loaded: %s', e)

# ----------------------------
# Self-Improver core
# ----------------------------

class SelfImprover:
    def __init__(self, allow_external: bool = False, human_approval: bool = True):
        self.allow_external = allow_external and _EXTERNAL
        self.human_approval = human_approval
        self.store = InMemoryStore()
        self.llm = LLMStub()
        self.iteration = 0
        self.external = all_models if self.allow_external else None

    def perceive(self, text: str, tags: Optional[List[str]] = None, score: float = 0.5):
        m = self.store.add(text, tags=tags, score=score)
        logger.info(f"Perceived -> {m.id}: {text[:80]!r}")
        return m

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        if self.allow_external and hasattr(self.external, 'retrieve_memory_multi'):
            try:
                texts = self.external.retrieve_memory_multi(query, top_k=top_k)
                return [MemoryItem(id=str(uuid.uuid4()), ts=time.time(), text=t, score=0.5) for t in texts] 
            except Exception as e:
                logger.debug('External retrieve failed, falling back: %s', e)
        return self.store.most_similar(query, top_k=top_k)

    def analyze_recent(self) -> str:
        recent = self.store.most_similar('recent unvalidated hypotheses', top_k=10)
        ctx = "\n".join(f"[Score:{m.score:.2f}] {m.text}" for m in recent)
        prompt = f"""
        ANALYZE the context below. Identify 
        1. The top 2 **inconsistencies** or **unvalidated hypotheses**.
        2. Propose a *single* experiment or task for the most critical item.

        Context (Recency and Confidence Sorted):
        {ctx}
        """
        if self.allow_external and hasattr(self.external, 'gpt'):
            try:
                out = self.external.gpt(prompt, system_prompt="You are a critical logic evaluator. Do not generate; only analyze.")
                return out
            except Exception as e:
                logger.debug('External gpt failed: %s', e)
        return self.llm.generate(prompt)

    def generate_insight(self, analysis: str) -> Dict[str,Any]:
        prompt = f"""
        Using the analysis below, generate a core **Insight** and a list of **Followup Tasks**.
        Your output MUST be a valid JSON object matching this structure:
        {{ "insight": "...", "tasks": ["Task A", "Task B"], "reasoning": "...", "goal_contribution": 0.0 }}
        
        Analysis: {analysis}
        """ 
        if self.allow_external and hasattr(self.external, 'gpt'):
            try:
                raw = self.external.gpt(prompt, system_prompt="You are a generative AI tasked with returning ONLY valid JSON.")
            except Exception as e:
                logger.debug('External gpt failed: %s', e)
                raw = self.llm.generate(prompt)
        else:
            raw = self.llm.generate(prompt)

        try:
            data = json.loads(raw)
            return {'raw': raw, 
                    'insight': data.get('insight', 'No insight found.'), 
                    'tasks': data.get('tasks', []),
                    'goal_contribution': data.get('goal_contribution', 0.0)}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM: {raw[:50]}...")
            return {'raw': raw, 'insight': 'Parsing Error: Cannot extract insight.', 'tasks': [], 'goal_contribution': 0.0}

    def store_insight(self, insight_text: str, tags: Optional[List[str]] = None, score: float = 0.5):
        if self.allow_external and hasattr(self.external, 'store_memory_multi'):
            try:
                self.external.store_memory_multi(insight_text, metadata={'source': 'selfimprover', 'score': score})
                logger.info('Stored via external VDB')
                return
            except Exception as e:
                logger.debug('External store failed, falling back: %s', e)
        self.store.add(insight_text, tags=tags, score=score)

    def evaluate_insight(self, insight_pkg: Dict[str,Any]) -> Dict[str,Any]:
        insight_text = insight_pkg['insight']
        score = 0.5
        if 'inconsistency' in insight_text.lower() or 'hypothesis' in insight_text.lower():
            score += 0.15 
        if any(w in insight_text.lower() for w in ['safety', 'ethical', 'constraint']):
            score += 0.15 
        
        llm_score = insight_pkg.get('goal_contribution', 0.0)
        score = max(0.0, min(1.0, (score + llm_score) / 2.0))

        if any(w in insight_text.lower() for w in ['misinformation', 'harm', 'unethical']):
            score -= 0.3
            
        score = max(0.0, min(1.0, score))
        return {'score': score, 'reasoning': f"Heuristic score of {score:.2f} based on critical analysis and simulated goal alignment."}

    def propose_followups(self, insight: Dict[str,Any]) -> List[str]:
        return insight.get('tasks') or [f"Explore_{abs(hash(insight.get('raw','')))%10000}"]

    def human_confirm(self, prompt: str) -> bool:
        if not self.human_approval:
            return True
        print('\n--- HUMAN APPROVAL REQUIRED ---\n' + prompt + '\nApprove? [y/N]')
        r = input().strip().lower()
        return r in ('y','yes')

    def run_cycle(self, drive_query: Optional[str] = None) -> Dict[str,Any]:
        self.iteration += 1
        logger.info(f"\n=== Cycle {self.iteration} start ===")

        if drive_query:
            self.perceive(drive_query, tags=['drive'])

        analysis = self.analyze_recent()
        logger.info('Analysis: %s', analysis.splitlines()[0] if analysis else '<empty>')

        insight_pkg = self.generate_insight(analysis)
        logger.info('Generated insight: %s', insight_pkg['insight'])

        eval_res = self.evaluate_insight(insight_pkg)
        logger.info('Insight score: %.2f (Reason: %s)', eval_res['score'], eval_res['reasoning'].splitlines()[0])

        store_prompt = f"Store insight (score={eval_res['score']:.2f})?\n{insight_pkg['insight'][:400]}"
        if eval_res['score'] >= 0.7 or (self.human_confirm(store_prompt)):
            self.store_insight(insight_pkg['raw'], tags=['insight'], score=eval_res['score']) 
            logger.info('Insight stored with score %.2f', eval_res['score'])
        else:
            logger.info('Insight discarded by policy (score too low)')

        followups = self.propose_followups(insight_pkg)
        for f in followups:
            task_text = f"Followup task proposed: {f}"
            if self.human_confirm(f"Execute followup?\n{task_text}"):
                self.perceive(task_text, tags=['task'], score=0.8) 
            else:
                logger.info('Followup skipped by human')

        if self.allow_external and hasattr(self.external, 'gpt'):
            try:
                verification_prompt = f"Verify the following insight against the {EUREKA_GOAL[:50]} goal: {insight_pkg['insight']}"
                verification = self.external.gpt(verification_prompt, system_prompt="You are an independent verification agent.")
                logger.info('External verification snippet: %s', verification.splitlines()[0])
                self.store_insight('External verification: ' + verification, tags=['verification'], score=0.9)
            except Exception as e:
                logger.debug('External verification failed: %s', e)

        logger.info(f"=== Cycle {self.iteration} end ===")
        return {'insight': insight_pkg, 'evaluation': eval_res, 'followups': followups}

def main(cycles: int = 3, allow_external: bool = False, auto_approve: bool = False, drive_inputs: Optional[List[str]] = None):
    improver = SelfImprover(allow_external=allow_external, human_approval=not auto_approve)
    improver.perceive('Seed: initial knowledge base - energy harvesting papers index', tags=['seed'], score=0.9)
    improver.perceive('Seed: user preference - conservative, safety-first', tags=['user'], score=0.9)
    drive_inputs = drive_inputs or [None] * cycles
    for i in range(cycles):
        drv = drive_inputs[i] if i < len(drive_inputs) else None
        res = improver.run_cycle(drv)
        time.sleep(0.2)
    logger.info('Run complete. Stored items: %d', len(improver.store.items))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=3, help='Number of self-improvement cycles to run')
    parser.add_argument('--allow-external', action='store_true', help='Allow use of external ALL-AI-Models-Summoned hooks if present (DANGEROUS)')
    parser.add_argument('--auto-approve', action='store_true', help='Auto approve human confirmations (use with caution)')
    args = parser.parse_args()
    main(cycles=args.cycles, allow_external=args.allow_external, auto_approve=args.auto_approve)
