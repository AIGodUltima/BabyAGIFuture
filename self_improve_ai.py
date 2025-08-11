"""
SelfImproveAI.py

A safe, auditable "self-improving" loop scaffold that can plug into your larger
multi-model, multi-VDB system. This file is intentionally conservative:
 - Runs locally with no network calls by default (uses LLM stub & in-memory VDB)
 - If you configure real LLM or VDB connectors, the code will use them but will
   require explicit `allow_external=True` to enable any network/hardware activity.
 - Includes human-in-the-loop gating and detailed logging.

Usage (demo):
    python SelfImproveAI.py --cycles 3

If you want this to integrate with your "ALL-AI-Models-Summoned.py", place both files
in the same directory and pass allow_external=True (after thorough security review).
"""

from __future__ import annotations
import time
import uuid
import math
import random
import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger("SelfImproveAI")
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Lightweight local components
# ----------------------------

@dataclass
class MemoryItem:
    id: str
    ts: float
    text: str
    tags: List[str] = field(default_factory=list)
    score: float = 0.0

class InMemoryStore:
    """Simple semantic memory store (text only).
    Optional: replace the embed/get_similar functions with real embeddings/VDB.
    """
    def __init__(self):
        self.items: List[MemoryItem] = []

    def add(self, text: str, tags: Optional[List[str]] = None, score: float = 0.0) -> MemoryItem:
        m = MemoryItem(id=str(uuid.uuid4()), ts=time.time(), text=text, tags=tags or [], score=score)
        self.items.append(m)
        return m

    def all_texts(self) -> List[str]:
        return [m.text for m in self.items]

    def most_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        # naive similarity: shared token overlap + recency boost
        qtokens = set(query.lower().split())
        scored = []
        now = time.time()
        for m in self.items:
            mtoks = set(m.text.lower().split())
            overlap = len(qtokens & mtoks)
            age = now - m.ts
            recency_bonus = math.exp(-age / (60.0 * 60.0))  # 1 hour half-life-ish
            score = overlap + 0.5 * recency_bonus + m.score
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
        # Very small deterministic responses to keep demo offline.
        choices = [
            "I notice related concepts and recommend deeper literature review.",
            "Consider validating this claim with an experiment or authoritative source.",
            "Suggested followup: investigate practical implementation details.",
            "Hypothesis: the observed pattern may correlate with resource constraints."
        ]
        # echo a short insight plus maybe a suggested subtask
        insight = random.choice(choices)
        if 'summarize' in prompt.lower():
            return f"Summary:\n{prompt[:240]}...\nInsight:\n{insight}"
        return f"Insight:\n{insight}\nSuggested_subtask: Investigate_{abs(hash(prompt))%10000}"

# Optional: try to import your large orchestrator (if present)
_EXTERNAL = False
try:
    # The user's large file may have hyphens which breaks imports; attempt a safe import by filename if present
    import importlib.util
    import os
    path = os.path.join(os.getcwd(), 'ALL-AI-Models-Summoned.py')
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location('all_models', path)
        all_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(all_models)  # type: ignore
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

        # If external module loaded and user allowed external, use its helpers
        if self.allow_external:
            # attempt to use store_memory_multi / retrieve_memory_multi / gpt if available
            self.external = all_models
        else:
            self.external = None

    def perceive(self, text: str, tags: Optional[List[str]] = None):
        m = self.store.add(text, tags=tags)
        logger.info(f"Perceived -> {m.id}: {text[:80]!r}")
        return m

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        if self.allow_external and hasattr(self.external, 'retrieve_memory_multi'):
            try:
                # external retrieval returns text list; convert to MemoryItems conservatively
                texts = self.external.retrieve_memory_multi(query, top_k=top_k)
                return [MemoryItem(id=str(uuid.uuid4()), ts=time.time(), text=t) for t in texts]
            except Exception as e:
                logger.debug('External retrieve failed, falling back: %s', e)
        return self.store.most_similar(query, top_k=top_k)

    def analyze_recent(self) -> str:
        # make a short combined context and ask the LLM for insights
        recent = self.store.most_similar('recent', top_k=10)
        ctx = "\n".join(m.text for m in recent)
        prompt = f"Summarize recent findings and propose up to 3 followup tasks. Context:\n{ctx}"
        if self.allow_external and hasattr(self.external, 'gpt'):
            try:
                out = self.external.gpt(prompt)
                return out
            except Exception as e:
                logger.debug('External gpt failed: %s', e)
        return self.llm.generate(prompt)

    def generate_insight(self, query: str) -> Dict[str,Any]:
        prompt = f"Generate an insight & at most 2 followup tasks for: {query}\nBe concise." 
        if self.allow_external and hasattr(self.external, 'gpt'):
            try:
                raw = self.external.gpt(prompt)
            except Exception as e:
                logger.debug('External gpt failed: %s', e)
                raw = self.llm.generate(prompt)
        else:
            raw = self.llm.generate(prompt)
        # parse toy outputs
        insight = raw.splitlines()[0] if raw else "No insight"
        tasks = [line.split(':',1)[1].strip() for line in raw.splitlines() if line.lower().startswith('suggested_subtask')]
        return {'raw': raw, 'insight': insight, 'tasks': tasks}

    def store_insight(self, insight_text: str, tags: Optional[List[str]] = None):
        if self.allow_external and hasattr(self.external, 'store_memory_multi'):
            try:
                self.external.store_memory_multi(insight_text, metadata={'source': 'selfimprover'})
                logger.info('Stored via external VDB')
                return
            except Exception as e:
                logger.debug('External store failed: %s', e)
        self.store.add(insight_text, tags=tags)

    def evaluate_insight(self, insight_text: str) -> Dict[str,Any]:
        # Low-fidelity evaluation: heuristic scoring + optional external validation
        score = 0.5
        if len(insight_text) > 80:
            score += 0.2
        if 'experiment' in insight_text.lower() or 'validate' in insight_text.lower():
            score += 0.1
        # clamp
        score = max(0.0, min(1.0, score))
        return {'score': score}

    def propose_followups(self, insight: Dict[str,Any]) -> List[str]:
        return insight.get('tasks') or [f"Explore_{abs(hash(insight.get('raw','')))%10000}"]

    def human_confirm(self, prompt: str) -> bool:
        if not self.human_approval:
            return True
        print('\nHUMAN APPROVAL REQUIRED:\n' + prompt + '\nApprove? [y/N]')
        r = input().strip().lower()
        return r in ('y','yes')

    def run_cycle(self, drive_query: Optional[str] = None) -> Dict[str,Any]:
        self.iteration += 1
        logger.info(f"=== Cycle {self.iteration} start ===")

        # 1) Perceive (here we optionally generate a perception point or use supplied driver)
        if drive_query:
            self.perceive(drive_query, tags=['drive'])

        # 2) Analyze recent
        analysis = self.analyze_recent()
        logger.info('Analysis: %s', analysis.splitlines()[0] if analysis else '<empty>')

        # 3) Generate one or more insights
        insight_pkg = self.generate_insight(analysis[:400])
        logger.info('Generated insight: %s', insight_pkg['insight'])

        # 4) Evaluate insight
        eval_res = self.evaluate_insight(insight_pkg['raw'])
        logger.info('Insight score: %.2f', eval_res['score'])

        # 5) Human approval to store & act
        store_prompt = f"Store insight (score={eval_res['score']:.2f})?\n{insight_pkg['raw'][:400]}"
        if eval_res['score'] >= 0.6 or (self.human_confirm(store_prompt)):
            self.store_insight(insight_pkg['raw'], tags=['insight'])
            logger.info('Insight stored')
        else:
            logger.info('Insight discarded by policy')

        # 6) Propose followups and add them to memory as tasks
        followups = self.propose_followups(insight_pkg)
        for f in followups:
            task_text = f"Followup task proposed: {f}"
            if self.human_confirm(f"Execute followup?\n{task_text}"):
                self.perceive(task_text, tags=['task'])
            else:
                logger.info('Followup skipped by human')

        # 7) optional external verification (if allowed)
        if self.allow_external and hasattr(self.external, 'gpt'):
            try:
                verification = self.external.gpt(f"Verify: {insight_pkg['raw']}")
                logger.info('External verification snippet: %s', verification.splitlines()[0])
                self.store_insight('External verification: ' + verification, tags=['verification'])
            except Exception as e:
                logger.debug('External verification failed: %s', e)

        logger.info(f"=== Cycle {self.iteration} end ===")
        return {'insight': insight_pkg, 'evaluation': eval_res, 'followups': followups}

# ----------------------------
# CLI entrypoint
# ----------------------------

def main(cycles: int = 3, allow_external: bool = False, auto_approve: bool = False, drive_inputs: Optional[List[str]] = None):
    improver = SelfImprover(allow_external=allow_external, human_approval=not auto_approve)

    # seed small facts
    improver.perceive('Seed: initial knowledge base - energy harvesting papers index', tags=['seed'])
    improver.perceive('Seed: user preference - conservative, safety-first', tags=['user'])

    drive_inputs = drive_inputs or [None] * cycles

    for i in range(cycles):
        drv = drive_inputs[i] if i < len(drive_inputs) else None
        res = improver.run_cycle(drv)
        # short pause
        time.sleep(0.2)
    logger.info('Run complete. Stored items: %d', len(improver.store.items))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=3, help='Number of self-improvement cycles to run')
    parser.add_argument('--allow-external', action='store_true', help='Allow use of external ALL-AI-Models-Summoned hooks if present (DANGEROUS)')
    parser.add_argument('--auto-approve', action='store_true', help='Auto approve human confirmations (use with caution)')
    args = parser.parse_args()
    main(cycles=args.cycles, allow_external=args.allow_external, auto_approve=args.auto_approve)
