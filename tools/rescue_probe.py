"""Paired synthetic CPU/coverage probe; NOT official scores or weather replay."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import pro_sim

parser = argparse.ArgumentParser()
parser.add_argument('--rescue', action='store_true')
parser.add_argument('--repair', action='store_true')
parser.add_argument('--card', default='cardB')
args = parser.parse_args()
original = pro_sim.ObserverAgent
instances = []
trace = hashlib.sha256()


def build(*a, **kw):
    agent = original(*a, **kw)
    agent.planner.rescue_mode = args.rescue
    if args.repair:
        agent.planner.attempts = [20 if r else 0 for r in agent.planner.required]
        agent.planner.forget_quality_history()
    instances.append(agent)
    respond = agent.respond
    def tracked(payload):
        action = respond(payload)
        trace.update(json.dumps(action, sort_keys=True).encode())
        return action
    agent.respond = tracked
    return agent


pro_sim.ObserverAgent = build
with open(os.devnull, 'w') as quiet, contextlib.redirect_stderr(quiet):
    result = pro_sim.run_card(args.card, nights_limit=2, max_decisions=2000)
p = instances[0].planner
result['estimated_required_completed'] = sum(r and f >= .5 for r, f in zip(p.required, p.factor))
result['rescue'] = args.rescue
result['action_trace_sha256'] = trace.hexdigest()
result['seeded_failed_attempts_before_repair'] = 20 if args.repair else 0
print(json.dumps(result))
