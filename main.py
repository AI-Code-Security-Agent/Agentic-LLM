from __future__ import annotations

import asyncio
import difflib
import glob
import hashlib
import httpx
import json
import logging
import os
import pathlib
import re
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("LLM_API")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv(), override=True)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

AGENT_MAX_ITERS = int(os.getenv("AGENT_MAX_ITERS", "3"))
AGENT_ENABLE_WEBSEARCH = os.getenv("AGENT_ENABLE_WEBSEARCH", "false").lower() == "true"
AGENT_WEBSEARCH_BASE = os.getenv("AGENT_WEBSEARCH_BASE", "")
AGENT_ENABLE_SEMGREP = os.getenv("AGENT_ENABLE_SEMGREP", "true").lower() == "true"
AGENT_ENABLE_ESLINT = os.getenv("AGENT_ENABLE_ESLINT", "true").lower() == "true"
SEMGREP_CONFIG = os.getenv("SEMGREP_CONFIG", "p/owasp-top-ten")
ESLINT_CONFIG_PATH = os.getenv("ESLINT_CONFIG_PATH", "") or None

VULN_CLASSIFIER_MODEL_ID = os.getenv("VULN_CLASSIFIER_MODEL_ID", "")
VULN_CLASSIFIER_PROVIDER = os.getenv("VULN_CLASSIFIER_PROVIDER", "")
VULN_FIXER_MODEL_ID = os.getenv("VULN_FIXER_MODEL_ID", "")
VULN_FIXER_PROVIDER = os.getenv("VULN_FIXER_PROVIDER", "")

# ---------------------------------------------------------------------------
# Optional integrations (metrics + langgraph)
# ---------------------------------------------------------------------------
METRICS_ENABLED = False
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    METRICS_ENABLED = True
    agent_iterations = Counter("agent_iterations_total", "Agent iterations", ["reason"])
    tool_latency = Histogram("tool_latency_seconds", "Latency per tool", ["tool"])
    model_latency = Histogram("model_latency_seconds", "Model call latency", ["provider", "model_id"])
except Exception:
    logger.debug("prometheus_client not available; metrics disabled")

LANGGRAPH_OK = False
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_OK = True
except Exception:
    logger.debug("langgraph not available; agent endpoints will return 500")
AGENT_GRAPH = None

# ---------------------------------------------------------------------------
# DTOs / Pydantic models
# ---------------------------------------------------------------------------

class APIModel(BaseModel):
    # allow fields like model_id without warnings
    model_config = ConfigDict(protected_namespaces=())

class Message(APIModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(APIModel):
    message: str
    session_id: Optional[str] = None
    messages: Optional[List[Message]] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    model_id: Optional[str] = None
    provider: Optional[str] = None  # openrouter | hf_inference

class ChatResponse(APIModel):
    response: str
    session_id: str
    message_count: int
    model_id: str
    provider: str

class SessionInfo(APIModel):
    session_id: str
    message_count: int
    created_at: datetime
    last_updated: datetime

class PublicModel(APIModel):
    id: str
    provider: str
    label: str
    stream: bool = False
    base_url: Optional[str] = None



# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
chat_sessions: Dict[str, List[Message]] = {}
session_metadata: Dict[str, Dict] = {}


def create_session_id() -> str:
    return str(uuid.uuid4())


def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id:
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
            session_metadata[session_id] = {
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
            }
            logger.info(f"📝 Adopted external session: {session_id}")
        return session_id
    sid = create_session_id()
    chat_sessions[sid] = []
    session_metadata[sid] = {"created_at": datetime.now(), "last_updated": datetime.now()}
    logger.info(f"📝 New session created: {sid}")
    return sid


def add_message_to_session(sid: str, msg: Message) -> None:
    chat_sessions.setdefault(sid, []).append(msg)
    if sid in session_metadata:
        session_metadata[sid]["last_updated"] = datetime.now()


def format_messages_for_llm(messages: List[Message]) -> List[Dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def build_conversation_context(current_message: str, history: List[Message]) -> List[Dict[str, Any]]:
    llm_messages = format_messages_for_llm(history)
    llm_messages.append({"role": "user", "content": current_message})
    return llm_messages

# ---------------------------------------------------------------------------
# Model registry loader
# ---------------------------------------------------------------------------

def _load_models() -> List[PublicModel]:
    raw = os.getenv("ALLOWED_MODELS_JSON", "").strip()
    if not raw:
        logger.warning("ALLOWED_MODELS_JSON missing or empty")
        return []
    try:
        cleaned = re.sub(r"\\\s*\n", "", raw)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        arr = json.loads(cleaned)
        models = [PublicModel(**x) for x in arr]
        return models
    except Exception as e:
        logger.error("Failed to parse ALLOWED_MODELS_JSON: %s", e)
        logger.debug("Raw value was: %s", raw)
        return []


ALLOWED_MODELS: List[PublicModel] = _load_models()
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openrouter")
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", ALLOWED_MODELS[0].id if ALLOWED_MODELS else "")


def resolve_model(model_id: Optional[str], provider: Optional[str]) -> PublicModel:
    if model_id and provider:
        for m in ALLOWED_MODELS:
            if m.id == model_id and m.provider == provider:
                return m
        raise HTTPException(400, "Requested model/provider not allowed")
    if model_id:
        for m in ALLOWED_MODELS:
            if m.id == model_id:
                return m
        raise HTTPException(400, "Requested model_id not allowed")
    if provider:
        for m in ALLOWED_MODELS:
            if m.provider == provider:
                return m
        raise HTTPException(400, "No allowed model for requested provider")
    # fallback
    for m in ALLOWED_MODELS:
        if m.id == DEFAULT_MODEL_ID and m.provider == DEFAULT_PROVIDER:
            return m
    if ALLOWED_MODELS:
        return ALLOWED_MODELS[0]
    raise HTTPException(500, "No models configured")

# ---------------------------------------------------------------------------
# HTTP client dependency
# ---------------------------------------------------------------------------

async def get_http_client():
    async with httpx.AsyncClient(timeout=120) as client:
        yield client

# ---------------------------------------------------------------------------
# Provider adapters (OpenRouter + HuggingFace Inference)
# ---------------------------------------------------------------------------
HEADERS_OR = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "FastAPI LLM Chatbot",
}
HEADERS_HF = {
    "Authorization": f"Bearer {HF_API_KEY}" if HF_API_KEY else "",
    "Content-Type": "application/json",
}


async def provider_openrouter_chat(
    client: httpx.AsyncClient, model: PublicModel, messages: List[Dict[str, Any]], max_tokens: int, temperature: float
) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(400, "OPENROUTER_API_KEY missing")
    payload = {
        "model": model.id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    start = time.perf_counter()
    resp = await client.post(f"{OPENROUTER_BASE}/chat/completions", headers=HEADERS_OR, json=payload)
    dur = time.perf_counter() - start
    if METRICS_ENABLED:
        model_latency.labels("openrouter", model.id).observe(dur)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"OpenRouter error: {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(500, "Malformed OpenRouter response")


async def provider_openrouter_stream(
    client: httpx.AsyncClient, model: PublicModel, messages: List[Dict[str, Any]], max_tokens: int, temperature: float
) -> AsyncGenerator[str, None]:
    if not OPENROUTER_API_KEY:
        yield f"event: error\ndata: {json.dumps({'detail':'OPENROUTER_API_KEY missing'})}\n\n"
        return
    payload = {
        "model": model.id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    async with client.stream("POST", f"{OPENROUTER_BASE}/chat/completions", headers=HEADERS_OR, json=payload, timeout=None) as resp:
        if resp.status_code != 200:
            detail = await resp.aread()
            yield f"event: error\ndata: {json.dumps({'detail': detail.decode(errors='ignore')})}\n\n"
            return
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                yield "data: [DONE]\n\n"
                break
            try:
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            except Exception:
                continue


async def provider_hf_inference_chat(
    client: httpx.AsyncClient, model: PublicModel, messages: List[Dict[str, Any]], max_tokens: int, temperature: float
) -> str:
    if not HF_API_KEY:
        raise HTTPException(400, "HF_API_KEY missing")
    prompt = ""
    for m in messages:
        role = m["role"]
        prefix = "User: " if role == "user" else "Assistant: " if role == "assistant" else "System: "
        prompt += f"{prefix}{m['content']}\n"
    prompt += "Assistant:"

    url = f"https://api-inference.huggingface.co/models/{model.id}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
    }
    start = time.perf_counter()
    resp = await client.post(url, headers=HEADERS_HF, json=payload)
    dur = time.perf_counter() - start
    if METRICS_ENABLED:
        model_latency.labels("hf_inference", model.id).observe(dur)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"HuggingFace error: {resp.text}")
    data = resp.json()
    try:
        out = data[0].get("generated_text", "")
        if out.startswith(prompt):
            out = out[len(prompt) :]
        return out.strip()
    except Exception:
        raise HTTPException(500, f"Malformed HF response: {data}")


async def provider_hf_inference_stream(
    client: httpx.AsyncClient, model: PublicModel, messages: List[Dict[str, Any]], max_tokens: int, temperature: float
) -> AsyncGenerator[str, None]:
    # simulate stream by splitting on spaces
    try:
        txt = await provider_hf_inference_chat(client, model, messages, max_tokens, temperature)
        for piece in txt.split(" "):
            if piece:
                await asyncio.sleep(0)
                yield f"event: token\ndata: {json.dumps({'token': piece + ' '})}\n\n"
        yield "data: [DONE]\n\n"
    except HTTPException as e:
        yield f"event: error\ndata: {json.dumps({'detail': str(e.detail)})}\n\n"

# ---------------------------------------------------------------------------
# JSON coercion / helpers
# ---------------------------------------------------------------------------

def json_coerce(s: str) -> Any:
    if not isinstance(s, str):
        return s
    t = s.strip()
    t = re.sub(r"^```(json)?", "", t)
    t = re.sub(r"```$", "", t).strip()
    t = re.sub(r",\s*([}\]])", r"\1", t)
    if '"' not in t and t.count("'") > 2:
        t = t.replace("'", '"')
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _timed(tool_name: str):
    """Decorator that records tool latency (works with sync and async functions)."""

    def deco(fn):
        if asyncio.iscoroutinefunction(fn):

            async def wrapper_async(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    if METRICS_ENABLED:
                        tool_latency.labels(tool_name).observe(time.perf_counter() - start)

            return wrapper_async
        else:

            def wrapper_sync(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    if METRICS_ENABLED:
                        tool_latency.labels(tool_name).observe(time.perf_counter() - start)

            return wrapper_sync

    return deco

CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)```", re.DOTALL)

def extract_code_and_lang(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    m = CODE_BLOCK_RE.search(text)
    if m:
        lang = (m.group(1) or "").strip().lower()
        snippet = m.group(2)
        return snippet, (lang or None)
    # Heuristic fallback: likely JS/TS code without fences
    looks_like = any(k in text for k in ["function ", "=>", "const ", "let ", "import ", ";", "{", "try {"])
    if looks_like and len(text) > 40:
        return text, None
    return None, None

def guess_filename(lang: Optional[str]) -> str:
    lang = (lang or "").lower()
    mapping = {
        "js": "snippet.js", "javascript": "snippet.js",
        "ts": "snippet.ts", "typescript": "snippet.ts",
        "jsx": "snippet.jsx", "tsx": "snippet.tsx",
        "py": "snippet.py", "python": "snippet.py",
        "json": "snippet.json",
    }
    return mapping.get(lang, "snippet.js")
# ---------------------------------------------------------------------------
# ESLint / Semgrep / Web search tools
# ---------------------------------------------------------------------------

@_timed("eslint")
def run_eslint(code_str: str, filename_hint: str = "snippet.js", fix: bool = False, config_path: Optional[str] = None, timeout: int = 20):
    """
    Fixed ESLint implementation with proper error handling and configuration
    """
    import tempfile
    import shutil
    
    workdir = tempfile.mkdtemp(prefix="eslint_")
    filepath = os.path.join(workdir, filename_hint)
    
    try:
        # Write the code to file
        pathlib.Path(filepath).write_text(code_str, encoding="utf-8")
        
        # Create basic package.json if it doesn't exist
        package_json_path = os.path.join(workdir, "package.json")
        if not os.path.exists(package_json_path):
            package_json = {
                "name": "eslint-temp",
                "version": "1.0.0",
                "private": True
            }
            with open(package_json_path, "w") as f:
                json.dump(package_json, f)
        
        # Create basic ESLint config if none provided
        eslint_config_path = os.path.join(workdir, ".eslintrc.json")
        if not config_path and not os.path.exists(eslint_config_path):
            eslint_config = {
                "env": {
                    "browser": True,
                    "es2021": True,
                    "node": True
                },
                "extends": ["eslint:recommended"],
                "parserOptions": {
                    "ecmaVersion": "latest",
                    "sourceType": "module"
                },
                "rules": {
                    "no-unused-vars": "error",
                    "no-undef": "error",
                    "no-console": "warn",
                    "semi": ["error", "always"],
                    "quotes": ["error", "single"]
                }
            }
            with open(eslint_config_path, "w") as f:
                json.dump(eslint_config, f, indent=2)
        
        # Prepare ESLint command
        args = ["npx", "--yes", "--package=eslint@latest", "eslint", "--format", "json"]
        
        if fix:
            args.append("--fix")
        
        if config_path and os.path.exists(config_path):
            args.extend(["--config", config_path])
        elif os.path.exists(eslint_config_path):
            args.extend(["--config", eslint_config_path])
        
        # Add ignore patterns for common non-JS files
        args.extend(["--ignore-pattern", "node_modules/"])
        
        args.append(filepath)
        
        # Run ESLint
        try:
            proc = subprocess.run(
                args,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "NODE_ENV": "development"}
            )
            
            # ESLint returns 1 for linting errors, 2 for fatal errors
            if proc.returncode > 2:
                return {
                    "ok": False,
                    "error": f"ESLint failed with code {proc.returncode}: {proc.stderr}",
                    "results": []
                }
            
            # Parse JSON output
            try:
                if proc.stdout.strip():
                    results = json.loads(proc.stdout)
                else:
                    results = []
            except json.JSONDecodeError as e:
                return {
                    "ok": False,
                    "error": f"Failed to parse ESLint JSON output: {e}",
                    "results": [],
                    "raw_stdout": proc.stdout,
                    "raw_stderr": proc.stderr
                }
            
            # Get fixed code if --fix was used
            fixed_code = None
            if fix:
                try:
                    fixed_code = pathlib.Path(filepath).read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not read fixed code: {e}")
                    fixed_code = None
            
            return {
                "ok": True,
                "exit_code": proc.returncode,
                "results": results,
                "fixed_code": fixed_code,
                "stderr": proc.stderr if proc.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"ESLint timed out after {timeout} seconds",
                "results": []
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "error": "ESLint not available. Install Node.js and npm to use ESLint.",
                "results": []
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"ESLint execution failed: {str(e)}",
                "results": []
            }
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


@_timed("semgrep")
def run_semgrep(code_str: str, filename_hint: str = "snippet.js", config: str = SEMGREP_CONFIG, timeout: int = 45):
    """
    Fixed Semgrep implementation with fallback rules
    """
    if not AGENT_ENABLE_SEMGREP:
        return {"ok": False, "error": "semgrep disabled", "results": []}
    
    import tempfile
    import shutil
    
    workdir = tempfile.mkdtemp(prefix="semgrep_")
    filepath = os.path.join(workdir, filename_hint)
    
    try:
        # Write code to file
        pathlib.Path(filepath).write_text(code_str, encoding="utf-8")
        
        # Try multiple Semgrep commands in order of preference
        semgrep_commands = [
            # Try with specified config first
            ["semgrep", "--config", config, "--json", filepath],
            # Fallback to auto config
            ["semgrep", "--config", "auto", "--json", filepath],
            # Fallback to basic security rules
            ["semgrep", "--config", "p/security-audit", "--json", filepath],
            # Last resort: use r2c rules
            ["semgrep", "--config", "r2c", "--json", filepath]
        ]
        
        last_error = None
        
        for cmd in semgrep_commands:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "SEMGREP_TIMEOUT": str(timeout)}
                )
                
                # Semgrep returns 0 for no findings, 1 for findings, >1 for errors
                if proc.returncode > 1:
                    last_error = proc.stderr
                    continue
                
                try:
                    if proc.stdout.strip():
                        data = json.loads(proc.stdout)
                    else:
                        data = {"results": [], "stats": {}}
                except json.JSONDecodeError:
                    last_error = f"Failed to parse Semgrep JSON output: {proc.stdout}"
                    continue
                
                return {
                    "ok": True,
                    "results": data.get("results", []),
                    "stats": data.get("stats", {}),
                    "config_used": " ".join(cmd[2:4])  # Show which config was used
                }
                
            except subprocess.TimeoutExpired:
                last_error = f"Semgrep timed out after {timeout} seconds"
                continue
            except FileNotFoundError:
                last_error = "Semgrep not installed. Install with: pip install semgrep"
                break
            except Exception as e:
                last_error = f"Semgrep execution failed: {str(e)}"
                continue
        
        # If all attempts failed, return fallback analysis
        if last_error and "not installed" in last_error:
            return {"ok": False, "error": last_error, "results": []}
        
        # Return basic pattern-based security check as fallback
        return _fallback_security_analysis(code_str, filename_hint)
        
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass

def _fallback_security_analysis(code_str: str, filename_hint: str):
    """
    Basic security pattern matching when Semgrep is not available
    """
    results = []
    lines = code_str.split('\n')
    
    # Define basic security patterns
    patterns = {
        "eval-usage": {
            "pattern": r'\beval\s*\(',
            "message": "Use of eval() can lead to code injection vulnerabilities",
            "severity": "ERROR",
            "cwe": "CWE-94"
        },
        "innerHTML-assignment": {
            "pattern": r'\.innerHTML\s*=',
            "message": "Direct innerHTML assignment may lead to XSS vulnerabilities",
            "severity": "WARNING", 
            "cwe": "CWE-79"
        },
        "document-write": {
            "pattern": r'document\.write\s*\(',
            "message": "document.write() can be dangerous and lead to XSS",
            "severity": "WARNING",
            "cwe": "CWE-79"
        },
        "sql-injection-patterns": {
            "pattern": r'SELECT\s+.*\+.*FROM|INSERT\s+.*\+.*INTO|UPDATE\s+.*\+.*SET',
            "message": "Potential SQL injection vulnerability in dynamic query",
            "severity": "ERROR",
            "cwe": "CWE-89"
        }
    }
    
    for i, line in enumerate(lines, 1):
        for rule_id, rule_info in patterns.items():
            if re.search(rule_info["pattern"], line, re.IGNORECASE):
                results.append({
                    "check_id": f"fallback.{rule_id}",
                    "message": rule_info["message"],
                    "path": filename_hint,
                    "start": {"line": i},
                    "end": {"line": i},
                    "extra": {
                        "severity": rule_info["severity"],
                        "metadata": {"cwe": rule_info["cwe"]},
                        "message": rule_info["message"]
                    }
                })
    
    return {
        "ok": True,
        "results": results,
        "stats": {"total_findings": len(results)},
        "fallback": True
    }

MAX_CTX_BYTES = 200_000  # Maximum bytes to read for context

def load_neighbor_context(base_dir: Optional[str], hint_file: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"files": [], "deps": {}}
    if not base_dir or not os.path.isdir(base_dir):
        return out
    seen = 0
    patterns = ["**/*.js", "**/*.ts", "**/package.json", "**/package-lock.json", "**/pnpm-lock.yaml"]
    for pat in patterns:
        for p in glob.glob(os.path.join(base_dir, pat), recursive=True):
            if "node_modules" in p:
                continue
            try:
                b = open(p, "rb").read()
                if seen + len(b) > MAX_CTX_BYTES:
                    continue
                seen += len(b)
                out["files"].append({"path": os.path.relpath(p, base_dir), "bytes": len(b)})
                if os.path.basename(p) == "package.json":
                    try:
                        pkg = json.loads(b.decode("utf-8", "ignore"))
                        out["deps"] = pkg.get("dependencies", {})
                    except Exception:
                        pass
            except Exception:
                continue
    return out


@_timed("web_search")
async def web_search(query: str, limit: int = 5, timeout: int = 12):
    """
    Simple DuckDuckGo search implementation using httpx
    """
    if not AGENT_ENABLE_WEBSEARCH:
        return {"ok": False, "error": "web search disabled", "results": []}
    
    # Use DuckDuckGo instant answer API (no API key required)
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try DuckDuckGo instant answers first
            response = await client.get("https://api.duckduckgo.com/", params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract abstract if available
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("AbstractSource", "DuckDuckGo"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("Abstract", "")
                    })
                
                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:limit-1]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else "Related",
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })
                
                # If no results from instant answers, try a simple search
                if not results:
                    # Fallback: create synthetic results based on query
                    results = await _fallback_security_search(query, limit)
                
                return {"ok": True, "results": results[:limit]}
            
            else:
                # Fallback to security-focused results
                results = await _fallback_security_search(query, limit)
                return {"ok": True, "results": results}
                
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        # Return security-focused fallback results
        results = await _fallback_security_search(query, limit)
        return {"ok": True, "results": results}

async def _fallback_security_search(query: str, limit: int = 5):
    """
    Fallback security-focused search results when external search fails
    """
    security_db = {
        "javascript vulnerabilities": [
            {"title": "OWASP Top 10 for JavaScript", "url": "https://owasp.org/www-project-top-ten/", "snippet": "XSS, injection, broken authentication are common JS vulnerabilities"},
            {"title": "JavaScript Security Best Practices", "url": "https://developer.mozilla.org/en-US/docs/Web/Security", "snippet": "Sanitize inputs, use CSP, validate on server-side"},
        ],
        "xss prevention": [
            {"title": "Cross-Site Scripting Prevention", "url": "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html", "snippet": "Encode outputs, validate inputs, use CSP headers"},
        ],
        "sql injection": [
            {"title": "SQL Injection Prevention", "url": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html", "snippet": "Use parameterized queries, input validation, least privilege"},
        ]
    }
    
    # Simple keyword matching
    query_lower = query.lower()
    results = []
    
    for topic, entries in security_db.items():
        if any(word in query_lower for word in topic.split()):
            results.extend(entries[:limit])
            break
    
    # If no matches, return generic security advice
    if not results:
        results = [
            {"title": "OWASP Security Guidelines", "url": "https://owasp.org", "snippet": "Follow OWASP guidelines for secure coding practices"},
            {"title": "Input Validation", "url": "https://owasp.org", "snippet": "Always validate and sanitize user inputs"},
        ]
    
    return results[:limit]

# ---------------------------------------------------------------------------
# Agent prompts & types
# ---------------------------------------------------------------------------

class Finding(TypedDict, total=False):
    rule_id: str
    severity: str
    file: str
    line: int
    message: str
    cwe: Optional[str]


class ClassifyOutput(TypedDict, total=False):
    is_vulnerable: bool
    categories: List[str]
    cwes: List[str]
    confidence: float
    summary: str


class AgentState(TypedDict, total=False):
    session_id: str
    code: str
    filename: str
    repo_dir: Optional[str]
    iter: int
    max_iters: int
    eslint: Dict[str, Any]
    semgrep: Dict[str, Any]
    context: Dict[str, Any]
    web: Dict[str, Any]
    classify: ClassifyOutput
    fixed_code: str
    done: bool
    reason: str


CLASSIFY_SYS = (
    """
You are a precise code vulnerability classifier.
Return STRICT JSON matching this schema:
{"is_vulnerable": true|false, "categories": ["OWASP-A*"], "cwes": ["CWE-79"], "confidence": 0.0-1.0, "summary": "…"}
Consider ESLint (syntax), Semgrep (CWE), dependencies and optional web intel.
Only output JSON.
"""
)

CLASSIFY_USER_TMPL = """Code (filename: {filename}):

```javascript
{code}
```
Context:

ESLint: {eslint}

Semgrep: {semgrep}

Neighbor/Deps: {context}

Web: {web}
"""

FIX_SYS = (
    """
You are a senior security engineer.
Fix vulnerabilities and syntax issues in the given file. Keep logic & exports compatible.
Return ONLY the complete corrected file content (no backticks, no diff, no extra text).
"""
)

FIX_USER_TMPL = """Original file ({filename}):

{code}

Issues to address:

{issues}

Return ONLY the corrected file content.
"""


def _short_json(data: Any, cap: int = 1200) -> str:
    try:
        s = json.dumps(data)
        return (s[:cap] + "…") if len(s) > cap else s
    except Exception:
        t = str(data)
        return (t[:cap] + "…") if len(t) > cap else t


async def call_model(
    provider: str,
    model_id: str,
    sys_prompt: str,
    user_prompt: str,
    max_tokens: int = 1400,
    temperature: float = 0.0,
) -> str:
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    async with httpx.AsyncClient(timeout=120) as client:
        if provider == "hf_inference":
            pm = PublicModel(id=model_id, provider="hf_inference", label=model_id, stream=False)
            return await provider_hf_inference_chat(client, pm, messages, max_tokens, temperature)
        elif provider == "openrouter":
            pm = PublicModel(id=model_id, provider="openrouter", label=model_id, stream=True)
            return await provider_openrouter_chat(client, pm, messages, max_tokens, temperature)
        else:
            raise RuntimeError(f"Unknown provider {provider}")

# ---------------------------------------------------------------------------
# LangGraph nodes and orchestration
# ---------------------------------------------------------------------------


def ensure_langgraph():
    if not LANGGRAPH_OK:
        raise HTTPException(500, "LangGraph is not installed. pip install langgraph")

def _graph_config(session_id: str) -> dict:
    # Optionally add a namespace if you’ll have multiple graphs
    return {"configurable": {"thread_id": session_id, "checkpoint_ns": "agent-secure"}}


async def node_ingest(state: AgentState) -> AgentState:
    state["iter"] = state.get("iter", 0)
    state["max_iters"] = state.get("max_iters", AGENT_MAX_ITERS)
    return state


async def node_gather(state: AgentState) -> AgentState:
    code = state.get("code", "")
    filename = state.get("filename", "snippet.js")
    repo = state.get("repo_dir")
    tasks: List[asyncio.Task] = []

    if AGENT_ENABLE_ESLINT:
        tasks.append(asyncio.to_thread(run_eslint, code, filename, False, ESLINT_CONFIG_PATH))
    if AGENT_ENABLE_SEMGREP:
        tasks.append(asyncio.to_thread(run_semgrep, code, filename, SEMGREP_CONFIG))

    ctx = load_neighbor_context(repo, filename) if repo else {"files": [], "deps": {}}
    if AGENT_ENABLE_WEBSEARCH:
        q = "common vulnerabilities in " + ", ".join(ctx.get("deps", {}).keys() or ["javascript"])
        tasks.append(web_search(q))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    idx = 0
    if AGENT_ENABLE_ESLINT:
        state["eslint"] = results[idx] if not isinstance(results[idx], Exception) else {"ok": False, "error": str(results[idx])}
        idx += 1
    if AGENT_ENABLE_SEMGREP:
        state["semgrep"] = results[idx] if not isinstance(results[idx], Exception) else {"ok": False, "error": str(results[idx])}
        idx += 1
    state["context"] = ctx
    if AGENT_ENABLE_WEBSEARCH:
        state["web"] = results[idx] if not isinstance(results[idx], Exception) else {"ok": False, "error": str(results[idx])}
    return state


async def node_classify(state: AgentState) -> AgentState:
    user = CLASSIFY_USER_TMPL.format(
        filename=state.get("filename", "snippet.js"),
        code=state.get("code", ""),
        eslint=_short_json(state.get("eslint")),
        semgrep=_short_json(state.get("semgrep")),
        context=_short_json(state.get("context")),
        web=_short_json(state.get("web")),
    )
    try:
        out = await call_model(VULN_CLASSIFIER_PROVIDER, VULN_CLASSIFIER_MODEL_ID, CLASSIFY_SYS, user, temperature=0.0)
        data = json_coerce(out)
    except Exception:
        data = {"is_vulnerable": True, "categories": [], "cwes": [], "confidence": 0.2, "summary": "Classifier JSON parse failed."}
    state["classify"] = data
    return state


async def node_decide(state: AgentState) -> AgentState:
    vul = bool(state.get("classify", {}).get("is_vulnerable"))
    syntax_bad = has_syntax_errors(state.get("eslint", {}))
    if (vul or syntax_bad) and state.get("iter", 0) < state.get("max_iters", AGENT_MAX_ITERS):
        state["done"] = False
    else:
        state["done"] = True
        state["reason"] = "clean" if not (vul or syntax_bad) else "max_iters_reached"
    return state


async def node_fix(state: AgentState) -> AgentState:
    issues: List[str] = []
    if state.get("classify"):
        issues.append(f"classifier: {_short_json(state['classify'])}")
    if state.get("eslint"):
        issues.append(f"eslint: {_short_json(state['eslint'])}")
    if state.get("semgrep"):
        issues.append(f"semgrep: {_short_json(state['semgrep'])}")

    user = FIX_USER_TMPL.format(filename=state.get("filename", "snippet.js"), code=state.get("code", ""), issues="\n- ".join(issues))
    fixed = await call_model(VULN_FIXER_PROVIDER, VULN_FIXER_MODEL_ID, FIX_SYS, user, temperature=0.1, max_tokens=1600)

    # strip surrounding fences or language tags if model inserted them
    fixed = re.sub(r"^```[a-zA-Z]*\n?", "", fixed).strip()
    fixed = re.sub(r"```$", "", fixed).strip()

    # Post-fix polish with eslint --fix
    eslint_after = run_eslint(fixed, state.get("filename", "snippet.js"), fix=True, config_path=ESLINT_CONFIG_PATH)
    state["fixed_code"] = eslint_after.get("fixed_code") or fixed

    # Re-scan
    state["eslint"] = run_eslint(state["fixed_code"], state.get("filename", "snippet.js"), fix=False, config_path=ESLINT_CONFIG_PATH)
    state["semgrep"] = run_semgrep(state["fixed_code"], state.get("filename", "snippet.js"), SEMGREP_CONFIG)
    state["iter"] = state.get("iter", 0) + 1
    return state


async def node_verify(state: AgentState) -> AgentState:
    syntax_bad = has_syntax_errors(state.get("eslint", {}))
    sem_findings = (state.get("semgrep", {}).get("results") or [])
    if not syntax_bad and len(sem_findings) == 0:
        state["done"] = True
        state["reason"] = "clean"
        if METRICS_ENABLED:
            agent_iterations.labels("clean").inc()
    else:
        state["done"] = state.get("iter", 0) >= state.get("max_iters", AGENT_MAX_ITERS)
        if state.get("done"):
            state["reason"] = f"residual_issues (syntax={syntax_bad}, semgrep={len(sem_findings)})"
            if METRICS_ENABLED:
                agent_iterations.labels("residual").inc()
    return state


def build_graph():
    ensure_langgraph()
    sg = StateGraph(AgentState)
    sg.add_node("ingest", node_ingest)
    sg.add_node("gather", node_gather)
    sg.add_node("classify_step", node_classify)
    sg.add_node("decide", node_decide)
    sg.add_node("fix", node_fix)
    sg.add_node("verify", node_verify)

    sg.set_entry_point("ingest")
    sg.add_edge("ingest", "gather")
    sg.add_edge("gather", "classify_step")
    sg.add_edge("classify_step", "decide")
    sg.add_edge("fix", "verify")

    def decide_fix(state: AgentState):
        return "fix" if not state.get("done") else "verify"

    sg.add_conditional_edges("decide", decide_fix, {"fix": "fix", "verify": "verify"})

    def loop_or_end(state: AgentState):
        return END if state.get("done") else "classify_step"

    sg.add_conditional_edges("verify", loop_or_end, {END: END, "classify_step": "classify_step"})

    memory = MemorySaver()
    return sg.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# FastAPI app & endpoints
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FastAPI LLM Service starting...")
    logger.info("🧩 Models: %s", [m.label for m in ALLOWED_MODELS])
    logger.info("🛠️ Agent tools: ESLint=%s, Semgrep=%s, Web=%s", AGENT_ENABLE_ESLINT, AGENT_ENABLE_SEMGREP, AGENT_ENABLE_WEBSEARCH)
    global AGENT_GRAPH
    AGENT_GRAPH = build_graph() if LANGGRAPH_OK else None
    yield
    logger.info("🛑 Shutdown")


app = FastAPI(title="LLM Chatbot API (Multi-LLM + Agentic Security)", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {
        "message": "LLM Chatbot API is running",
        "default_model_id": DEFAULT_MODEL_ID,
        "default_provider": DEFAULT_PROVIDER,
        "active_sessions": len(chat_sessions),
        "agent_enabled": LANGGRAPH_OK,
    }


@app.get("/models", response_model=List[PublicModel])
async def list_models():
    return ALLOWED_MODELS

# --------------------- Core chat (sync + streaming) ------------------------


def log_chat_response(session_id: str, user_message: str, ai_response: str, model_id: str, provider: str, response_time: Optional[float] = None):
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info(f"🤖 CHAT RESPONSE GENERATED")
    logger.info(f"🔑 Session ID: {session_id}")
    logger.info(f"🧠 Model: {model_id} ({provider})")
    if response_time is not None:
        logger.info(f"⏱️ Response Time: {response_time:.2f}s")
    logger.info(f"👤 User Message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
    logger.info(f"🤖 AI Response: {ai_response[:200]}{'...' if len(ai_response) > 200 else ''}")
    logger.info(f"{separator}\n")


def log_stream_start(session_id: str, user_message: str, model_id: str, provider: str):
    logger.info(f"🔄 STREAM START - Session: {session_id} | Model: {model_id} ({provider}) | Message: {user_message[:60]}...")


def log_stream_complete(session_id: str, total_tokens: int, response_time: Optional[float] = None):
    tinfo = f" | Time: {response_time:.2f}s" if response_time else ""
    logger.info(f"✅ STREAM END - Session: {session_id} | Tokens: {total_tokens}{tinfo}")

# --- Reasoning & reporting helpers ------------------------------------------

def _first(items, n):
    return items[:n] if isinstance(items, list) else []

def summarize_eslint(eslint_json: dict, limit: int = 5) -> str:
    if not eslint_json or "results" not in eslint_json:
        return "- None"
    msgs = []
    for f in _first(eslint_json.get("results", []), limit):
        file_path = f.get("filePath", "file")
        for m in _first(f.get("messages", []), limit):
            sev = {2: "error", 1: "warn"}.get(m.get("severity", 0), "info")
            rule = m.get("ruleId") or "syntax"
            line = m.get("line") or m.get("endLine") or "?"
            text = (m.get("message") or "").strip()
            msgs.append(f"- {file_path}:{line} [{sev}] {rule}: {text}")
            if len(msgs) >= limit:
                break
    return "\n".join(msgs) if msgs else "- None"

def summarize_semgrep(semgrep_json: dict, limit: int = 5) -> str:
    results = (semgrep_json or {}).get("results", []) or []
    if not results:
        return "- None"
    rows = []
    for r in _first(results, limit):
        rid = r.get("check_id") or (r.get("extra", {}) or {}).get("ruleId") or "rule"
        sev = ((r.get("extra", {}) or {}).get("severity") or "").upper()
        msg = ((r.get("extra", {}) or {}).get("message") or r.get("message") or "").strip()
        meta = (r.get("extra", {}) or {}).get("metadata") or {}
        cwe = meta.get("cwe")
        line = (r.get("start", {}) or {}).get("line") or (r.get("start", {}) or {}).get("line_start") or "?"
        cwe_txt = f" ({cwe})" if cwe else ""
        rows.append(f"- {rid}{cwe_txt} [{sev}] line {line}: {msg[:120]}{'…' if len(msg) > 120 else ''}")
    return "\n".join(rows) if rows else "- None"

def short_diff(before: str, after: str, max_lines: int = 14) -> str:
    try:
        diff = difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile="before", tofile="after", lineterm=""
        )
        lines = list(diff)
        if not lines:
            return "- (Only formatting/whitespace changes)"
        # keep header + first hunks
        head = lines[:max_lines]
        if len(lines) > max_lines:
            head.append("… (diff truncated)")
        return "\n".join(head)
    except Exception:
        return "- (diff unavailable)"

def collect_vuln_types(classify: dict, semgrep_json: dict) -> str:
    cats = (classify or {}).get("categories", []) or []
    cwes = (classify or {}).get("cwes", []) or []
    # also grab CWEs from semgrep metadata if present
    for r in (semgrep_json or {}).get("results", []) or []:
        meta = (r.get("extra", {}) or {}).get("metadata") or {}
        cwe = meta.get("cwe")
        if cwe and cwe not in cwes:
            cwes.append(cwe)
    parts = []
    if cats: parts.append("OWASP: " + ", ".join(cats))
    if cwes: parts.append("CWE: " + ", ".join(cwes))
    return " | ".join(parts) if parts else "-"

def has_syntax_errors(eslint_json: dict) -> bool:
    """
    Returns True if ESLint results contain fatal errors (severity 2 or fatal flag).
    """
    if not eslint_json or "results" not in eslint_json:
        return False
    for file_result in eslint_json.get("results", []):
        for msg in file_result.get("messages", []):
            if msg.get("fatal") or msg.get("severity", 0) == 2:
                return True
    return False

async def generate_recommendations(classify: dict, eslint_json: dict, semgrep_json: dict, deps: dict, final_code: str) -> str:
    """
    Uses the classifier model to produce concise, actionable recommendations.
    Keeps latency low and tokens bounded.
    """
    try:
        sys_p = (
            "You are a secure code advisor. Given the analysis context, output 5–8 short, "
            "actionable recommendations. Be concrete (inputs, sanitization, auth, dependency pinning, "
            "headers, CSP, rate limiting, logging). No code unless critical; each bullet <= 20 words."
        )
        ctx = {
            "classify": classify,
            "eslint_top": summarize_eslint(eslint_json, 5),
            "semgrep_top": summarize_semgrep(semgrep_json, 5),
            "dependencies": deps or {},
        }
        usr_p = "Context:\n" + json.dumps(ctx, ensure_ascii=False) + "\n\nFinal code (trimmed):\n" + final_code[:2000]
        # Reuse classifier model for advice
        text = await call_model(
            VULN_CLASSIFIER_PROVIDER, VULN_CLASSIFIER_MODEL_ID,
            sys_p, usr_p, temperature=0.2, max_tokens=400
        )
        return text.strip() or "-"
    except Exception:
        # Safe fallback
        return (
            "- Validate & sanitize all inputs (whitelist patterns)\n"
            "- Escape outputs & use templating safely\n"
            "- Enforce authZ checks on sensitive routes\n"
            "- Pin and update vulnerable dependencies\n"
            "- Add security headers (CSP, HSTS, X-Frame-Options)\n"
            "- Implement rate limiting & request size caps\n"
            "- Centralize error handling; avoid leaking stack traces"
        )

# --- LangGraph config helper (thread_id for checkpointer) ------------------
def _graph_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id, "checkpoint_ns": "agent-secure"}}



# Replace the chat endpoints in main.py with these implementations

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    """
    Enhanced chat with proper session history integration
    """
    start_time = time.perf_counter()
    
    # Use provided session_id or create new one
    sid = req.session_id or create_session_id()
    
    # Build conversation history from messages array (sent from Node.js backend)
    history = req.messages or []
    
    # --- Agentic path if code is present ---
    snippet, lang = extract_code_and_lang(req.message)
    if snippet:
        if not LANGGRAPH_OK or AGENT_GRAPH is None:
            raise HTTPException(500, "LangGraph not available. Install with `pip install langgraph`.")
        filename = guess_filename(lang)
        state: AgentState = {
            "session_id": sid,
            "code": snippet,
            "filename": filename,
            "max_iters": AGENT_MAX_ITERS,
        }
        
        out = await AGENT_GRAPH.ainvoke(state, config=_graph_config(sid))

        eslint = out.get("eslint", {}) or {}
        semg = out.get("semgrep", {}) or {}
        cls = out.get("classify", {}) or {}
        final_code = (out.get("fixed_code") or out.get("code") or "").strip()
        iters = int(out.get("iter", 0))
        reason = out.get("reason", "")
        syntax_bad = has_syntax_errors(eslint)
        sem_count = len((semg.get("results") or []))
        clean = (not syntax_bad) and (sem_count == 0)

        # Build comprehensive response with context awareness
        steps = []
        steps.append("1) Syntax check (ESLint): " + ("fatal errors found" if syntax_bad else "no fatal errors"))
        steps.append("   Top issues:\n" + summarize_eslint(eslint, 5))
        steps.append(f"2) Semgrep scan: {sem_count} finding(s)")
        steps.append("   Top rules:\n" + summarize_semgrep(semg, 5))
        if cls:
            conf = cls.get("confidence", 0.0)
            steps.append(
                f"3) LLM classification: {'VULNERABLE' if cls.get('is_vulnerable') else 'SECURE'}"
            )
            if cls.get("summary"):
                steps.append(f"   Summary: {cls.get('summary')[:400]}")
        if iters > 0 and final_code:
            steps.append(f"4) Fix applied in {iters} iteration(s). Key changes (trimmed diff):\n" +
                         short_diff(out.get("code", ""), final_code, 14))
        steps.append(
            f"5) Post-fix verification: {'clean ✅' if clean else 'residual issues ⚠️'} "
            f"(eslint_fatal={syntax_bad}, semgrep={sem_count})"
        )
        reasoning_block = "\n".join(steps)

        vuln_types = collect_vuln_types(cls, semg)
        deps = (out.get("context", {}) or {}).get("deps", {}) if isinstance(out.get("context"), dict) else {}
        recs = await generate_recommendations(cls, eslint, semg, deps, final_code or snippet)

        header = "✅ Code appears secure after fixes" if clean else "⚠️ Code is VULNERABLE"
        parts = [
            f"{header}",
            f"Reason: {reason or ('clean' if clean else 'issues remain')}",
            f"Iterations: {iters}",
            f"Vulnerability Types: {vuln_types}",
            "",
            "### Reasoning Steps",
            reasoning_block,
            "",   
        ]

        if final_code:
            parts += [
                "",
                "### Final Fixed Code",
                f"```{(lang or 'javascript')}\n{final_code}\n```",
            ]

        parts +=["### Additional Recommendations", recs,]
        text = "\n".join(parts)

        total = len(history) + 2
        response_time = time.perf_counter() - start_time
        log_chat_response(sid, req.message, text, "agent-secure", "agent", response_time)
        return ChatResponse(response=text, session_id=sid, message_count=total, model_id="agent-secure", provider="agent")

    # --- Regular chat with conversation history ---
    # Build context-aware conversation including history
    llm_messages = build_conversation_context_with_history(req.message, history)
    model = resolve_model(req.model_id, req.provider)
    
    if model.provider == "openrouter":
        text = await provider_openrouter_chat(client, model, llm_messages, req.max_tokens or 1000, req.temperature or 0.7)
    elif model.provider == "hf_inference":
        text = await provider_hf_inference_chat(client, model, llm_messages, req.max_tokens or 1000, req.temperature or 0.7)
    else:
        raise HTTPException(400, f"Unknown provider: {model.provider}")

    total = len(history) + 2
    response_time = time.perf_counter() - start_time
    log_chat_response(sid, req.message, text, model.id, model.provider, response_time)
    return ChatResponse(response=text, session_id=sid, message_count=total, model_id=model.id, provider=model.provider)

def build_conversation_context_with_history(current_message: str, history: List[Message]) -> List[Dict[str, Any]]:
    """
    Build conversation context from history messages for LLM
    """
    llm_messages = []
    
    # Add conversation history
    for msg in history:
        llm_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Add current message
    llm_messages.append({
        "role": "user", 
        "content": current_message
    })
    
    # Limit context length to prevent token overflow
    MAX_CONTEXT_MESSAGES = 20
    if len(llm_messages) > MAX_CONTEXT_MESSAGES:
        # Keep system messages at start (if any) and recent messages
        system_msgs = [msg for msg in llm_messages if msg["role"] == "system"]
        other_msgs = [msg for msg in llm_messages if msg["role"] != "system"]
        
        # Keep most recent messages
        recent_msgs = other_msgs[-(MAX_CONTEXT_MESSAGES - len(system_msgs)):]
        llm_messages = system_msgs + recent_msgs
    
    return llm_messages

async def _dispatch_stream(client: httpx.AsyncClient, model: PublicModel, messages: List[Dict[str, Any]], max_tokens: int, temperature: float):
    if model.provider == "openrouter":
        async for chunk in provider_openrouter_stream(client, model, messages, max_tokens, temperature):
            yield chunk
    elif model.provider == "hf_inference":
        async for chunk in provider_hf_inference_stream(client, model, messages, max_tokens, temperature):
            yield chunk
    else:
        yield f"event: error\ndata: {json.dumps({'detail':'Unknown provider'})}\n\n"

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    """
    Enhanced streaming chat with proper session history
    """
    start_time = time.perf_counter()
    
    # Use provided session_id or create new one
    sid = req.session_id or create_session_id()
    
    # Build conversation history from messages array
    history = req.messages or []
    llm_messages = build_conversation_context_with_history(req.message, history)
    model = resolve_model(req.model_id, req.provider)

    log_stream_start(sid, req.message, model.id, model.provider)
    collected_tokens: List[str] = []
    token_count = 0

    async def gen():
        nonlocal token_count
        async for sse in _dispatch_stream(client, model, llm_messages, req.max_tokens or 1000, req.temperature or 0.7):
            if sse.startswith("event: token"):
                try:
                    data_part = sse.split("data: ")[1].strip()
                    token = json.loads(data_part).get("token", "")
                    if token:
                        collected_tokens.append(token)
                        token_count += 1
                except Exception:
                    pass
            yield sse
            
        full_response = "".join(collected_tokens).strip()
        response_time = time.perf_counter() - start_time
        if full_response:
            log_chat_response(sid, req.message, full_response, model.id, model.provider, response_time)
        log_stream_complete(sid, token_count, response_time)
        total = len(history) + 2
        yield f"event: meta\ndata: {json.dumps({'session_id': sid, 'message_count': total, 'model_id': model.id, 'provider': model.provider})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", 
                           headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

# --------------------- Agent endpoints ------------------------------------

@app.post("/agent/analyze", response_model=ChatResponse)
async def agent_analyze(req: ChatRequest):
    if not LANGGRAPH_OK:
        raise HTTPException(500, "LangGraph not available. Install with pip install langgraph.")
    sid = get_or_create_session(req.session_id)
    code = req.message or ""
    filename = "snippet.js"
    state: AgentState = {"session_id": sid, "code": code, "filename": filename, "max_iters": 1}

    out = await AGENT_GRAPH.ainvoke(state, config=_graph_config(sid))

    cls = out.get("classify", {}) or {}
    eslint = out.get("eslint", {}) or {}
    semg = out.get("semgrep", {}) or {}
    is_v = bool(cls.get("is_vulnerable")) or has_syntax_errors(eslint) or len((semg.get("results") or [])) > 0
    summary = cls.get("summary") or "No summary."
    vuln_types = collect_vuln_types(cls, semg)
    deps = (out.get("context", {}) or {}).get("deps", {}) if isinstance(out.get("context"), dict) else {}
    recs = await generate_recommendations(cls, eslint, semg, deps, code)

    final_text = (
        f"Vulnerable: {is_v}\n"
        f"Vulnerability Types: {vuln_types}\n"
        f"Confidence: {cls.get('confidence', 0):.2f}\n"
        f"Categories: {', '.join(cls.get('categories', [])) or '-'}\n"
        f"CWEs: {', '.join(cls.get('cwes', [])) or '-'}\n\n"
        f"Summary: {summary}\n\n"
        f"ESLint findings: {len((eslint or {}).get('results', []))}\n"
        f"Semgrep findings: {len((semg or {}).get('results', []))}\n\n"
        f"Top ESLint issues:\n{summarize_eslint(eslint, 5)}\n\n"
        f"Top Semgrep rules:\n{summarize_semgrep(semg, 5)}\n\n"
        f"Recommendations:\n{recs}"
    )

    add_message_to_session(sid, Message(role="user", content=req.message))
    add_message_to_session(sid, Message(role="assistant", content=final_text))
    return ChatResponse(
        response=final_text,
        session_id=sid,
        message_count=len(chat_sessions[sid]),
        model_id="agent-classifier",
        provider="agent"
    )

@app.post("/agent/secure", response_model=ChatResponse)
async def agent_secure(req: ChatRequest):
    if not LANGGRAPH_OK:
        raise HTTPException(500, "LangGraph not available. Install with pip install langgraph.")
    sid = get_or_create_session(req.session_id)
    code = req.message or ""
    filename = "snippet.js"
    start = time.perf_counter()
    state: AgentState = {"session_id": sid, "code": code, "filename": filename, "max_iters": AGENT_MAX_ITERS}
    out = await AGENT_GRAPH.ainvoke(state, config=_graph_config(sid))
    dur = time.perf_counter() - start

    eslint = out.get("eslint", {})
    semg = out.get("semgrep", {})
    clean = not has_syntax_errors(eslint) and len((semg or {}).get("results", [])) == 0

    report = [
        f"Secure: {clean}",
        f"Reason: {out.get('reason','')}",
        f"Iterations: {out.get('iter',0)} in {dur:.2f}s",
        "",
        "=== Final Code ===",
        (out.get("fixed_code") or out.get("code") or "").strip(),
        "",
        "=== Last ESLint ===",
        json.dumps(eslint, indent=2)[:2000],
        "",
        "=== Last Semgrep ===",
        json.dumps(semg, indent=2)[:2000],
    ]

    resp_text = "\n".join(report)
    add_message_to_session(sid, Message(role="user", content=req.message))
    add_message_to_session(sid, Message(role="assistant", content=resp_text))
    return ChatResponse(response=resp_text, session_id=sid, message_count=len(chat_sessions[sid]), model_id="agent-secure", provider="agent")

# --------------------- Sessions -------------------------------------------


@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    out: List[SessionInfo] = []
    for sid, meta in session_metadata.items():
        out.append(
            SessionInfo(session_id=sid, message_count=len(chat_sessions.get(sid, [])), created_at=meta["created_at"], last_updated=meta["last_updated"])
        )
    return out


@app.get("/sessions/{session_id}/history", response_model=List[Message])
async def get_session_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    return chat_sessions[session_id]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    del chat_sessions[session_id]
    del session_metadata[session_id]
    logger.info(f"🗑️ Session deleted: {session_id}")
    return {"message": f"Session {session_id} deleted"}


@app.delete("/sessions")
async def clear_all_sessions():
    n = len(chat_sessions)
    chat_sessions.clear()
    session_metadata.clear()
    logger.info(f"🗑️ All sessions cleared: {n} sessions")
    return {"message": f"Cleared {n} sessions"}


@app.post("/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(404, "Session not found")
    cnt = len(chat_sessions[session_id])
    chat_sessions[session_id] = []
    session_metadata[session_id]["last_updated"] = datetime.now()
    logger.info(f"🗑️ Session history cleared: {session_id} ({cnt} messages)")
    return {"message": f"Cleared {cnt} messages from session {session_id}"}

# --------------------- Metrics --------------------------------------------

if METRICS_ENABLED:

    @app.get("/metrics")
    async def metrics():
        return StreamingResponse(iter([generate_latest()]), media_type=CONTENT_TYPE_LATEST)

# --------------------- Dev entrypoint ------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)