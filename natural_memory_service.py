"""Local production API for a single-user Natural Memory v2 instance.

The service binds to localhost by default, keeps one model lock so concurrent
requests cannot corrupt the model-owned memory, and persists changed memory
back into the embedded third safetensors shard when enabled.
"""

from __future__ import annotations

import argparse
import gc
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .qwen_integration import compose_corrected_evidence, load_qwen_dynamic, load_tokenizer
from .stream_chat_qwen_memory import _chat_tensor, _persist_memory, _write_turn


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


class NaturalMemoryService:
    def __init__(
        self,
        model_path: str | Path,
        *,
        no_4bit: bool = False,
        auto_persist: bool = True,
        auth_token: str | None = None,
    ) -> None:
        self.model_path = _project_path(model_path)
        self.auth_token = auth_token
        self.auto_persist = bool(auto_persist)
        self.lock = threading.RLock()
        self.tokenizer = load_tokenizer(self.model_path)
        self.model = load_qwen_dynamic(self.model_path, load_in_4bit=not no_4bit)
        self.model.eval()
        self.device = self.model._find_layer_device()
        if self.model.memory_os_v2 is None:
            raise RuntimeError("the selected package does not contain hierarchical memory")

    def close(self) -> None:
        with self.lock:
            self.model.close_memory_storage()
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _persist(self) -> None:
        if not self.auto_persist:
            return
        _persist_memory(self.model, embedded_dir=self.model_path, state_path=None)

    def _encode_plain(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"].to(self.device)
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        return ids, mask.to(self.device)

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": "ok",
                "model_path": str(self.model_path),
                "device": str(self.device),
                "auto_persist": self.auto_persist,
                "memory": self.model.memory_v2_stats(),
                "audit": self.model.audit_memory(),
                "cuda": {
                    "available": torch.cuda.is_available(),
                    "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                    if torch.cuda.is_available()
                    else None,
                    "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
                    if torch.cuda.is_available()
                    else None,
                },
            }

    def list_memory(self, params: dict[str, list[str]]) -> dict[str, Any]:
        with self.lock:
            query = params.get("query", [""])[0]
            status = params.get("status", ["active"])[0]
            limit = min(10000, max(1, int(params.get("limit", ["100"])[0])))
            offset = max(0, int(params.get("offset", ["0"])[0]))
            records = self.model.list_memory_records(
                query_text=query, status=status, limit=limit, offset=offset
            )
            return {"records": records, "returned": len(records), "offset": offset, "limit": limit}

    def get_memory(self, record_id: str) -> dict[str, Any]:
        with self.lock:
            return self.model.get_memory_record(record_id)

    def export_memory(self, params: dict[str, list[str]]) -> dict[str, Any]:
        with self.lock:
            limit = min(10000, max(1, int(params.get("limit", ["10000"])[0])))
            offset = max(0, int(params.get("offset", ["0"])[0]))
            return self.model.export_memory_records(limit=limit, offset=offset)

    def audit(self) -> dict[str, Any]:
        with self.lock:
            return self.model.audit_memory()

    def edit_memory(self, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock, torch.inference_mode():
            text = payload.get("text")
            entity = payload.get("entity")
            attribute = payload.get("attribute")
            value = payload.get("value")
            if not (isinstance(text, str) and text.strip()):
                # The model reads the record's stored evidence card, not its fields, so a
                # correction that only supplies new fields left the card asserting the old
                # value: the call reported success while every answer stayed stale.  Rebuild
                # the card from the fields actually being changed; the token cache follows
                # because it is encoded from this text just below.
                rebuilt = compose_corrected_evidence(
                    self.model.get_memory_record(record_id),
                    entity=entity if isinstance(entity, str) else None,
                    attribute=attribute if isinstance(attribute, str) else None,
                    value=value if isinstance(value, str) else None,
                )
                if rebuilt is not None:
                    text = rebuilt
            ids = mask = None
            if isinstance(text, str) and text.strip():
                ids, mask = self._encode_plain(text)
            result = self.model.edit_memory_record(
                record_id,
                text=text if isinstance(text, str) else None,
                entity=entity,
                attribute=attribute,
                value=value,
                importance=payload.get("importance"),
                confidence=payload.get("confidence"),
                evidence=payload.get("evidence") if isinstance(payload.get("evidence"), list) else None,
                token_ids=ids,
                token_mask=mask,
            )
            self._persist()
            return result

    def retract_memory(self, record_id: str) -> dict[str, Any]:
        with self.lock:
            result = self.model.retract_memory_record(record_id)
            self._persist()
            return result

    def reset_memory(self) -> dict[str, Any]:
        with self.lock:
            self.model.reset_memory(batch_size=1, device=self.device)
            self._persist()
            return {"reset": True, "memory": self.model.memory_v2_stats()}

    def direct_write(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("text is required")
        with self.lock, torch.inference_mode():
            ids, mask = self._encode_plain(text)
            key = self.model._encode_model_key(ids, mask)[0]
            record, action = self.model.write_hierarchical_memory(
                text=text,
                key=key,
                summary=key,
                token_ids=ids[0].detach().cpu(),
                token_mask=mask[0].detach().cpu().bool(),
                entity=str(payload.get("entity", "")),
                attribute=str(payload.get("attribute", "")),
                value=str(payload.get("value", "")),
                importance=float(payload.get("importance", 0.9)),
                confidence=float(payload.get("confidence", 0.99)),
                source="api",
                trusted=True,
                force=bool(payload.get("force", True)),
            )
            self._persist()
            return {"action": action, "record": self.model.get_memory_record(record.record_id)}

    def _prepare_chat(self, message: str) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, bool]:
        encoded = {key: value.to(self.device) for key, value in _chat_tensor(self.tokenizer, message).items()}
        query_ids, query_mask = self._encode_plain(message)
        reset_id = self.model.memory_config.reset_token_id
        if reset_id is not None and bool((encoded["input_ids"] == reset_id).any()):
            self.model.reset_memory(batch_size=1, device=self.device)
            self._persist()
            return encoded, query_ids, query_mask, True
        changed = False
        if self.model.memory_config.native_mode:
            changed = _write_turn(self.model, self.tokenizer, message, self.device)
            if changed:
                self._persist()
        return encoded, query_ids, query_mask, changed

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message", "")).strip()
        if not message:
            raise ValueError("message is required")
        max_new_tokens = min(256, max(1, int(payload.get("max_new_tokens", 128))))
        with self.lock, torch.inference_mode():
            encoded, query_ids, query_mask, changed = self._prepare_chat(message)
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                update_memory=False,
                memory_query_input_ids=query_ids,
                memory_query_attention_mask=query_mask,
                memory_query_text=message,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            answer = self.tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            return {
                "answer": answer,
                "memory_changed": changed,
                "memory": self.model.memory_v2_stats(),
            }

    def stream_chat(self, payload: dict[str, Any]):
        """Yield answer fragments while holding the single-model lock."""

        from transformers import TextIteratorStreamer

        message = str(payload.get("message", "")).strip()
        if not message:
            raise ValueError("message is required")
        max_new_tokens = min(256, max(1, int(payload.get("max_new_tokens", 128))))
        self.lock.acquire()
        try:
            encoded, query_ids, query_mask, changed = self._prepare_chat(message)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    self.model.generate(
                        **encoded,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        update_memory=False,
                        memory_query_input_ids=query_ids,
                        memory_query_attention_mask=query_mask,
                        memory_query_text=message,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                except BaseException as error:
                    errors.append(error)
                    streamer.on_finalized_text("", stream_end=True)

            thread = threading.Thread(target=worker, name="natural-memory-api-generation", daemon=True)
            thread.start()
            yield {"type": "meta", "memory_changed": changed}
            for chunk in streamer:
                yield {"type": "token", "text": chunk}
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("generation thread did not stop")
            if errors:
                raise RuntimeError("stream generation failed") from errors[0]
            yield {"type": "done", "memory": self.model.memory_v2_stats()}
        finally:
            self.lock.release()


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[natural-memory] {self.address_string()} {format % args}")

    @property
    def service(self) -> NaturalMemoryService:
        return self.server.service

    def _authorized(self) -> bool:
        expected = self.service.auth_token
        if not expected:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {expected}"

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, error: BaseException, status: int = 400) -> None:
        self._send_json({"error": type(error).__name__, "message": str(error)}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("request body exceeds 1 MiB")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send_json(self.service.health())
            elif parsed.path == "/v1/memory":
                self._send_json(self.service.list_memory(parse_qs(parsed.query)))
            elif parsed.path.startswith("/v1/memory/export"):
                self._send_json(self.service.export_memory(parse_qs(parsed.query)))
            elif parsed.path == "/v1/memory/audit":
                self._send_json(self.service.audit())
            elif parsed.path.startswith("/v1/memory/"):
                self._send_json(self.service.get_memory(parsed.path.rsplit("/", 1)[1]))
            else:
                self._send_json({"error": "not_found"}, 404)
        except KeyError as error:
            self._send_error_json(error, 404)
        except Exception as error:
            self._send_error_json(error, 400)

    def do_POST(self) -> None:
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        try:
            payload = self._body()
            parsed = urlparse(self.path)
            if parsed.path == "/v1/chat":
                if bool(payload.get("stream", False)):
                    self._send_stream(self.service.stream_chat(payload))
                else:
                    self._send_json(self.service.chat(payload))
            elif parsed.path == "/v1/memory":
                self._send_json(self.service.direct_write(payload), 201)
            elif parsed.path == "/v1/memory/reset":
                self._send_json(self.service.reset_memory())
            elif parsed.path.startswith("/v1/memory/"):
                self._send_json(self.service.edit_memory(parsed.path.rsplit("/", 1)[1], payload))
            else:
                self._send_json({"error": "not_found"}, 404)
        except KeyError as error:
            self._send_error_json(error, 404)
        except Exception as error:
            self._send_error_json(error, 400)

    def do_DELETE(self) -> None:
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        try:
            path = urlparse(self.path).path
            if not path.startswith("/v1/memory/"):
                self._send_json({"error": "not_found"}, 404)
                return
            self._send_json(self.service.retract_memory(path.rsplit("/", 1)[1]))
        except KeyError as error:
            self._send_error_json(error, 404)
        except Exception as error:
            self._send_error_json(error, 400)

    def _send_stream(self, events) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event in events:
                body = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {body}\n\n".encode("utf-8"))
                self.wfile.flush()
        except Exception as error:
            body = json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False)
            self.wfile.write(f"data: {body}\n\n".encode("utf-8"))


class _Server(ThreadingHTTPServer):
    def __init__(self, address, service: NaturalMemoryService):
        super().__init__(address, _Handler)
        self.service = service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="qwen3_5_4b_natural_memory_v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--no-auto-persist", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    service = NaturalMemoryService(
        args.model_path,
        no_4bit=args.no_4bit,
        auto_persist=not args.no_auto_persist,
        auth_token=args.auth_token,
    )
    server = _Server((args.host, args.port), service)
    print(f"Natural Memory API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Natural Memory API")
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
