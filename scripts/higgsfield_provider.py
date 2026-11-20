#!/usr/bin/env python3
"""
Higgsfield backend for the provider layer — drives the `higgsfield` CLI
(https://github.com/higgsfield-ai/cli; log in once with `higgsfield auth login`).

The stage scripts keep speaking Atlas Cloud model ids + Atlas-shaped params. This
provider translates both: each Atlas id maps to the closest Higgsfield job type, and
each param is renamed / converted to a CLI flag, then filtered against that job
type's live schema (`higgsfield model get <job_type> --json`) so nothing unknown is
sent. A bare Higgsfield job type (no "/") is passed straight through.

Select it per project with beats.json `{"provider": "higgsfield"}`.

Two behaviours differ from Atlas on purpose:
  * resubmit_stalled = False — Higgsfield jobs routinely sit "queued" for 1-2 min
    before starting. Resubmitting a slow job would bill it twice, so run_jobs only
    resubmits jobs that actually FAIL.
  * deadline_factor = 3 — the stage deadlines were tuned for Atlas latency.

Voices: Higgsfield has its own roster (`higgsfield voices list`). Put a Higgsfield
voice name or id in beats.json voice.voice_id; an unknown id (e.g. an xAI one)
falls back to HF_VOICE env or the default narrator below.
"""
import base64
import json
import os
import re
import subprocess
import tempfile

from provider import Provider

CLI =os.environ.get("HIGGSFIELD_CLI", "higgsfield")
DEFAULT_VOICE = os.environ.get("HF_VOICE", "Arthur")
TTS_VARIANT = os.environ.get("HF_TTS_VARIANT", "elevenlabs")

# Atlas model id (substring, first match wins) -> Higgsfield job type + fixed extras.
MODEL_MAP = [
    ("nano-banana-2/text-to-image",       "nano_banana_flash", {}),
    ("nano-banana-2/edit",                "nano_banana_flash", {}),
    ("nano-banana",                       "nano_banana_flash", {}),
    ("gemini-omni-flash",                 "gemini_omni", {}),
    ("seedance-2.0",                      "seedance_2_0", {}),
    ("kling-video-o3-pro/image-to-video", "kling3_0", {"mode": "pro"}),
    ("kling-video-o3-pro/reference",      "kling_o3_image_reference", {"mode": "pro", "multi_shots": False}),
    ("kling-video-o3-pro/video-edit",     "kling_video_edit", {}),
    ("kling",                             "kling3_0", {"mode": "pro"}),
    ("xai/tts",                           "text2speech_v2", {}),
    ("seed-audio",                        "seed_audio", {}),
    ("music",                             "elevenlabs_music_v2_5", {}),
    ("remove-background",                 "image_background_remover", {}),
]

# Atlas param name -> CLI media flag (values are URLs / paths / ids, maybe lists).
MEDIA_FLAGS = {
    "image": "--image-references", "images": "--image-references",
    "reference_images": "--image-references",
    "video": "--video-references", "reference_videos": "--video-references",
    "reference_audios": "--audio-references",
    "start_image": "--start-image", "end_image": "--end-image",
}
# Job types whose single `image` is the first frame, not a loose reference.
START_FRAME_MODELS = {"kling3_0", "seedance_2_0"}

RENAME = {"ratio": "aspect_ratio", "is_instrumental": "force_instrumental",
          "text": "prompt"}

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_schemas, _voices = {}, None


class HiggsfieldError(RuntimeError):
    pass


def _cli(*args):
    r = subprocess.run([CLI, *args, "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise HiggsfieldError((r.stderr or r.stdout).strip()[:500])
    return json.loads(r.stdout)


def _schema(job_type):
    if job_type not in _schemas:
        _schemas[job_type] = {p["name"] for p in _cli("model", "get", job_type)["params"]}
    return _schemas[job_type]


def _job_type(model):
    if "/" not in model:
        return model, {}
    for key, jt, extra in MODEL_MAP:
        if key in model:
            return jt, dict(extra)
    raise HiggsfieldError(f"no Higgsfield mapping for '{model}' — add it to MODEL_MAP "
                          f"in higgsfield_provider.py or pass a Higgsfield job type")


def _voice(name_or_id):
    """Higgsfield voice name or id -> (id, voice_type). Anything unknown (e.g. an
    xAI voice_id left in beats.json) falls back to the default narrator."""
    global _voices
    if name_or_id and _UUID.fullmatch(name_or_id):
        return name_or_id, "preset"
    if _voices is None:
        _voices = {v["name"].lower(): v for v in _cli("voices", "list")["items"]}
    v = _voices.get((name_or_id or "").lower()) or _voices.get(DEFAULT_VOICE.lower())
    if not v:
        raise HiggsfieldError(f"voice '{DEFAULT_VOICE}' not found — see `higgsfield voices list`")
    if name_or_id and v["name"].lower() != name_or_id.lower():
        print(f"[higgsfield] voice '{name_or_id}' not on Higgsfield -> using {v['name']}")
    return v["id"], v.get("voice_type") or "preset"


def _media(value):
    """URL / local path / id -> something the CLI media flags accept."""
    if os.path.exists(os.path.expanduser(value)):
        return os.path.expanduser(value)
    if _UUID.fullmatch(value):
        return value
    if value.startswith("http"):
        ids = _UUID.findall(value.rsplit("/", 1)[-1])
        if ids and ("cloudfront.net" in value or "higgsfield" in value):
            return ids[-1]                       # a Higgsfield job/upload result
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(value.split("?")[0])[1] or ".bin")
        os.close(fd)
        subprocess.run(["/usr/bin/curl", "-sL", "--retry", "3", "-o", path, value], check=True)
        return path
    raise HiggsfieldError(f"can't use media input '{value[:80]}'")


def _flag_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class HiggsfieldProvider(Provider):
    name = "higgsfield"
    resubmit_stalled = False
    deadline_factor = 3
    music_needs_duration = True

    def _submit(self, model, prompt=None, **params):
        jt, extra = _job_type(model)
        schema = _schema(jt)
        params = {**extra, **params}
        if prompt is not None:
            params["prompt"] = prompt

        if jt == "text2speech_v2":
            vid, vtype = _voice(params.pop("voice_id", None))
            params.update(voice_id=vid, voice_type=vtype, variant=TTS_VARIANT)
        if "references" in params:                # seed-audio clone: inline base64 sample
            refs = params.pop("references")
            paths = []
            for r in refs:
                fd, path = tempfile.mkstemp(suffix=".mp3")
                with os.fdopen(fd, "wb") as f:
                    f.write(base64.b64decode(r["audio_data"]))
                paths.append(path)
            params["reference_audios"] = paths
        if params.get("sound") is False:
            params["sound"] = "off"
        if jt in START_FRAME_MODELS and isinstance(params.get("image"), str):
            params["start_image"] = params.pop("image")
        if jt == "gemini_omni" and "video" in params:
            params.pop("aspect_ratio", None)    # video-edit follows the input
        for k in ("ratio", "aspect_ratio"):
            if params.get(k) == "adaptive":     # Atlas "adaptive" == Higgsfield "auto"
                params[k] = "auto"

        args = ["generate", "create", jt]
        for k, v in params.items():
            if v is None:
                continue
            if k in MEDIA_FLAGS:
                for item in (v if isinstance(v, list) else [v]):
                    args += [MEDIA_FLAGS[k], _media(item)]
                continue
            k = RENAME.get(k, k)
            if k in schema:
                args += [f"--{k}", _flag_value(v)]
        ids = _cli(*args)
        return ids[0] if isinstance(ids, list) else ids.get("id")

    def submit_image(self, model, prompt, **params):
        return self._submit(model, prompt, **params)

    def submit_video(self, model, prompt, **params):
        return self._submit(model, prompt, **params)

    def submit_audio(self, model, **params):
        return self._submit(model, params.pop("prompt", None), **params)

    def remove_bg(self, model, image_url, **params):
        return self._submit(model, None, image=image_url)

    def get_status(self, job_id):
        try:
            d = _cli("generate", "get", job_id)
        except HiggsfieldError as e:
            return {"status": "pending", "output": None, "error": str(e)}   # transient CLI hiccup
        st = d.get("status")
        if st == "completed":
            return {"status": "completed", "output": d.get("result_url"), "error": None}
        if st in ("queued", "in_progress", "pending", "processing"):
            return {"status": "pending", "output": None, "error": None}
        return {"status": "failed", "output": None, "error": f"{st}: {d.get('error', '')}"}

    def upload(self, path):
        return _cli("upload", "create", os.path.expanduser(path))["url"]

    def download(self, url, dest):
        subprocess.run(["/usr/bin/curl", "-sL", "--retry", "3", "-o", dest, url], check=True)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise HiggsfieldError(f"download produced empty file: {url}")
        return dest
