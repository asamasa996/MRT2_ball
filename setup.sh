#!/usr/bin/env bash
# ---------------------------------------------------------------
# runpod/setup.sh  —  RunPod 上の準備をまとめて行い、MRT2 サーバーを起動する
#
# 【自動運用】RunPod のテンプレートに環境変数 HF_TOKEN / MRT2_KEY を入れ、起動時コマンドでこのファイルを
#   取得して実行すれば、Pod を起動するだけで最後まで進む (入力なし)。詳しくは runpod/自動起動の設定.md
#
# 【small / base の選び方】環境変数 MRT2_SIZE で選ぶ (未設定なら small)。
#   small = 2.3億パラメーター (重み約1.1GB) … 通常はこちら。GPU は 12GB 以上
#   base  = 24億パラメーター (重み約9.2GB) … より高品質とされるが重い。1フレームが 40ms に収まるかは実測が必要
#   RunPod ではテンプレートを 2 つ作り、MRT2_SIZE だけ変える (runpod/自動起動の設定.md)。
#
# 使い方 (手動: Jupyter Lab の Terminal で):
#   1. このファイル (setup.sh) だけを /workspace にドラッグして置く (2回目以降は /workspace に残っているので不要)
#   2. bash /workspace/setup.sh
#   3. 途中で 2回だけ入力を求められる
#        - Hugging Face のトークン (重みのダウンロード用。画面には表示されない)
#        - 合言葉 MRT2_KEY        (手元の .env と同じもの。画面には表示されない)
#   HF トークンは保存しない (1回きり)。合言葉は /workspace/.mrt2_key に保存し、次回から入力不要。
#
# 途中で止まっても、もう一度 bash /workspace/setup.sh を実行すれば
# 終わっている所は飛ばして続きから進む。
# 2回目以降 (サーバーを Ctrl+C で止めた後) は bash /workspace/start.sh で起動だけできる。
# ---------------------------------------------------------------
set -euo pipefail

SIZE="${MRT2_SIZE:-small}"
case "$SIZE" in
  small) SIZE_GB="1.1"; MIN_BYTES=$((1 * 1024 * 1024 * 1024)); MIN_VRAM_MIB=12000 ;;
  base)  SIZE_GB="9.2"; MIN_BYTES=$((9 * 1024 * 1024 * 1024)); MIN_VRAM_MIB=12000 ;;
  *) echo "MRT2_SIZE は small か base にしてください (いまの値: $SIZE)"; exit 1 ;;
esac
echo "モデルの大きさ: $SIZE"

W=/workspace
export MAGENTA_HOME=$W/magenta-home
step() { echo; echo "==== [$1] $2"; }

mkdir -p "$W"
cd "$W"
# 改造パッチ (patch_mrt2_server.py) はこのファイルに埋め込んである。別ファイルを置く必要はない
cat > "$W/patch_mrt2_server.py" <<'MRT2_PATCH_EOF'
"""
patch_mrt2_server.py
---------------------------------------------------------------
babakarto/magenta-rt-local-windows の app_local.py を、
mrt2_swing から使えるように改造するパッチ。RunPod 上で1回だけ実行する。

使い方:
    python patch_mrt2_server.py jam/app_local.py

改造内容:
  1. 55秒で生成をリセットする制限を外す
     (元は無料GPU枠向けの名残。55秒ごとに曲の流れが初期化されてしまう)
     長さは環境変数 MRT_STREAM_SECONDS で変えられる (既定 36000秒 = 10時間)
  2. 15秒ごとに [timing] 1フレームの処理時間 (avg / p95 / max) を表示する
  3. 外部から接続できるように 0.0.0.0:7860 で起動する
  4. 画面なしで使える WebSocket 窓口 /mrt を追加する
       接続後、最初のメッセージで {"type":"hello","key":"<MRT2_KEY>"} を送る
       (環境変数 MRT2_KEY が空なら合言葉の確認はしない)
       以降:
         client -> server : {"type":"set", ...操作...}  (prompts / weights / cfg / temperature / notes / drumless / buffer など)
                            {"type":"buffer","lead":<ブラウザ側の再生待ち ms>}
         server -> client : バイナリ = [float32 LE: 1フレームの処理時間ms][int16 LE PCM 48kHz]
                            JSON     = {"type":"ready"} / {"type":"ended"}
---------------------------------------------------------------
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

MRT_ENDPOINT = '''

# ---- [mrt2_swing patch] headless WebSocket endpoint -----------------------
import hmac as _hmac
_MRT2_KEY = os.environ.get("MRT2_KEY", "")


@app.websocket("/mrt")
async def mrt_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
    except Exception:
        await websocket.close(code=1008); return
    if _MRT2_KEY and not _hmac.compare_digest(str(hello.get("key", "")), _MRT2_KEY):
        print("[mrt] rejected: bad key", flush=True)
        await websocket.close(code=1008); return
    sid = "mrt-" + uuid.uuid4().hex[:12]
    first = dict(hello.get("set") or {}); first["session_id"] = sid
    _apply_set(first)                                   # default conditioning so the worker can start
    audio_q = _q.Queue(maxsize=48); stop_ev = threading.Event()
    t = threading.Thread(target=_worker, args=(sid, audio_q, stop_ev), daemon=True)
    with _slock:
        _sessions[sid] = {"audio": audio_q, "stop": stop_ev}
    t.start()
    print(f"[mrt] session {sid} started", flush=True)
    await websocket.send_json({"type": "ready", "session_id": sid})
    loop = asyncio.get_event_loop()

    async def send_audio():
        try:
            while not stop_ev.is_set():
                try:
                    ab, fm = await loop.run_in_executor(None, lambda: audio_q.get(timeout=0.5))
                except Exception:
                    continue
                await websocket.send_bytes(struct.pack("<f", fm) + ab)
            try: await websocket.send_json({"type": "ended"})
            except Exception: pass
        finally:
            stop_ev.set()

    async def recv_control():
        try:
            while not stop_ev.is_set():
                msg = await websocket.receive_json()
                kind = msg.get("type")
                if kind == "set":
                    body = dict(msg); body["session_id"] = sid
                    try: _apply_set(body)
                    except Exception as e: print("[mrt set]", repr(e), flush=True)
                elif kind == "buffer":
                    try: _apply_buffer(sid, msg.get("lead", 0))
                    except Exception: pass
        finally:
            stop_ev.set()

    try:
        await asyncio.gather(send_audio(), recv_control(), return_exceptions=True)
    finally:
        stop_ev.set()
        with _slock:
            _sessions.pop(sid, None)
        print(f"[mrt] session {sid} closed", flush=True)
# ---- [/mrt2_swing patch] ---------------------------------------------------

'''

edits = [
    # 1) 55秒制限を外す
    ('''    while time.time() - t0 < 55.0:
''', '''    _stream_s = float(os.environ.get("MRT_STREAM_SECONDS", "36000"))  # [mrt2_swing patch] was 55.0
    while time.time() - t0 < _stream_s:
'''),
    # 2) 計測ログ
    ('''    buf_log, buf_t = [], t0
''', '''    buf_log, buf_t = [], t0
    frame_log = []  # [mrt2_swing patch]
'''),
    ('''        frame_ms = (time.time() - gen_t) * 1000.0 / CHUNK     # real per-frame inference time
''', '''        frame_ms = (time.time() - gen_t) * 1000.0 / CHUNK     # real per-frame inference time
        frame_log.append(frame_ms)  # [mrt2_swing patch]
'''),
    ('''            buf_log, buf_t = [], time.time()
''', '''            if frame_log:  # [mrt2_swing patch]
                s = sorted(frame_log)
                print("[timing] avg=%.1fms p95=%.1fms max=%.1fms n=%d (budget 40ms)" % (
                    sum(s) / len(s), s[int(len(s) * 0.95) - 1] if len(s) > 1 else s[0], s[-1], len(s)), flush=True)
                frame_log = []
            buf_log, buf_t = [], time.time()
'''),
    # 4) /mrt 窓口 (トップページ定義の直前に差し込む)
    ('''@app.get("/")
''', MRT_ENDPOINT + '''@app.get("/")
'''),
    # 3) 外部公開
    ('''app.launch(show_error=True, ssr_mode=False)''',
     '''app.launch(show_error=True, ssr_mode=False, server_name="0.0.0.0", server_port=7860)  # [mrt2_swing patch]'''),
]

if "[mrt2_swing patch]" in src:
    sys.exit("already patched")
for old, new in edits:
    if src.count(old) != 1:
        sys.exit("patch target not found: " + old.strip()[:60])
    src = src.replace(old, new)
open(path, "w", encoding="utf-8").write(src)
print("patched OK")
MRT2_PATCH_EOF

if [ -z "${MRT2_KEY:-}" ] && [ ! -s /workspace/.mrt2_key ] && [ ! -t 0 ]; then
  echo "MRT2_KEY が未設定で、入力もできない環境です。RunPod のテンプレートの環境変数に MRT2_KEY (16文字以上) を入れてください"; exit 1
fi

step 0/6 "GPU の確認"
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
echo "GPU: ${GPU_INFO:-不明}"
CC=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
if [ -n "$CC" ]; then
  CC_MAJOR=${CC%%.*}
  if [ "$CC_MAJOR" -lt 8 ]; then
    echo "このGPUは世代が古く (compute capability $CC)、MRT2 の計算形式 bf16 に対応していません。"
    echo "Pod を Terminate して、RTX 30xx / 40xx、A系、L系などに変えてください。"; exit 1
  fi
  if [ "$CC_MAJOR" -ge 10 ]; then
    echo "このGPUは新しすぎます (compute capability $CC。RTX 50xx / B200 など)。"
    echo "今の構成 (PyTorch 2.6 + CUDA 12.4) では動きません。Pod を Terminate して別のGPUにしてください。"; exit 1
  fi
fi
VRAM_MIB=$(echo "$GPU_INFO" | awk -F', ' '{print $3}' | tr -dc '0-9')
if [ -n "$VRAM_MIB" ] && [ "$VRAM_MIB" -lt "$MIN_VRAM_MIB" ]; then
  echo "このGPUのメモリ (${VRAM_MIB} MiB) では $SIZE を動かせません (${MIN_VRAM_MIB} MiB 以上が必要)。"
  echo "Pod を Terminate して、メモリの大きいGPUにしてください。"; exit 1
fi
echo "${GPU_INFO:-不明} / MRT2_SIZE=$SIZE" > "$W/gpu_used.txt"   # どのGPUで試したかの記録用
echo "$SIZE" > "$W/.mrt2_size"                                   # start.sh が読む

step 1/6 "コードの取得"
[ -d mrt-local ] || git clone https://github.com/babakarto/magenta-rt-local-windows mrt-local
[ -d jam ] || GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/magenta-community/magenta-rt-jam jam
[ -f jam/app.py ] || { echo "jam の取得に失敗しています。jam フォルダを消して (rm -rf /workspace/jam) やり直してください"; exit 1; }

step 2/6 "Python 環境の準備 (初回は 5〜10分)"
command -v uv >/dev/null || pip install -q uv
[ -d venv ] || uv venv -p 3.11 "$W/venv"
# shellcheck disable=SC1091
source "$W/venv/bin/activate"
if [ ! -f "$W/venv/.mrt2_ready" ]; then
  uv pip install -r mrt-local/requirements-local.txt
  uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  uv pip install setuptools            # triton (高速化の部品) が動くのに必要。10/6 に詰まった所
  touch "$W/venv/.mrt2_ready"
fi
python -c "import triton, triton.language; from triton.compiler.compiler import triton_key" \
  || { echo "triton が動きません。上のエラーを Claude に見せてください"; exit 1; }

step 3/6 "モデルの重み ($SIZE: 約${SIZE_GB}GB)"
CKPT="$MAGENTA_HOME/magenta-rt-v2/checkpoints/mrt2_${SIZE}.safetensors"
if [ ! -f "$CKPT" ] || [ "$(stat -c %s "$CKPT")" -lt "$MIN_BYTES" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    [ -t 0 ] || { echo "HF_TOKEN が未設定で、入力もできない環境です。RunPod のテンプレートの環境変数に HF_TOKEN を入れてください"; exit 1; }
    echo "Hugging Face のトークンを貼り付けて Enter (画面には表示されません):"
    read -rs HF_TOKEN; echo
  fi
  [ ${#HF_TOKEN} -ge 20 ] || { echo "トークンが短すぎます (${#HF_TOKEN}文字)。もう一度 setup.sh を実行してください"; exit 1; }
  export HF_TOKEN
  python mrt-local/download_weights.py "$SIZE" --home "$MAGENTA_HOME"
  unset HF_TOKEN
else
  echo "ダウンロード済みなので飛ばします"
fi

step 4/6 "改造パッチ"
if ! grep -q "mrt2_swing patch" jam/app_local.py 2>/dev/null; then
  cp "$W/mrt-local/app_local.py" "$W/jam/app_local.py"
  python "$W/patch_mrt2_server.py" "$W/jam/app_local.py"
else
  echo "適用済みなので飛ばします"
fi

step 5/6 "起動用スクリプト start.sh を作成"
cat > "$W/start.sh" <<'EOS'
#!/usr/bin/env bash
# MRT2 サーバーの起動だけを行う (setup.sh が済んでいること)
set -eu
export MAGENTA_HOME=/workspace/magenta-home
source /workspace/venv/bin/activate
# モデルの大きさ: ①環境変数 MRT2_SIZE → ②setup.sh が書いた /workspace/.mrt2_size → ③small
SIZE="${MRT2_SIZE:-$(cat /workspace/.mrt2_size 2>/dev/null || echo small)}"
# 合言葉の決め方: ①環境変数 MRT2_KEY → ②保存済み /workspace/.mrt2_key → ③入力 (入力したら②に保存)
KEYFILE=/workspace/.mrt2_key
if [ -z "${MRT2_KEY:-}" ] && [ -s "$KEYFILE" ]; then
  MRT2_KEY=$(cat "$KEYFILE"); echo "保存済みの合言葉を使います (変えたいときは rm $KEYFILE)"
fi
if [ -z "${MRT2_KEY:-}" ]; then
  echo "合言葉 (MRT2_KEY) を貼り付けて Enter (画面には表示されません。次回から保存分を使います):"
  read -rs MRT2_KEY; echo
  [ ${#MRT2_KEY} -ge 16 ] || { echo "合言葉が短すぎます (${#MRT2_KEY}文字。16文字以上)"; exit 1; }
  ( umask 077; printf %s "$MRT2_KEY" > "$KEYFILE" )
fi
export MRT2_KEY
cd /workspace/jam
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  (記録表に書いておく)"
echo "起動します (モデル: mrt2_$SIZE)。準備運転に 30〜60秒 (base はもっと長い)。'mrt2_$SIZE ready' が出たら手元から接続してください"
echo "手元の .env に MRT2_MODEL=mrt2_$SIZE と書いておくと、ログに使ったモデルが残ります"
echo "接続先: wss://${RUNPOD_POD_ID:-<PodのID>}-7860.proxy.runpod.net/mrt"
MRT_LOAD=$SIZE exec python app_local.py
EOS
chmod +x "$W/start.sh"

step 6/6 "MRT2 サーバーを起動"
exec bash "$W/start.sh"
