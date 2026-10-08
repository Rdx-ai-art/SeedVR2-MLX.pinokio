module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      // Install the Python runtime + dependencies (mflux pulls in MLX, safetensors,
      // pillow, opencv, huggingface-hub, etc.). The model weights are NOT downloaded
      // here — they are fetched lazily in the web UI on first use, so a fresh install
      // stays fast and lets the user pick which variant to download.
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --no-deps mflux==0.20.0",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}
