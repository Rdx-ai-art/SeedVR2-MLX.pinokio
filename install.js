const backend = (name) =>
  `python -c "import importlib.metadata as m,shutil,subprocess;subprocess.run([shutil.which('uv') or 'uv','pip','install','${name}=='+m.version('mlx')],check=True)"`

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
      when: "{{platform !== 'darwin' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          backend("mlx-cuda-12")
        ]
      }
    },
    {
      when: "{{platform !== 'darwin' && gpu !== 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          backend("mlx-cpu")
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
