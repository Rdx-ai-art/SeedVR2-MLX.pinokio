const backend = (name) =>
  `python -c "import importlib.metadata as m,shutil,subprocess;subprocess.run([shutil.which('uv') or 'uv','pip','install','${name}=='+m.version('mlx')],check=True)"`

module.exports = {
  run: [
    {
      // Pull the latest launcher + app source.
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      // Refresh Python dependencies in case requirements.txt changed.
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
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
    }
  ]
}
