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
    }
  ]
}
