module.exports = {
  // Remove only the generated artifacts (venv, downloaded models, outputs).
  // The app/ source files (app.py, seedvr2_mlx.py, requirements.txt) are committed
  // and must be preserved — so we do NOT fs.rm the whole app folder here.
  run: [
    {
      method: "fs.rm",
      params: { path: "app/env" }
    },
    {
      method: "fs.rm",
      params: { path: "app/models" }
    },
    {
      method: "fs.rm",
      params: { path: "app/outputs" }
    }
  ]
}
