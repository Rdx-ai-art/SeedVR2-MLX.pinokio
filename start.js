module.exports = {
  // daemon: true keeps the server process alive after the run array completes.
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Uncomment + set to cap Metal memory, e.g. "12" for 12 GB.
          // int8 floor is ~12 GB (weights alone are ~7.5 GB); fp16 needs ~20 GB.
            // "SEEDVR2_MEM_LIMIT_GB": "10"
        },
        path: "app",
        message: [
          "python app.py --port {{port}}"
        ],
        on: [{
          // Capture the URL Gradio prints when the server is up.
          // The regex match object is passed to the next step as `input.event`.
          "event": "/(http:\\/\\/[0-9.:]+)/",

          // done: true moves to the next step while keeping the server shell alive.
          "done": true
        }]
      }
    },
    {
      // Set the local 'url' variable from the captured match; pinokio.js uses it
      // to surface the "Open Web UI" tab.
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
