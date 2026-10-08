// The level-0 defaults of POST /api/runs (nanoscope/server/routes/runs.py RunRequest): the
// generated types list them as required, so a request spells them out.
export const RUN_DEFAULTS = {
  preset: "tinystories-5min",
  seed: 0,
  kwargs: {},
  compile: false,
  wandb: false,
} as const;
