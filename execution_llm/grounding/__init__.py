"""IR -> decision against a capability registry: `inputs` builds a domain's grounding inputs from the
supplied registry, `catalogs` holds the nurse and retail catalog fixtures, `signatures` the
comparator's per-domain GradingSignature data, `cover` matches each frame against the domain's
capability forms, `grounder` grounds a whole IR document, and `decide` applies the rule behind the
dataset's `gold_decision`."""
