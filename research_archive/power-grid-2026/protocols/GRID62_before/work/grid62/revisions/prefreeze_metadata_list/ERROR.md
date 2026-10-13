# Before-freeze metadata parser error

The first qualification attempt stopped before creating outputs/grid62 or reading any episode payload. An old run_manifest.json is a JSON list rather than a dict; calling obj.items() raised AttributeError. The original prepare.py is retained here. The repair accepts the historical list representation for conservative metadata date exclusion. No simulator, model, selection outcome, or date result had run.
