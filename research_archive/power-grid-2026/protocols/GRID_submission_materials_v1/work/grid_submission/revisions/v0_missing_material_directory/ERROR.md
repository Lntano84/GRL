# Material preparation directory error

The initial offline material build wrote tables, code excerpts and figures, then stopped because the reproducibility destination directory had not yet been created. No simulator or model fit ran, no historical scientific files changed, and no material validation was claimed. The original builder is retained. The repair creates the directory and verifies that already generated JSON/CSV are byte-identical before resuming the same artifact preparation. Original experiment and manuscript versions remain untouched.
