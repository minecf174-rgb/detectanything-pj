# Project Structure

```text
app/
  core/       shared infrastructure boundaries
  modules/    business domains, grouped by feature
  ml/         dataset building, training, prediction boundaries
  workers/    background job runtime and handlers
  shared/     enums and cross-module helpers

frontend/     frontend application placeholder
storage/      permanent local file storage for development
temp/         disposable training and auto-label workspaces
docs/         architecture and database references
tests/        automated tests
```

Each domain module can grow from the same basic shape:

```text
router.py      HTTP endpoints only
schemas.py     request/response DTOs
service.py     use cases and transaction boundaries
repository.py  persistence boundary
models.py      ORM models
```

