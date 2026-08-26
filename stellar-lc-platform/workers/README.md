# Workers

Placeholder for asynchronous/background analysis workers.

A queue such as Celery/RQ/Arq can be introduced later when analyses become too
expensive for synchronous API requests. Do not add queue infrastructure before
the local scientific workflow is stable.
